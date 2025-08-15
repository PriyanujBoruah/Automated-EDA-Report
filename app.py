import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, confusion_matrix, roc_curve, auc
import json
import webview

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_key'] = 'a-very-secret-key-that-you-should-change'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload size

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_analysis(df, target_name):
    """Helper function to calculate correlations for a given target."""
    analysis_df = df.copy()
    for col in analysis_df.select_dtypes(include=['object', 'category']).columns:
        analysis_df[col], _ = pd.factorize(analysis_df[col])
    
    numeric_df = analysis_df.select_dtypes(include=['number'])
    if target_name not in numeric_df.columns:
        raise ValueError(f"Target column '{target_name}' is not numeric or could not be converted.")
    if len(numeric_df.columns) < 2:
        raise ValueError("Not enough numeric columns for correlation analysis.")
        
    correlations = numeric_df.corr()[target_name].sort_values(ascending=False)
    correlations = correlations.drop(target_name, errors='ignore')
    chart_labels = correlations.index.tolist()
    chart_data_raw = correlations.values.tolist()
    # Sanitize NaN values for JSON compatibility
    chart_data = [round(val, 3) if pd.notna(val) else None for val in chart_data_raw]
    return chart_labels, chart_data

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file upload and renders the initial results page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
                
                # --- UPDATED: Detailed Descriptive Statistics ---
                # Numerical Stats
                numeric_df = df.select_dtypes(include=np.number)
                if not numeric_df.empty:
                    stats_num = numeric_df.describe().T
                    stats_num['range'] = stats_num['max'] - stats_num['min']
                    stats_num['IQR'] = stats_num['75%'] - stats_num['25%']
                    stats_num['skew'] = numeric_df.skew()
                    stats_num['kurtosis'] = numeric_df.kurtosis()
                    stats_num['sum'] = numeric_df.sum()
                    stats_num['CV'] = stats_num['std'] / stats_num['mean']
                    stats_num['zeros_%'] = (numeric_df == 0).mean() * 100
                    numerical_stats_html = stats_num.to_html(classes='stats-table w-full text-sm', border=0, float_format='{:,.2f}'.format)
                else:
                    numerical_stats_html = "<p class='text-center text-gray-500'>No numeric columns found.</p>"

                # Categorical Stats
                categorical_df = df.select_dtypes(include=['object', 'category'])
                if not categorical_df.empty:
                    stats_cat = categorical_df.describe(include=['object', 'category']).T
                    categorical_stats_html = stats_cat.to_html(classes='stats-table w-full text-sm', border=0)
                else:
                    categorical_stats_html = "<p class='text-center text-gray-500'>No categorical columns found.</p>"
                # --- END UPDATE ---

                missing_counts = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Missing Count': missing_counts,
                    'Missing Percentage (%)': (missing_counts / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage (%)', ascending=False)
                
                if missing_df.empty:
                    missing_values_html = "<p class='text-center text-gray-500 p-4'>🎉 No missing values found in this dataset!</p>"
                else:
                    missing_values_html = missing_df.to_html(classes='quality-table w-full text-sm', border=0)

                default_target = df.columns[-1]
                all_columns = df.columns.tolist()
                numeric_columns = numeric_df.columns.tolist()

                chart_labels, chart_data = perform_analysis(df, default_target)

                preview_html = df.head(500).to_html(classes='stats-table w-full text-sm', border=0, index=False)

                return render_template('results.html',
                                       target_variable=default_target,
                                       columns=all_columns,
                                       numeric_columns=numeric_columns,
                                       chart_labels=json.dumps(chart_labels),
                                       chart_data=json.dumps(chart_data),
                                       data_preview=preview_html,
                                       filename=filename,
                                       numerical_stats_table=numerical_stats_html,
                                       categorical_stats_table=categorical_stats_html,
                                       missing_values_table=missing_values_html)

            except Exception as e:
                flash(f'An error occurred: {e}')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)

    return render_template('index.html')

@app.route('/update_chart', methods=['POST'])
def update_chart():
    data = request.get_json()
    filename, target_name = data.get('filename'), data.get('target_column')
    if not filename or not target_name: return jsonify({'error': 'Missing filename or target column'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found, please re-upload'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        chart_labels, chart_data = perform_analysis(df, target_name)
        return jsonify({'labels': chart_labels, 'data': chart_data, 'target_variable': target_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_histogram', methods=['POST'])
def get_histogram():
    data = request.get_json()
    filename, column_name = data.get('filename'), data.get('column_name')
    if not filename or not column_name: return jsonify({'error': 'Missing filename or column name'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found, please re-upload'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        if column_name not in df.columns: return jsonify({'error': 'Column not found'}), 404
        
        column_data = df[column_name].dropna()
        counts, bin_edges = np.histogram(column_data, bins=20)
        labels = [f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]
        return jsonify({'labels': labels, 'data': counts.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_scatter_data', methods=['POST'])
def get_scatter_data():
    data = request.get_json()
    filename, x_column, y_column = data.get('filename'), data.get('x_column'), data.get('y_column')
    if not all([filename, x_column, y_column]): return jsonify({'error': 'Missing filename or column names'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found, please re-upload'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        if x_column not in df.columns or y_column not in df.columns: return jsonify({'error': 'One or more columns not found'}), 404
        
        scatter_df = df[[x_column, y_column]].dropna()
        points = [{'x': row[x_column], 'y': row[y_column]} for _, row in scatter_df.iterrows()]
        return jsonify({'data': points})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_feature_importance', methods=['POST'])
def get_feature_importance():
    data = request.get_json()
    filename, target_name, model_type = data.get('filename'), data.get('target_column'), data.get('model_type')
    params = data.get('params', {})
    if not all([filename, target_name, model_type]): return jsonify({'error': 'Missing filename, target, or model type'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found, please re-upload'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        
        X = df.drop(columns=[target_name])
        y = df[target_name]

        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        X = X.fillna(X.mean())

        is_classification = y.dtype == 'object' or y.nunique() <= 10
        
        response_data = {}
        
        if is_classification:
            y_series, class_names = pd.factorize(y)
            y = y_series
            if model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth')) if params.get('max_depth') else None,
                    random_state=42, n_jobs=-1
                )
            else:
                model = LGBMClassifier(
                    n_estimators=int(params.get('n_estimators', 100)),
                    learning_rate=float(params.get('learning_rate', 0.1)),
                    num_leaves=int(params.get('num_leaves', 31)),
                    random_state=42, n_jobs=-1
                )
        else:
            if model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth')) if params.get('max_depth') else None,
                    random_state=42, n_jobs=-1
                )
            else:
                model = LGBMRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    learning_rate=float(params.get('learning_rate', 0.1)),
                    num_leaves=int(params.get('num_leaves', 31)),
                    random_state=42, n_jobs=-1
                )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {}
        if is_classification:
            metrics['Accuracy'] = round(accuracy_score(y_test, y_pred), 3)
            metrics['F1-Score (Weighted)'] = round(f1_score(y_test, y_pred, average='weighted'), 3)
            metrics['Precision (Weighted)'] = round(precision_score(y_test, y_pred, average='weighted'), 3)
            metrics['Recall (Weighted)'] = round(recall_score(y_test, y_pred, average='weighted'), 3)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_labels = [str(c) for c in class_names]
            response_data['confusion_matrix'] = {'matrix': cm.tolist(), 'labels': cm_labels}

            if len(class_names) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                response_data['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(roc_auc, 3)}

        else:
            metrics['R-Squared'] = round(r2_score(y_test, y_pred), 3)
            metrics['Mean Absolute Error (MAE)'] = round(mean_absolute_error(y_test, y_pred), 3)
            metrics['Mean Squared Error (MSE)'] = round(mean_squared_error(y_test, y_pred), 3)

        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        importances_list = importances.values.tolist()
        sanitized_importances = [val if pd.notna(val) else None for val in importances_list]

        response_data.update({
            'labels': importances.index.tolist(), 
            'data': sanitized_importances,
            'metrics': metrics
        })
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_heatmap_data', methods=['POST'])
def get_heatmap_data():
    data = request.get_json()
    filename = data.get('filename')
    if not filename: return jsonify({'error': 'Missing filename'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return jsonify({'error': 'Not enough numeric columns for a heatmap'}), 400
        
        corr_matrix = numeric_df.corr()
        
        heatmap_data = []
        for r_index, row in corr_matrix.iterrows():
            for c_index, value in row.items():
                sanitized_value = round(value, 3) if pd.notna(value) else None
                heatmap_data.append({'x': c_index, 'y': r_index, 'v': sanitized_value})
        
        return jsonify({'data': heatmap_data, 'labels': corr_matrix.columns.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_unsupervised_model', methods=['POST'])
def run_unsupervised_model():
    data = request.get_json()
    filename, model_type = data.get('filename'), data.get('model_type')
    if not filename or not model_type: return jsonify({'error': 'Missing filename or model type'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        numeric_df = df.select_dtypes(include=['number']).dropna()
        if numeric_df.shape[1] < 2:
            return jsonify({'error': 'Unsupervised models require at least 2 numeric columns with no missing values.'}), 400

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        
        if model_type == 'pca':
            explained_variance = pca.explained_variance_ratio_.tolist()
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            points = [{'x': (row['PC1'] if pd.notna(row['PC1']) else None), 'y': (row['PC2'] if pd.notna(row['PC2']) else None)} for _, row in pca_df.iterrows()]
            return jsonify({'data': points, 'explained_variance': explained_variance})

        elif model_type == 'kmeans':
            n_clusters = int(data.get('n_clusters', 3))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
        
        elif model_type == 'dbscan':
            eps = float(data.get('eps', 0.5))
            min_samples = int(data.get('min_samples', 5))
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(scaled_data)

        cluster_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        cluster_df['cluster'] = clusters
        
        unique_clusters = sorted(cluster_df['cluster'].unique())
        grouped_data = []
        for cluster_id in unique_clusters:
            cluster_points = cluster_df[cluster_df['cluster'] == cluster_id]
            points = [{'x': (row['PC1'] if pd.notna(row['PC1']) else None), 'y': (row['PC2'] if pd.notna(row['PC2']) else None)} for _, row in cluster_points.iterrows()]
            grouped_data.append({'cluster_id': int(cluster_id), 'points': points})
        
        return jsonify({'grouped_data': grouped_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_elbow_data', methods=['POST'])
def get_elbow_data():
    data = request.get_json()
    filename = data.get('filename')
    if not filename: return jsonify({'error': 'Missing filename'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        numeric_df = df.select_dtypes(include=['number']).dropna()
        if numeric_df.shape[1] < 2:
            return jsonify({'error': 'Elbow method requires at least 2 numeric columns with no missing values.'}), 400

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        inertia_scores = []
        cluster_range = range(2, 11)
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia_scores.append(kmeans.inertia_)
        
        sanitized_scores = [score if pd.notna(score) else None for score in inertia_scores]
        return jsonify({'cluster_numbers': list(cluster_range), 'inertia_scores': sanitized_scores})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create a pywebview window and start the application
    webview.create_window('Automated EDA Report', app)
    webview.start()
