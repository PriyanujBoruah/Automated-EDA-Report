# Automated EDA Report

An interactive desktop application that performs a comprehensive Exploratory Data Analysis (EDA) on your datasets. Built with Python, Flask, and PyWebview, this tool allows you to upload a CSV or XLSX file and instantly generate key statistics, visualizations, and machine learning insights without writing any code.

## ✨ Features

This application provides a rich, interactive report with the following sections:

  * **Descriptive Statistics** 📈: Get a complete statistical summary for both numerical and categorical columns, including metrics like mean, standard deviation, skewness, kurtosis, and unique value counts.
  * **Data Quality Analysis** 🧼: Quickly identify missing values with a clear summary table showing the count and percentage of missing data for each column.
  * **Interactive Visualizations** 🎨:
      * **Correlation Chart**: Instantly see how different numerical features correlate with your target variable.
      * **Distribution Histograms**: Visualize the distribution of any numerical column.
      * **Scatter Plots**: Explore relationships between any two numerical features.
      * **Correlation Heatmap**: Generate a full heatmap to understand the relationships between all numerical variables at a glance.
  * **Supervised Learning** 🧠:
      * Automatically identifies if your task is a **classification** or **regression** problem.
      * Runs **Random Forest** and **LightGBM** models to determine feature importance.
      * Displays key performance metrics like **Accuracy, F1-Score, R-Squared,** and **MAE**.
      * For classification tasks, it also generates a **Confusion Matrix** and **ROC Curve**.
  * **Unsupervised Learning** 🤖:
      * Discover hidden patterns in your data using clustering algorithms.
      * **K-Means Clustering**: Includes an "Elbow Method" helper to find the optimal number of clusters.
      * **DBSCAN**: Identifies density-based clusters and highlights noise points.
      * **PCA (Principal Component Analysis)**: Reduces dimensionality and visualizes your data in 2D space.

## 🛠️ How It Works

This application is a hybrid of a web application and a desktop application, built with a modern Python stack:

  * **Backend**: A **Flask** web server handles file uploads, data processing, and all the machine learning model training.
  * **Frontend**: The user interface is built with standard **HTML**, **Tailwind CSS**, and **Chart.js** for interactive visualizations.
  * **Desktop Wrapper**: **PyWebview** wraps the Flask application in a lightweight, native GUI window, creating a seamless desktop experience.

## 🚀 Getting Started

Follow these instructions to run the application on your local machine.

### 1\. Create a Virtual Environment

It is highly recommended to create a virtual environment to keep the project's dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2\. Install Dependencies

First, create a `requirements.txt` file by running the following command in your activated virtual environment:

```bash
pip freeze > requirements.txt
```

Then, install all the necessary libraries:

```bash
pip install -r requirements.txt
```

*(If you don't have a `requirements.txt` file yet, you can install the packages directly: `pip install pandas flask scikit-learn lightgbm pywebview numpy`)*

### 3\. Run the Application

Once the dependencies are installed, you can start the application by running the `app.py` script:

```bash
python app.py
```

This will launch the desktop application window.

## 📦 Building an Executable (.exe)

You can package this application into a single standalone executable file using **PyInstaller**.

1.  **Install PyInstaller**:
    ```bash
    pip install pyinstaller
    ```
2.  **Run the Build Command**:
    From your project directory, run the following command. The `--add-data` flag is crucial for including your HTML templates.
    ```bash
    pyinstaller --onefile --add-data "templates;templates" app.py
    ```
3.  **Find Your Executable**:
    The final `app.exe` file will be located in the newly created `dist` folder. You can distribute and run this file on other Windows machines without needing a Python installation.

## 📚 Dependencies

The core libraries used in this project are:

  * Flask
  * PyWebview
  * Pandas
  * NumPy
  * Scikit-learn
  * LightGBM

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
