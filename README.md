# Spotify Analysis Project

This project involves analyzing Spotify's Top 200 playlist data to predict popular songs and artists, applying clustering techniques, feature engineering, and classification algorithms.

---

## Directory Structure

spotify-popularity-prediction/
├── data/
│ ├──ARIMA_Appearances.csv
│ ├──ARIMA_Date.csv
│ ├──ARIMA_Weighted_pts.csv
│ ├──Artist_pop.csv
│ ├──Artist_Pop_App.csv
│ ├──Final_data.csv
│ ├──LightGBM_Predictions_2024.csv
│ ├──prophet_App.zip
│ ├──Prophet_Pts.csv
│ ├──Prophet_Weighted_pts.csv
│ ├──spotify_data_with_features.zip
│ ├── XGBoost_Predictions_2024.csv
├── report/
│ ├── final_report.pdf
├── src/
│ ├──ARIMA.ipynb
│ ├──Date_Preprocess.ipynb
│ ├──decision_tree.py
│ ├──K-means.py
│ ├──kmeans_minibatch.py
│ ├──LGBM.py
│ ├──logistic_regression.py
│ ├──Month_Preporcess.ipynb
│ ├──Plot.ipynb
│ ├──plot_features_trend.py
│ ├──Prophet.ipynb
│ ├──randomForest.py
│ ├──XGB.py
├── README.md
├── requirements.txt

**Files in src/**

1. **k-means.py**
   This script implements K-Means clustering for dimensionality-reduced Spotify song data. It evaluates clustering performance using the Silhouette score and visualizes cluster assignments.This file will output the Silhouette score and will plot the k-means.

2. **randomForest.py**  
   This script performs Random Forest classification to predict song popularity based on engineered features. It includes model training, evaluation metrics, and feature importance analysis. This file will output the Accuracy ,Precision ,Recall
   ,F1-Score ,Cross-Validation Metrics (5-fold):{Mean Accuracy,
   Mean Precision,Mean Recall, Mean F1-Score}.

3. **XGB.py**
   This script uses XGBoost to forecast music artists' popularity in 2024 based on historical data. It processes a dataset (Final_data.csv), fills missing dates, extracts temporal features (day, month, year), and encodes artist names. Each artist's data is used to train an XGBoost regressor, predict daily points for 2024, and compute total annual points. Results are saved in XGBoost_Predictions_2024.csv, with the most popular artist of 2024 identified and printed.

4. **LGBM.py**
   This script leverages LightGBM to forecast the popularity of music artists in 2024. It processes historical data from Final_data.csv by filling gaps in dates and engineering features like day, month, and year. Artist names are transformed into numerical labels using LabelEncoder. For each artist, a LightGBM model is trained to predict daily points for 2024, which are then aggregated into yearly totals. The predictions are stored in LightGBM_Predictions_2024_Basic.csv, and the script highlights the artist expected to be the most popular in 2024.

5. **logistic_regression.py**
   Train a logistic regression model using the dataset, outputting the accuracy score, classification report, and feature importance plot.

6. **Date_Preprocess.ipynb**
   Preprocess the data for date based forecasting. It drops the unnecessary columns, drops the artist who are only present in 2023, combine multiple instance of the artist on a same day as one but take mean of all the values and sum for 'points', drops the artist who have less than 24 data points before 2023 as the forecasting model in tested on 2023 data and certain number of data points are required for forecasting.

7. **Month_Preprocess.ipynb**
   Preprocess the data for month based forecasting. It drops the unnecessary columns, groups the artist for each month of the year and calculate the total number of appearance that artist made in a single month and calulates the mean of all the values.

8. **ARIMA.ipynb**
   ARIMA is used to forecast the popularity of different music artists in 2024 based on historical data.It is applied for both month based and date based analysis to forecast the popular artist of 2024. It processes a dataset, fills missing dates to ensure consistent daily time series, and aggregates points by artist and date. Artists with insufficient data are skipped, and (p,d,q) values are assigned based on the trails made on some popular artist as using auto_arima provided the same result as the default value. Results include the total forecasted points for each artist, with the top 20 most popular artist of 2024 identified and printed. Results are stored ARIMA_Appearances.csv, ARIMA_Date.csv and ARIMA_Weighted_pts.csv

9. **Prophet.ipynb**
   Forecasts the popularity of different music artists in 2024 based on historical data. It is applied for both month based(Artist_Pop_App.csv) and date based (Final_data.csv) analysis to forecast the popular artist of 2024. It processes a dataset, fills missing dates to ensure consistent daily time series, and aggregates points by artist and date. Artists with insufficient data are skipped and using regressor provided results with more error. Results include the total forecasted points for each artist, with the top 20 most popular artist of 2024 identified and printed. Results are stored in prophet_App.csv, Prophet_Pts.csv and Prophet_Weighted_pts.csv

10. **Plot.ipynb**
    This plots all the top 20 artist of each models previously mentioned. This also plots the forecasting Graph for the Weeknd with prophet and LightGBM.

## Prerequisites

To run this project, you need:

- Python 3.8 or higher
- `pip` installed on your system
- Jupyter notebook installed on your system

## Environment Setup

Follow these steps to set up the environment:

1. **Extract the Zip File:**
   Unzip the  folder into your working directory.

2. **Set Up a Python Virtual Environment:**
   Set Up a Python Virtual Environment: Create and activate a virtual environment to isolate the project dependencies:

   On Linux/macOS:
   python3 -m venv env
   source env/bin/activate

   On Windows:
   python -m venv env
   .\env\Scripts\activate

3. **Install Dependencies: Use the requirements.txt file to install the required packages:**

   pip install -r requirements.txt

## How to Run

1. **Run Clustering Analysis:**
   Execute the k-means.py script: src/k-means.py
   Execute the kmeans_minibatch.py script: src/kmeans_minibatch.py
   Execute the gmm_clustering.py script: src/gmm_clustering.py

2. **Run Classification Analysis:**  
   Execute the randomForest.py script: src/randomForest.py

3. **Run Data Preprocess for Prediction:**
   Execute the Date_Preprocess.ipynb script: src/Date_Preprocess.ipynb
   Execute the Month_Preprocess.ipynb script: src/Month_Preprocess.ipynb

4. **Run Prediction Analysis:**  
   Execute the ARIMA.ipynb script: src/ARIMA.ipynb
   Execute the Prophet.ipynb script: src/Prophet.ipynb
   Execute the XGB.py script: src/XGB.py
   Execute the LGBM.py script: src/LGBM.py

5. **Run Visualization Analysis:**  
   Execute the Plot.ipynb script: src/Plot.ipynb

## Notes

Ensure the data files are placed in the data/ directory as expected.
The project outputs include visualizations and metrics, which are explained in the report.
