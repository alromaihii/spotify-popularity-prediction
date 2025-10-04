import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = 'data/Final_data.csv'
artist_data = pd.read_csv(file_path)

# Data Preparation
time_series_data = artist_data.groupby(['Date', 'artist'])['points'].sum().reset_index()
time_series_data['Date'] = pd.to_datetime(time_series_data['Date'])

# Fill missing dates for each artist
artists = time_series_data['artist'].unique()
filled_data = []

for artist in artists:
    artist_data_filtered = time_series_data[time_series_data['artist'] == artist]
    artist_data_full = artist_data_filtered.set_index('Date').asfreq('D', fill_value=0).reset_index()
    artist_data_full['artist'] = artist
    filled_data.append(artist_data_full)

time_series_data_full = pd.concat(filled_data)

# Feature Engineering
time_series_data_full['day'] = time_series_data_full['Date'].dt.day
time_series_data_full['month'] = time_series_data_full['Date'].dt.month
time_series_data_full['year'] = time_series_data_full['Date'].dt.year

# Encode artist names
label_encoder = LabelEncoder()
time_series_data_full['artist_encoded'] = label_encoder.fit_transform(time_series_data_full['artist'])

# Define model parameters
model_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Forecasting for Each Artist
forecast_results = []

for artist in artists:
    artist_data_filtered = time_series_data_full[time_series_data_full['artist'] == artist]
    if len(artist_data_filtered) < 12:
        forecast_results.append({
            'Artist': artist,
            'Total Points 2024': 0
        })
        continue

    # Prepare training data
    X = artist_data_filtered[['day', 'month', 'year']]
    y = artist_data_filtered['points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM model
    model = LGBMRegressor(**model_params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Artist: {artist}, MSE: {mse}")

    # Predict points for 2024
    future_dates = pd.date_range(start='2024-01-01', end='2024-12-31')
    future_features = pd.DataFrame({
        'day': future_dates.day,
        'month': future_dates.month,
        'year': future_dates.year
    })
    future_predictions = model.predict(future_features)
    total_points_2024 = future_predictions.sum()

    forecast_results.append({
        'Artist': artist,
        'Total Points 2024': total_points_2024
    })

# Combine all results into a DataFrame
forecast_summary = pd.DataFrame(forecast_results)
forecast_summary.to_csv('LightGBM_Predictions_2024_Basic.csv', index=False)

# Find the most popular artist of 2024
most_popular_artist = forecast_summary.sort_values(by='Total Points 2024', ascending=False).iloc[0]
print(f"Most Popular Artist of 2024: {most_popular_artist['Artist']}")
print(f"Total Points: {most_popular_artist['Total Points 2024']}")
