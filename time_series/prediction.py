import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the original dataset
df = pd.read_csv('/content/corrected_temperature_humidity_dataset.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# Get the latest timestamp in the dataset
latest_timestamp = df['Timestamp'].max()
print(f"Latest timestamp in the dataset: {latest_timestamp}")

# Generate the next 10 days of timestamps for each building/floor/flat combination
unique_locations = df[['Building', 'Floor', 'Flat']].drop_duplicates()
future_dates = pd.date_range(start=latest_timestamp + timedelta(hours=1), periods=10*24, freq='h')  # 10 days of hourly data

# Create a DataFrame for future predictions
future_rows = []
for _, location in unique_locations.iterrows():
    for future_date in future_dates:
        future_rows.append({
            'Timestamp': future_date,
            'Building': location['Building'],
            'Floor': location['Floor'],
            'Flat': location['Flat'],
        })

future_df = pd.DataFrame(future_rows)

# Make sure we have the Anomaly column in future_df (will be overwritten later)
future_df['Anomaly'] = 0  # Placeholder

# Calculate average temperature and humidity for each location
location_stats = df.groupby(['Building', 'Floor', 'Flat']).agg({
    'Temperature': ['mean', 'std'],
    'Humidity': ['mean', 'std']
}).reset_index()

# Flatten the column names
location_stats.columns = ['Building', 'Floor', 'Flat', 'Temp_Mean', 'Temp_Std', 'Hum_Mean', 'Hum_Std']

# Create a dictionary for faster lookups
location_dict = {}
for _, row in location_stats.iterrows():
    key = (row['Building'], row['Floor'], row['Flat'])
    location_dict[key] = {
        'temp_mean': row['Temp_Mean'],
        'temp_std': row['Temp_Std'],
        'hum_mean': row['Hum_Mean'],
        'hum_std': row['Hum_Std']
    }

# Synthesize temperature and humidity data for future timestamps
for idx, row in future_df.iterrows():
    key = (row['Building'], row['Floor'], row['Flat'])
    stats = location_dict.get(key, {'temp_mean': 25, 'temp_std': 3, 'hum_mean': 50, 'hum_std': 5})
    
    # Add time-based patterns
    hour = row['Timestamp'].hour
    day_of_week = row['Timestamp'].weekday()
    
    # Temperature: Higher during day, lower at night
    time_factor = np.sin(hour * np.pi / 12) * 3  # Peaks at noon, lowest at midnight
    day_factor = 1 if day_of_week < 5 else 0  # Weekday vs weekend
    
    temp_base = stats['temp_mean'] + time_factor + day_factor
    temp_noise = np.random.normal(0, stats['temp_std'] * 0.3)  # Reduce noise amplitude
    
    # Humidity: Higher during night/early morning, lower during day
    hum_time_factor = -np.sin(hour * np.pi / 12) * 5  # Opposite of temperature pattern
    hum_base = stats['hum_mean'] + hum_time_factor
    hum_noise = np.random.normal(0, stats['hum_std'] * 0.3)
    
    future_df.loc[idx, 'Temperature'] = temp_base + temp_noise
    future_df.loc[idx, 'Humidity'] = hum_base + hum_noise

# Now we have a complete future dataframe with synthesized Temperature and Humidity
# We can proceed with feature engineering

# Create time-based features for the future data
future_df['hour'] = future_df['Timestamp'].dt.hour
future_df['day'] = future_df['Timestamp'].dt.day
future_df['month'] = future_df['Timestamp'].dt.month
future_df['day_of_week'] = future_df['Timestamp'].dt.dayofweek
future_df['is_weekend'] = future_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Get the last few days of historical data to help with lagged features
days_to_include = 7  # Include last 7 days of historical data
historical_cutoff = latest_timestamp - timedelta(days=days_to_include)
recent_df = df[df['Timestamp'] >= historical_cutoff].copy()

# Add the same time-based features to the recent data
recent_df['hour'] = recent_df['Timestamp'].dt.hour
recent_df['day'] = recent_df['Timestamp'].dt.day
recent_df['month'] = recent_df['Timestamp'].dt.month
recent_df['day_of_week'] = recent_df['Timestamp'].dt.dayofweek
recent_df['is_weekend'] = recent_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Combine recent and future data for feature engineering
combined_df = pd.concat([recent_df, future_df], ignore_index=True)
combined_df = combined_df.sort_values(['Building', 'Floor', 'Flat', 'Timestamp'])

# Create lagged features
for lag in range(1, 4):
    combined_df[f'temp_lag_{lag}'] = combined_df.groupby(['Building', 'Floor', 'Flat'])['Temperature'].transform(lambda x: x.shift(lag))
    combined_df[f'hum_lag_{lag}'] = combined_df.groupby(['Building', 'Floor', 'Flat'])['Humidity'].transform(lambda x: x.shift(lag))

# Create rolling window features
combined_df['temp_rolling_mean_3'] = combined_df.groupby(['Building', 'Floor', 'Flat'])['Temperature'].transform(lambda x: x.rolling(window=3).mean())
combined_df['hum_rolling_mean_3'] = combined_df.groupby(['Building', 'Floor', 'Flat'])['Humidity'].transform(lambda x: x.rolling(window=3).mean())

# Create temperature and humidity differentials
combined_df['temp_diff'] = combined_df['Temperature'] - combined_df['temp_lag_1']
combined_df['hum_diff'] = combined_df['Humidity'] - combined_df['hum_lag_1']

# One-hot encode the Building category
combined_df = pd.get_dummies(combined_df, columns=['Building'])

# Split back into recent and future data
future_df = combined_df[combined_df['Timestamp'] > latest_timestamp].copy()

# Now we can handle NaN values for the future data
# Instead of just dropping NaN rows, we'll fill them with appropriate values
future_df = future_df.fillna({
    'temp_lag_1': future_df['Temperature'],
    'temp_lag_2': future_df['Temperature'],
    'temp_lag_3': future_df['Temperature'],
    'hum_lag_1': future_df['Humidity'],
    'hum_lag_2': future_df['Humidity'],
    'hum_lag_3': future_df['Humidity'],
    'temp_rolling_mean_3': future_df['Temperature'],
    'hum_rolling_mean_3': future_df['Humidity'],
    'temp_diff': 0,
    'hum_diff': 0
})

# Check if we have data in the future dataframe
print(f"Number of data points for future prediction: {len(future_df)}")

# Define features (same as in the training code)
features = ['hour', 'day', 'month', 'day_of_week', 'is_weekend', 
            'Floor', 'Flat', 'Temperature', 'Humidity',
            'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
            'hum_lag_1', 'hum_lag_2', 'hum_lag_3',
            'temp_rolling_mean_3', 'hum_rolling_mean_3',
            'temp_diff', 'hum_diff'] + [col for col in combined_df.columns if col.startswith('Building_')]

# Ensure all features are available
missing_features = set(features) - set(future_df.columns)
if missing_features:
    print(f"Missing features: {missing_features}")
    # Use only available features
    features = [f for f in features if f in future_df.columns]

# Check for remaining NaN values
nan_count = future_df[features].isna().sum().sum()
if nan_count > 0:
    print(f"Warning: {nan_count} NaN values found in feature columns")
    print(future_df[features].isna().sum())
    # Fill any remaining NaN values with column means
    future_df[features] = future_df[features].fillna(future_df[features].mean())

# Prepare the data for prediction
X_future = future_df[features]

# Scale the data using the same scaler
X_future_scaled = scaler.transform(X_future)

# Make predictions
future_df['Predicted_Anomaly'] = model.predict(X_future_scaled)
future_df['Predicted_Probability'] = model.predict_proba(X_future_scaled)[:, 1]

# Analyze the predictions
print("\nForecast Summary:")
print(f"Total number of forecasted data points: {len(future_df)}")
print(f"Predicted anomalies in the next 10 days: {future_df['Predicted_Anomaly'].sum()}")
print(f"Anomaly rate: {future_df['Predicted_Anomaly'].mean()*100:.2f}%")

# Calculate anomaly forecasts by day
daily_forecast = future_df.groupby(future_df['Timestamp'].dt.date)['Predicted_Anomaly'].agg(['mean', 'sum', 'count']).reset_index()
daily_forecast['anomaly_percentage'] = daily_forecast['mean'] * 100

# Visualize the forecasted anomalies
plt.figure(figsize=(14, 6))
plt.plot(daily_forecast['Timestamp'], daily_forecast['anomaly_percentage'], marker='o', linestyle='-', linewidth=2)
plt.title('Forecasted Daily Anomaly Rate for Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Predicted Anomaly Rate (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Anomalies by hour of day
hourly_forecast = future_df.groupby(future_df['hour'])['Predicted_Anomaly'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.bar(hourly_forecast['hour'], hourly_forecast['Predicted_Anomaly'] * 100)
plt.title('Predicted Anomaly Rate by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Anomaly Rate (%)')
plt.grid(axis='y')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

# Show the top 10 highest risk locations and times
print("\nTop 10 Time Periods with Highest Anomaly Risk:")
high_risk = future_df.sort_values('Predicted_Probability', ascending=False).head(10)
for idx, row in high_risk.iterrows():
    building_cols = [col for col in row.index if col.startswith('Building_') and row[col] == 1]
    building_name = building_cols[0].replace('Building_', '') if building_cols else 'Unknown'
    print(f"- {row['Timestamp']}: Building {building_name}, Floor {int(row['Floor'])}, Flat {int(row['Flat'])}")
    print(f"  Temperature: {row['Temperature']:.2f}, Humidity: {row['Humidity']:.2f}")
    print(f"  Anomaly Probability: {row['Predicted_Probability']:.4f}")

# Create anomaly forecast summary table for each day
print("\nDaily Anomaly Forecast Summary:")
print(daily_forecast[['Timestamp', 'sum', 'count', 'anomaly_percentage']].rename(
    columns={'Timestamp': 'Date', 'sum': 'Total Anomalies', 'count': 'Data Points', 'anomaly_percentage': 'Anomaly Rate (%)'}
).to_string(index=False))
