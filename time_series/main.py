import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from datetime import timedelta

# Load the dataset again
df = pd.read_csv('/content/corrected_temperature_humidity_dataset.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')  # Ensure data is sorted by time

# Feature engineering
# Extract time-based features
df['hour'] = df['Timestamp'].dt.hour
df['day'] = df['Timestamp'].dt.day
df['month'] = df['Timestamp'].dt.month
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Create lagged features for temperature and humidity
for lag in range(1, 4):  # Create 3 lags
    df[f'temp_lag_{lag}'] = df.groupby(['Building', 'Floor', 'Flat'])['Temperature'].transform(lambda x: x.shift(lag))
    df[f'hum_lag_{lag}'] = df.groupby(['Building', 'Floor', 'Flat'])['Humidity'].transform(lambda x: x.shift(lag))

# Create rolling window features (average of last 3 observations)
# Use transform instead of rolling directly
df['temp_rolling_mean_3'] = df.groupby(['Building', 'Floor', 'Flat'])['Temperature'].transform(lambda x: x.rolling(window=3).mean())
df['hum_rolling_mean_3'] = df.groupby(['Building', 'Floor', 'Flat'])['Humidity'].transform(lambda x: x.rolling(window=3).mean())

# Create temperature and humidity differentials
df['temp_diff'] = df['Temperature'] - df['temp_lag_1']
df['hum_diff'] = df['Humidity'] - df['hum_lag_1']

# Drop NaN values created by lagging
df = df.dropna()

# One-hot encode the Building category
df = pd.get_dummies(df, columns=['Building'])

# Split the data: use all data up to a certain point for training, and the rest for testing
# This is more appropriate for time series than random splitting
try:
    # First approach: use the last 10 days for testing
    split_date = df['Timestamp'].max() - timedelta(days=10)
    train_df = df[df['Timestamp'] <= split_date]
    test_df = df[df['Timestamp'] > split_date]
    
    # Check if we have enough data in the test set
    if len(test_df) < 100:
        # If not enough data, use percentage split
        print("Not enough data in the last 10 days. Using percentage split instead.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
except:
    # Fallback to percentage split if there's an issue with the date-based split
    print("Error in date-based split. Using percentage split instead.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {train_df.shape}")
print(f"Test set size: {test_df.shape}")

# Define features and target
features = ['hour', 'day', 'month', 'day_of_week', 'is_weekend', 
            'Floor', 'Flat', 'Temperature', 'Humidity',
            'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
            'hum_lag_1', 'hum_lag_2', 'hum_lag_3',
            'temp_rolling_mean_3', 'hum_rolling_mean_3',
            'temp_diff', 'hum_diff'] + [col for col in df.columns if col.startswith('Building_')]

X_train = train_df[features]
y_train = train_df['Anomaly']
X_test = test_df[features]
y_test = test_df['Anomaly']

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Feature Importance")
plt.show()

# Plot actual vs predicted
test_df['Predicted_Anomaly'] = y_pred
test_df['Predicted_Probability'] = y_pred_proba

# Try grouping by date if timestamps are available
try:
    # Group by date and plot average actual vs predicted
    daily_results = test_df.groupby(test_df['Timestamp'].dt.date).agg({
        'Anomaly': 'mean',
        'Predicted_Anomaly': 'mean',
        'Predicted_Probability': 'mean'
    }).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_results['Timestamp'], daily_results['Anomaly'], label='Actual Anomaly Rate')
    plt.plot(daily_results['Timestamp'], daily_results['Predicted_Anomaly'], label='Predicted Anomaly Rate')
    plt.plot(daily_results['Timestamp'], daily_results['Predicted_Probability'], label='Predicted Probability')
    plt.title('Daily Anomaly Rate: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Anomaly Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
except:
    print("Unable to group by date for plotting.")
