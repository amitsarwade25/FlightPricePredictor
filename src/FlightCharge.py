import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from the CSV file
df = pd.read_csv('flight_prices_extended.xlsx.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Features and target variable
X = df[['day_of_week', 'num_stops', 'duration', 'airline']]
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Function to predict flight price based on details
def predict_price(day_of_week, num_stops, duration, airline):
    return model.predict([[day_of_week, num_stops, duration, airline]])[0]

# Example prediction
example_price = predict_price(3, 1, 4, 1)
print(f"Predicted Price: ${example_price:.2f}")

# Plot actual prices over time
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['price'], marker='o', linestyle='-', label='Actual Price')
plt.title('Flight Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid(True)

# Predict prices for future dates
future_dates = pd.date_range(start='2023-02-01', periods=14, freq='D')
future_days_of_week = [date.weekday() for date in future_dates]
future_num_stops = [1] * 14  # Assuming 1 stop for future predictions
future_durations = [3] * 14  # Assuming a duration of 3 hours for future predictions
future_airlines = [0] * 14  # Assuming Airline A for future predictions

future_prices = [
    predict_price(day, stops, duration, airline)
    for day, stops, duration, airline in zip(future_days_of_week, future_num_stops, future_durations, future_airlines)
]

plt.plot(future_dates, future_prices, marker='x', linestyle='--', color='red', label='Predicted Price')
plt.legend()
plt.tight_layout()
plt.show()
