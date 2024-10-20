# house_price_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset from CSV file
df = pd.read_csv('house_prices.csv')

# Features and target variable
X = df[['Size (sq ft)', 'Bedrooms', 'Age (years)']]
y = df['Price ($)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the results
print(f"Predicted prices: {y_pred}")
print(f"Mean Squared Error: {mse}")
