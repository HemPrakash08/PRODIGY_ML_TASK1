import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

# Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Define columns to keep
columns_to_keep = ['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath', 'SalePrice']

# Filter train and test datasets to keep only required columns
train = train[columns_to_keep]
test = test[['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath']]  # No SalePrice column in the test dataset

# Log transform the target variable
train['SalePrice'] = np.log1p(train['SalePrice'])
new_skewness = skew(train['SalePrice'])
print("Skewness after logarithmic transformation:", new_skewness)

# Split data into features and target
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values in test data
imputer = SimpleImputer(strategy='mean')
X_test_scaled = imputer.fit_transform(X_test_scaled)

# Train Ridge regression model
ridge = Ridge(alpha=1.0)  # You can tune alpha parameter as needed
ridge.fit(X_train_scaled, y_train)

# Make predictions on the test split
predictions = ridge.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Instead of old scatter plot, plot distribution of predicted sale prices on original scale
predicted_sale_price = np.expm1(predictions)  # Convert predictions back to original scale

plt.figure(figsize=(10,6))
plt.hist(predicted_sale_price, bins=30, color='#4f46e5', alpha=0.7, edgecolor='black')
plt.title('Distribution of Predicted Sale Prices')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Number of Houses')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Also prepare final predictions on test dataset
test_scaled = scaler.transform(test)
test_scaled = imputer.transform(test_scaled)

test_predictions = ridge.predict(test_scaled)
predicted_test_prices = np.expm1(test_predictions)

# Optional: plot LotArea vs predicted prices for test dataset
plt.figure(figsize=(10,6))
plt.scatter(test['LotArea'], predicted_test_prices, alpha=0.6, color='#06b6d4')
plt.title('Lot Area vs. Predicted Sale Price')
plt.xlabel('Lot Area')
plt.ylabel('Predicted Sale Price')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
