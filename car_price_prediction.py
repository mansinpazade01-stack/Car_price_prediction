# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# 2. Load Dataset (IMPORTANT FIX HERE)
# ==============================
data = pd.read_csv("car_data.csv", sep="\t")

print("First 5 Rows:")
print(data.head())

print("\nColumns:")
print(data.columns)

# ==============================
# 3. Basic Cleaning
# ==============================

# Drop ID column (not useful)
if 'Car ID' in data.columns:
    data.drop(['Car ID'], axis=1, inplace=True)

# Create Car Age
if 'Year' in data.columns:
    data['Car_Age'] = 2025 - data['Year']
    data.drop(['Year'], axis=1, inplace=True)

print("\nData after cleaning:")
print(data.head())

# ==============================
# 4. Convert Categorical Columns
# ==============================
data = pd.get_dummies(data, drop_first=True)

print("\nData after encoding:")
print(data.head())

# ==============================
# 5. Define Features & Target
# ==============================

X = data.drop('Price', axis=1)   # Your dataset uses "Price"
y = data['Price']

# ==============================
# 6. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. Train Model
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==============================
# 8. Evaluation
# ==============================
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# ==============================
# 9. Visualization
# ==============================
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()