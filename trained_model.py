import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("car_data.csv", sep="\t")

# Create Car Age feature
current_year = 2024
data["Car_Age"] = current_year - data["Year"]

# Select features
X = data[["Mileage", "Car_Age"]]
y = data["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")