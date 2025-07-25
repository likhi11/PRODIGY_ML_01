# Linear Regression: House Price Prediction
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (downloaded from Kaggle)
data = pd.read_csv("train.csv")

# Select relevant features
features = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
target = data["SalePrice"]

# Handle missing values (if any)
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))

# Visualization
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
