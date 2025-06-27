
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset

df = pd.read_csv("Housing.csv")
print(df.head())
print(df.info())
print(df.describe())


# Preprocessing

# Convert categorical 'mainroad', 'guestroom', etc. to numeric
#for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']:
#    if col in df.columns:
#        df[col] = pd.get_dummies(df[col], drop_first=True)
df = pd.get_dummies(df, drop_first=True)


# Define features (X) and target (y)

# Let's predict 'price' using other numerical columns
X = df.drop('price', axis=1)
y = df['price']


# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Linear Regression model

lr = LinearRegression()
lr.fit(X_train, y_train)


# Make predictions

y_pred = lr.predict(X_test)


# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")


# Interpret coefficients

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})
print(coefficients)


# Visualize prediction vs actual

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
