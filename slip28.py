import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('car_dataset.csv')
# Split into X and y
X = df[['horsepower']]
y = df['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression object
lr = LinearRegression()

# Fit the linear regression model to the training data
lr.fit(X_train, y_train)
# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
