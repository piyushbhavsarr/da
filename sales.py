import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('sales.csv')

X = df[['TV', 'Radio', 'Newspaper']] # independent variables
y = df['Sales'] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'Training set size: {len(X_train)}')
print(f'Testing set size: {len(X_test)}')

regressor = LinearRegression()
regressor.fit(X_train, y_train)


print(f'Coefficients: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
