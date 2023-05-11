import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Fish.csv')

X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('X_train:', X_train)
print('X_test:', X_test)
print('y_train:', y_train)
print('y_test:', y_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)

print("Predicted values are:\n",y_pred)
print("Actual values are:\n",y_test)
