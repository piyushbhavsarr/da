import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {'Height': [1.5, 1.6, 1.7, 1.8, 1.9],
        'Weight': [50, 60, 70, 80, 90],
        'Purchases': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

X = df[['Height', 'Weight']]
y = df['Purchases']

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