import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {'User ID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Age': [20, 30, 40, 50, 60],
        'Estimated Salary': [1000, 2000, 3000, 4000, 5000],
        'Purchased': [0, 1, 0, 1, 1]}
df = pd.DataFrame(data)

X = df[['Age', 'Estimated Salary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('X_train:', X_train)
print('X_test:', X_test)
print('y_train:', y_train)
print('y_test:', y_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print(y_pred)

print("Predicted values are:\n",y_pred)
print("Actual values are:\n",y_test)
