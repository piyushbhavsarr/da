import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager', 'Region Manager', 'Partner', 'Senior Partner', 'C-level'], 
        'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000]}

df = pd.DataFrame(data)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('X_train:', X_train)
print('X_test:', X_test)
print('y_train:', y_train)
print('y_test:', y_test)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)
print(y_pred)
print("Predicted values are:\n",y_pred)
print("Actual values are:\n",y_test)
