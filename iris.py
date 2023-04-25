import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = pd.read_csv('iris.csv')
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("Iris-setosa:")
print(iris[iris.species=='Iris-setosa'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
print("\nIris-versicolor:")
print(iris[iris.species=='Iris-versicolor'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
print("\nIris-virginica:")
print(iris[iris.species=='Iris-virginica'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())

X_train, X_test, y_train, y_test = train_test_split(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris['species'], test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
