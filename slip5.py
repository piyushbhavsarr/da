import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

df=pd.read_csv("Iris.csv")
print(df)

setosa=df[df['Species']=='Iris-setosa']
print("Statistic for Iris-setosa",setosa.describe())

versicolor=df[df['Species']=='Iris-versicolor']
print("Statistic for Iris-versicolor",versicolor.describe())

virginica=df[df['Species']=='Iris-virginica']
print('Statistic for Iris-virginica',virginica.describe())

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuray",accuracy)
