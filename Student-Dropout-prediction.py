import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pygments import highlight
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
df=pd.read_csv("C:\\Users\\user\\Downloads\\student drop out dataset.csv")
print(df)
print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.isnull().sum())

lb=LabelEncoder()
df["Target"]=lb.fit_transform(df["Target"])


y=df["Target"]
x=df[['Course','Daytime/evening attendance']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)
print("Accuracy:",round(acc*100,2),"%")


















