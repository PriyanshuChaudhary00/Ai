import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Boston.csv')
# print(df.describe())

df.fillna(df.mean , inplace=True)
df.drop_duplicates(inplace=True)
 
lr = LinearRegression()
# print(df.describe())

x = df[["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]]
y = df["medv"]
# print(y)
x_train , x_test , y_train , y_test = train_test_split(x , y ,  test_size=0.1 , random_state=100)


lr.fit(x_train , y_train)
y_pred = lr.predict(x_test)

e1 = mean_absolute_error(y_test , y_pred)
e2 = mean_squared_error(y_test , y_pred)
e3 = r2_score(y_test , y_pred)
print("mean_absolute_error" , e1 , "mean_squared_error" , e2 , "r2_score" , e3 )

my_house = pd.DataFrame({
    'crim': [0.01],
    'zn': [18.0],
    'indus': [2.3],
    'chas': [0],
    'nox': [0.5],
    'rm': [6.5],     
    'age': [65.2],
    'dis': [4.0],
    'rad': [1],
    'tax': [296],
    'ptratio': [15.3],
    'black': [396],
    'lstat': [4.98]
})
predicted_price = lr.predict(my_house)
print(f"The predicted value of this house is: ${predicted_price[0] * 1000:.2f}")

