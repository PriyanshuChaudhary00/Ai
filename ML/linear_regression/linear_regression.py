import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score


#   generatig data
np.random.seed(42)
num_sample = 500
year_of_experience = np.random.randint(2 , 20 , size = num_sample)
slope = (200_000 - 60_000)/18
intercept = 60_000
salery = slope * year_of_experience + intercept 
data = {"year_of_exp":year_of_experience ,
        "salery": salery}
df = pd.DataFrame(data)
df['salery'] = df['salery'] + np.random.normal(0, 15000, size=len(df))

# ploting graph for it 
print(df.describe())
plt.figure(figsize=(10 , 6))
sns.scatterplot(x = "year_of_exp" , y = "salery" , data = df , color = "blue")
sns.regplot(x = "year_of_exp" , y = "salery" , data = df , scatter = False , color = "red")
plt.xlabel("year_of_exp")
plt.ylabel("salery")
plt.show()

x = df[["year_of_exp"]]
y = df["salery"]

#training model

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=10)  
lr = LinearRegression()
lr.fit(x_train , y_train)
lr.score(x_train , y_train)
lr.score(x_test , y_test)

y_pred = lr.predict(x_test)

e1 = mean_absolute_error(y_test , y_pred)
e2 = mean_squared_error(y_test , y_pred)
e3 = r2_score(y_test , y_pred)
print(e1 , e2 , e3)

coef = lr.coef_
inter = lr.intercept_
X = np.linspace(0 , 20 , 100)
Y = coef * X + inter
plt.scatter(x , y , label = f"y = {coef[0]}x + {inter}" , color = "blue")
plt.xlabel("years of ex")
plt.ylabel("salery")
plt.grid()
plt.show()