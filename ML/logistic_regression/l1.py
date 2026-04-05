import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("./ML/logistic_regression/insurance_data.csv")
# print(df)
lr = LogisticRegression(solver="newton-cg")

x = df[["age"]]
y = df["bought_insurance"]
# print(x, y)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.1 , random_state=10)

lr.fit(x_train , y_train)
lr.predict(x_test)
print(lr.score(x_test , y_test))
test = pd.DataFrame({
    'age':[30]
})
print(lr.predict(test))
