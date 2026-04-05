import pandas as pd

	# 1.	Load CSV file
df = pd.read_csv('./data/row.csv')
print("loading data....")
	
df.drop_duplicates(inplace=True)
print("removing duplicates")


df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

df.fillna(df.mean() , inplace=True)

print("Saving cleaned file...")
df.to_csv("clear.csv", index=False)
print("Cleaned file saved as: clear.csv")