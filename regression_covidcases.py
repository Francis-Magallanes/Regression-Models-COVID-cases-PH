import pandas as pd

df = pd.read_csv('covid_trainingdata.csv')

#this will only get the Philippine data and important feature such as
#total case, date
df = df[(df.location == 'Philippines')][["total_cases", "date"]]

#this will replace the null value in the total cases to zero
df["total_cases"].fillna(0, inplace = True)

print(df)
