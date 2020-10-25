import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('covid_trainingdata.csv')

#this will only get the Philippine data and important feature such as
#total case, date
df = df[(df.location == 'Philippines')][["total_cases", "date"]]

#this will replace the null value in the total cases to zero
df["total_cases"].fillna(0, inplace = True)

df['nthdayinfection'] = df.index - df.index[0]


x = np.array(df[['nthdayinfection']])
y = np.array(df[['total_cases']])


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

linearmodel = LinearRegression()
linearmodel.fit(x_train,y_train)

predictions = linearmodel.predict(x_test)

plt.plot(df['nthdayinfection'],df['total_cases'],color='red')
plt.plot(x_test,predictions)
plt.show()



