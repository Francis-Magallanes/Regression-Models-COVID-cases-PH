import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def exponentialpred(constant,xinputs):
    predicts = []
    for i in xinputs:
        predicts.append( xinputs[0]*math.log10(i) + xinputs[1])
    
    return predicts


#load the data
df = pd.read_csv('covid_trainingdata.csv')

#this will only get the Philippine data and important feature such as
#total case, date
df = df[(df.location == 'Philippines')][["total_cases", "date"]]

#this will replace the null value in the total cases to zero
df["total_cases"].fillna(0, inplace = True)

#this will make numeric values from the dates.
#note that this column is the nth day from start of the covid virus outbreak
#which is on 31/12/2019
df['nthdayinfection'] = df.index - df.index[0] + 1

#this transform the dataframe to arrays
#x-axis values are the nth day
#y-axis values are the total number of cases
x = np.array(df[['nthdayinfection']])
y = np.array(df[['total_cases']])

#split the data to make training data and testing data
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

#making the linear model based on the training data
linearmodel = LinearRegression()
linearmodel.fit(x_train,y_train)

#making predictions based on the testing data
linearpredictions = linearmodel.predict(x_test)

#this will make a exponential model of the covid cases
exponentialmodel = np.polyfit(np.log(x_train.flatten()), y_train.flatten(),1)

print(exponentialmodel)
#ploting the important stuff
plt.plot(df['nthdayinfection'],df['total_cases'],color='red')
plt.plot(x_test,linearpredictions)
plt.show()



