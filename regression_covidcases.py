import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def squarepred(constants, inputs):
    predicts = []

    for i in inputs:
        y = constants[0]*math.pow(i,2) + constants[1]*math.pow(i,1) +constants[2]
        predicts.append(y)
    
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
squaremodel = np.polyfit(x_train.flatten(), y_train.flatten(),2)
squarepredictions = np.array(squarepred(squaremodel,x_test))

#this will sort the values for plotting
lists = sorted(zip(*[x_test, squarepredictions]))

#this will store the sorted values
new_x_test, squarepredictions = list(zip(*lists))

#ploting the important stuff
#plot for the total covid cases
plt.plot(df['nthdayinfection'],df['total_cases'],color='red',label="Actual Total Covid Cases") 

#plot for the linear regression
plt.plot(x_test,linearpredictions,label="Linear Regression Model")

#this will plot for the quadratic regression
plt.plot(new_x_test, squarepredictions, color="green",label="Quadratic Regression Model")

#showing the labels for axis and legends
plt.xlabel("N days since 31/12/2019")
plt.ylabel("Total Number of Covid Cases")
plt.legend(loc="lower right")

plt.show()

