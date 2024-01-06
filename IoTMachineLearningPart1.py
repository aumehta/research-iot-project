
import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold

from scipy.interpolate import interp1d

df = pd.read_csv('four_LETbulbs_with_timestamps.csv')
 
 
def lightbulbName(originalName):
     if(lightbulb == '78:6d:eb:b2:da:d3'):
        return 'd3'
     elif(lightbulb == '78:6d:eb:48:44:b8'):
        return 'b8'
     elif(lightbulb == '78:6d:eb:48:45:64'):
        return '64'
     elif(lightbulb == '78:6d:eb:b2:2b:ac'):
        return 'ac'
 
 
def sortList(list):
  returnList = []
  indexAC = list.index('ac')
  returnList.append(indexAC)
  indexB8 = list.index('b8')
  returnList.append(indexB8)
  indexd3 = list.index('d3')
  returnList.append(indexd3)
  index64 = list.index('64')
  returnList.append(index64)
  return returnList
 
timeOriginal = df.iloc[:,3][0] #first time in dataframe
listTotalBulb = []
bulbOriginal = df.iloc[:,0][0] #first time in dataframe
listAppendBulb = []
listTotalOutput = []
listTotalStrength = []
listAppendStrength = []
listTotalTime = [] 
listAppendTime = [] 
signalOriginal = df.iloc[:,1][0]
print(signalOriginal)
indexCount= -1
 
for index, row in df[0:].iterrows():
  lightbulb = row[0]
  strength = row[1]
  time = row[3]
  output = row[2]
  if(time - timeOriginal <=6):
     shortBulbName = lightbulbName(lightbulb)
     indexCount = indexCount +1
     if(indexCount>0):
        if shortBulbName in listAppendBulb:
           index = listAppendBulb.index(shortBulbName)
           previousStrength = listAppendStrength[index]
           averageStrength =(previousStrength + strength)/2
           listAppendStrength[index] = np.round(averageStrength,3)
 
        else:
           listAppendBulb.append(shortBulbName)
           listAppendStrength.append(strength)
           listAppendTime.append(time)
 
     else:
        listAppendBulb.append(shortBulbName)
        listAppendStrength.append(strength)
        listAppendTime.append(time)
     
  else:
     counter = 0
     index  = 0
     list = listAppendBulb
     for elem in list:
        if('ac' not in list):
           list.append('ac')
           counter = counter +1
        if('b8' not in list):
           list.append('b8')
           counter = counter +1
        if('d3' not in list):
           list.append('d3')
           counter = counter+1
        if('64' not in list):
           list.append('64')
           counter  = counter +1
     while(index < counter):
        listAppendStrength.append(0)
        index = index+1
     changeIndexStrength= sortList(listAppendBulb)
     newListAppendBulb = []
     newListAppendStrength = []
     for index in changeIndexStrength:
        newListAppendBulb.append(listAppendBulb[index])
        newListAppendStrength.append(listAppendStrength[index])
     listTotalBulb.append(newListAppendBulb)
     listTotalStrength.append(newListAppendStrength)
     listTotalTime.append(listAppendTime)
     listTotalOutput.append(output)
     timeOriginal = time
     listAppendBulb = []
     listAppendStrength = []
     listAppendTime = [] 
     indexCount = 0
     listAppendBulb.append(lightbulbName(lightbulb))
     listAppendStrength.append(strength)
     listAppendTime.append(time)
 
 
# from sklearn import datasets
 
 
# import pandas as pd
 
 
# Import train_test_split function
from sklearn.model_selection import train_test_split
 
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(listTotalStrength, listTotalOutput, test_size=0.3,random_state=109) # 70% training and 30% test
 
from sklearn import svm
 
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
 
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
 
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
 
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision: " , sklearn.metrics.precision_score(y_test, y_pred, average = "weighted"))

 
#take output and input data in a different file
 
 
# from sklearn import preprocessing, svm
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
 # # Dropping any rows with Nan values
# X_train, X_test, y_train, y_test = train_test_split(dfStrength, listTotalOutput, test_size = 0.25)
 # # Splitting the data into training and testing data
# regr = LinearRegression()
 # regr.fit(X_train, y_train)
# print(regr.score(X_test, y_test))
 
from sklearn import tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
X_train, X_test, y_train, y_test = train_test_split(listTotalStrength, listTotalOutput, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier()
 
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
 
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=12)
 
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
 
#Predict the response for test dataset
y_pred = clf.predict(X_test)
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision: " , sklearn.metrics.precision_score(y_test, y_pred, average = "weighted"))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(listTotalStrength, listTotalOutput, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)
 
#looks at models that can change the input width
#r and n can be used to change the arrival length
#load data from external file
# use data interprolation





