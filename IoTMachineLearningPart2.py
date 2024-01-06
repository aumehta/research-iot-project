
import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
import math
from scipy.interpolate import interp1d
from collections import Counter
from sklearn.model_selection import train_test_split


df = pd.read_csv('four_LETbulbs_with_timestamps.csv')
 
 
def lightbulbName(originalName):
     if(originalName == '78:6d:eb:b2:da:d3'):
        return 'd3'
     elif(originalName == '78:6d:eb:48:44:b8'):
        return 'b8'
     elif(originalName == '78:6d:eb:48:45:64'):
        return '64'
     elif(originalName == '78:6d:eb:b2:2b:ac'):
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
 
timeAC = [] 
strengthAC = [] 
outputAC = []
timeB8 = [] 
strengthB8= [] 
outputB8 = [] 
timeD3 = []
strengthD3 = []  
outputD3 = [] 
time64 = []
strength64 = [] 
output64 = [] 
totalStrength = [] 
timeX = []



def checkBulbs(): 
   for index, row in df[0:].iterrows():
      bulb = row[0] #first time in dataframe
      output = row[2]
      bulbName = lightbulbName(bulb)
      timeX.append(row[3])
      if(bulbName == 'ac'):
         timeAC.append(row[3])
         strengthAC.append(row[1])
         outputAC.append(output)
      if(bulbName == 'b8'):
         timeB8.append(row[3])
         strengthB8.append(row[1])
         outputB8.append(output)
      if(bulbName == 'd3'):
         timeD3.append(row[3])
         strengthD3.append(row[1])
         outputD3.append(output)
      if(bulbName == '64'):
         time64.append(row[3])
         strength64.append(row[1])
         output64.append(output)


checkBulbs() 

bulbac = interp1d(timeAC, strengthAC, kind='nearest', bounds_error=False, fill_value= "extrapolate")
bulbb8 =  interp1d(timeB8, strengthB8, kind='nearest', bounds_error=False, fill_value= "extrapolate")
bulbd3 =  interp1d(timeD3, strengthD3, kind='nearest', bounds_error=False, fill_value= "extrapolate")
bulb64 =  interp1d(time64, strength64, kind='nearest', bounds_error=False, fill_value= "extrapolate")

outputACIP = interp1d(timeAC, outputAC, kind='nearest', bounds_error=False)
outputB8IP = interp1d(timeB8, outputB8, kind='nearest', bounds_error=False)
outputD3IP = interp1d(timeD3, outputD3, kind='nearest', bounds_error=False)
output64IP = interp1d(time64, output64, kind='nearest', bounds_error=False)
outputACFinal= outputACIP(timeX).tolist()
outputB8Final= outputB8IP(timeX).tolist()
outputD3Final= outputD3IP(timeX).tolist()
output64Final= output64IP(timeX).tolist()

listReturnOutput = []
def cleanOutputData(): 
   for i in range(37643):
      valueAC = outputACFinal[i]
      valueB8 = outputB8Final[i]
      valueD3 = outputD3Final[i]
      value64 = output64Final[i]
      countNan = 0
      if(math.isnan(valueAC)):
         countNan = countNan +1
      elif(math.isnan(valueB8)):
         countNan = countNan +1
      elif(math.isnan(valueD3)):
         countNan = countNan +1
      elif(math.isnan(value64)):
         countNan = countNan +1
      if(countNan>=2):
         outputACFinal.pop(i)
         outputB8Final.pop(i)
         outputD3Final.pop(i)
         output64Final.pop(i)
         timeX.pop(i)
      else:
         listVals = [valueAC, valueB8, valueD3, value64]
         counter = Counter(listVals)
         most_common = counter.most_common(1)[0][0]
         listReturnOutput.append(most_common)
   return listReturnOutput

output = cleanOutputData()
output[0] = output[1]
fAC =bulbac(timeX).tolist()
fB8 = bulbb8(timeX).tolist()
fD3 = bulbd3(timeX).tolist()
f64 = bulb64(timeX).tolist()

listTotalVectors = [] 
for x in range(37643):
   listAppend = [] 
   listAppend.append(fAC[x])
   listAppend.append(fB8[x])
   listAppend.append(fD3[x])
   listAppend.append(f64[x])
   listTotalVectors.append(listAppend)
print(timeX[0:5])
print(listTotalVectors[0:5])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(listTotalVectors, output, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)
from sklearn import metrics
