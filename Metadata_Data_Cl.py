import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import LabelEncoder,StandardScaler

dataset = pd.read_csv("Metadata.csv")
print(dataset)

print(dataset.info())

dataset['ID'] = dataset['ID'].astype("int64")
print(dataset.info())

#Removing null data's from the metadata
print(dataset.isnull())

#Identifying overall data
print(dataset.shape)

#Removing null values...
print(dataset.isnull().sum()/dataset.shape[0]*100)
a = dataset.dropna(inplace=True)
print(a)

#Separating data and time column
print(dataset['StartTimeToronto'])
print(dataset['StartTimeToronto'][0])

#Using slicing method...
x = dataset['StartTimeToronto'][0][1] #Finding the index number...
print("Index : ",x)

dataset['StartTimeToronto'] = pd.to_datetime(dataset['StartTimeToronto'])
dataset['EndTimeToronto'] = pd.to_datetime(dataset['EndTimeToronto'])
print(dataset.head(3))

#Separating the data and time column for starting and ending timing and date

dataset['StartDate'] = dataset['StartTimeToronto'].dt.date
dataset['StartTime'] = dataset['StartTimeToronto'].dt.time

dataset['EndDate'] = dataset['EndTimeToronto'].dt.date
dataset['EndTime'] = dataset['EndTimeToronto'].dt.time

dataset = dataset.drop(['StartTimeToronto', 'EndTimeToronto'], axis=1)

print(dataset)

#Bonus cell cleaning....
Bonus_dt = dataset['Bonus'] = dataset['Bonus'].astype('str')
dataset['Bonus'] = dataset['Bonus'].str.replace("0.", "")
dataset['Bonus'] = dataset['Bonus'].str.replace(".0", "")

#ID cell cleaning...
ID_datatype = dataset['ID'] = dataset['ID'].astype('str')
print(ID_datatype)

dataset['ID'] = dataset['ID'].str.replace("\.0","",regex=True)
dataset['ID'] = dataset['ID'].str.replace("E\+\d","",regex=True)
print(dataset.head(3))
print(dataset.info())
##Data is cleaned now working on the converting.....
#Converting column "ID" as a number to work in ML...

ID_en = LabelEncoder()
dataset['ID'] = (ID_en.fit_transform(dataset['ID']))
print("ID : ","\n",dataset['ID'])

CompletionCode_en = LabelEncoder()
dataset['CompletionCode'] = (CompletionCode_en.fit_transform(dataset['CompletionCode']))
print("CompletionCode : ","\n",dataset['CompletionCode'])

Bonus_en = LabelEncoder()
dataset['Bonus'] = (Bonus_en.fit_transform(dataset['Bonus']))
print("Bonus : ","\n",dataset['Bonus'])

StartDate_en = LabelEncoder()
dataset['StartDate'] = (StartDate_en.fit_transform(dataset['StartDate']))
print("StartDate : ","\n",dataset['StartDate'])

StartTime_en = LabelEncoder()
dataset['StartTime'] = (StartTime_en.fit_transform(dataset['StartTime']))
print("StartTime : ","\n",dataset['StartTime'])

EndDate_en = LabelEncoder()
dataset['EndDate'] = (EndDate_en.fit_transform(dataset['EndDate']))
print("EndDate : ","\n",dataset['EndDate'])

EndTime_en = LabelEncoder()
dataset['EndTime'] = (EndTime_en.fit_transform(dataset['EndTime']))
print("EndTime : ","\n",dataset['EndTime'])

