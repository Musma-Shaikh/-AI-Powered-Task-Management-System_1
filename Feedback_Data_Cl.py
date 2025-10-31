import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import LabelEncoder,StandardScaler

#Seeing the dataset...
dataset = pd.read_csv("Feedback_102.csv")
print(dataset)

print(dataset.head(2))

#Datatype ...
dataset.info()

#Removing the null values...
print(dataset.isnull())

#Identifying overall data
print(dataset.shape)

#Removing null values...
print(dataset.isnull().sum()/dataset.shape[0]*100)
a = dataset.dropna(inplace=True)
print(a)

#Changeing the datatype of the column and removing the unwanted things...
ID_datatype = dataset['ID'] = dataset['ID'].astype('str')
print(ID_datatype)

dataset['ID'] = dataset['ID'].str.replace("\.0","",regex=True)
dataset['ID'] = dataset['ID'].str.replace("E\+\d","",regex=True)
print(dataset.head(3))

##Data is cleaned now working on the converting.....
#Converting column "ID" as a number to work in ML...

ID_en = LabelEncoder()
dataset['ID'] = (ID_en.fit_transform(dataset['ID']))
print("ID : ","\n",dataset['ID'])

Feedback_en = LabelEncoder()
dataset['Feedback'] = (Feedback_en.fit_transform(dataset['Feedback']))
print("Feedback : ","\n",dataset['Feedback'])