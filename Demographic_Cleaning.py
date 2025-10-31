import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import LabelEncoder,StandardScaler

dataset = pd.read_csv("Demographic _102.csv")
print(dataset)

#Seeing the first data row
print(dataset.head(3))

#Seeing the Data Type of the dataset
print(dataset.info())

#Removing Missing Content From Dataset....
print("Null Values in the cell : ","\n",dataset.isnull())

print(dataset.shape)

#To find which dataset have the missing values in percentage...
print(dataset.isnull().sum()/dataset.shape[0]*100)

#Removing the missing dataset from Demographic_102.csv
a = dataset.dropna(inplace=True)
#print(a)

#Taking one by one column to solve the problems...
b = dataset['ID']
print(b)

#Changeing the datatype of the column and removing the unwanted things...
ID_datatype = dataset['ID'] = dataset['ID'].astype('str')
print(ID_datatype)

dataset['ID'] = dataset['ID'].str.replace("\.0","",regex=True)
dataset['ID'] = dataset['ID'].str.replace("E\+\d","",regex=True)
print(dataset.head(3))

Age_dt = dataset['Age'] = dataset['Age'].astype('str')
print(Age_dt)

dataset['Age'] = dataset['Age'].str.replace(".0","")
print(dataset.head(3))

#Converting column's as a number to work in ML...
ID_en = LabelEncoder()
dataset['ID'] = (ID_en.fit_transform(dataset['ID']))
print("ID : ","\n",dataset['ID'])

Age_en = LabelEncoder()
dataset['Age'] = (Age_en.fit_transform(dataset['Age']))
print("Age : ","\n",dataset['Age'])

Gender_en = LabelEncoder()
dataset['Gender'] = (Gender_en.fit_transform(dataset['Gender']))
print("Gender : ", "\n",dataset['Gender'])

Education_en = LabelEncoder()
dataset['Education'] = (Education_en.fit_transform(dataset['Education']))
print("Education : ", "\n",dataset['Education'])

Major_en = LabelEncoder()
dataset['Major'] = (Major_en.fit_transform(dataset['Major']))
print("Major : ", "\n",dataset['Major'])

Income_en = LabelEncoder()
dataset['Income'] = (Income_en.fit_transform(dataset['Income']))
print("Income : ", "\n",dataset['Income'])

Occupation_en = LabelEncoder()
dataset['Occupation'] = (Occupation_en.fit_transform(dataset['Occupation']))
print("Occupation : ", "\n",dataset['Occupation'])