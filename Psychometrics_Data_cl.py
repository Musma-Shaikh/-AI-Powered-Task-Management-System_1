import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import LabelEncoder,StandardScaler

dataset = pd.read_csv("Psychometrics.csv")
print(dataset)

print(dataset.info())

#Removing null data's from the metadata
print(dataset.isnull())

#Identifying overall data
print(dataset.shape)

#Removing null values...
print(dataset.isnull().sum()/dataset.shape[0]*100)
a = dataset.dropna(inplace=True)
print(a)

#ID cell cleaning...
ID_datatype = dataset['ID'] = dataset['ID'].astype('str')
print(ID_datatype)

dataset['ID'] = dataset['ID'].str.replace("\.0","",regex=True)
dataset['ID'] = dataset['ID'].str.replace("E\+\d","",regex=True)
print(dataset.head(3))

print(dataset.info())

#Converting column's as a number to work in ML...
ID_en = LabelEncoder()
dataset['ID'] = (ID_en.fit_transform(dataset['ID']))
print("ID : ","\n",dataset['ID'])

dssQuestion1_en = LabelEncoder()
dataset['dssQuestion1'] = (dssQuestion1_en.fit_transform(dataset['dssQuestion1']))
print("dssQuestion1 : ","\n",dataset['dssQuestion1'])

dssQuestion2_en = LabelEncoder()
dataset['dssQuestion2'] = (dssQuestion2_en.fit_transform(dataset['dssQuestion2']))
print("dssQuestion2 : ","\n",dataset['dssQuestion2'])

dssQuestion3_en = LabelEncoder()
dataset['dssQuestion3'] = (dssQuestion3_en.fit_transform(dataset['dssQuestion3']))
print("dssQuestion3 : ","\n",dataset['dssQuestion3'])

dssQuestion4_en = LabelEncoder()
dataset['dssQuestion4'] = (dssQuestion4_en.fit_transform(dataset['dssQuestion4']))
print("dssQuestion4 : ","\n",dataset['dssQuestion4'])

dssQuestion5_en = LabelEncoder()
dataset['dssQuestion5'] = (dssQuestion5_en.fit_transform(dataset['dssQuestion5']))
print("dssQuestion5 : ","\n",dataset['dssQuestion5'])

dssQuestion6_en = LabelEncoder()
dataset['dssQuestion6'] = (dssQuestion6_en.fit_transform(dataset['dssQuestion6']))
print("dssQuestion6 : ","\n",dataset['dssQuestion6'])

dssQuestion7_en = LabelEncoder()
dataset['dssQuestion7'] = (dssQuestion7_en.fit_transform(dataset['dssQuestion7']))
print("dssQuestion7 : ","\n",dataset['dssQuestion7'])

dssQuestion8_en = LabelEncoder()
dataset['dssQuestion8'] = (dssQuestion8_en.fit_transform(dataset['dssQuestion8']))
print("dssQuestion8 : ","\n",dataset['dssQuestion8'])

dssQuestion9_en = LabelEncoder()
dataset['dssQuestion9'] = (dssQuestion9_en.fit_transform(dataset['dssQuestion9']))
print("dssQuestion9 : ","\n",dataset['dssQuestion9'])

dssQuestion10_en = LabelEncoder()
dataset['dssQuestion10'] = (dssQuestion10_en.fit_transform(dataset['dssQuestion10']))
print("dssQuestion10 : ","\n",dataset['dssQuestion10'])

dssQuestion11_en = LabelEncoder()
dataset['dssQuestion11'] = (dssQuestion11_en.fit_transform(dataset['dssQuestion11']))
print("dssQuestion11 : ","\n",dataset['dssQuestion11'])

msShortQuestion1_en = LabelEncoder()
dataset['msShortQuestion1'] = (msShortQuestion1_en.fit_transform(dataset['msShortQuestion1']))
print("msShortQuestion1 : ","\n",dataset['msShortQuestion1'])

msShortQuestion2_en = LabelEncoder()
dataset['msShortQuestion2'] = (msShortQuestion2_en.fit_transform(dataset['msShortQuestion2']))
print("msShortQuestion2 : ","\n",dataset['msShortQuestion2'])

msShortQuestion3_en = LabelEncoder()
dataset['msShortQuestion3'] = (msShortQuestion3_en.fit_transform(dataset['msShortQuestion3']))
print("msShortQuestion3 : ","\n",dataset['msShortQuestion3'])

msShortQuestion4_en = LabelEncoder()
dataset['msShortQuestion4'] = (msShortQuestion4_en.fit_transform(dataset['msShortQuestion4']))
print("msShortQuestion4 : ","\n",dataset['msShortQuestion4'])

msShortQuestion5_en = LabelEncoder()
dataset['msShortQuestion5'] = (msShortQuestion5_en.fit_transform(dataset['msShortQuestion5']))
print("msShortQuestion5 : ","\n",dataset['msShortQuestion5'])

msShortQuestion6_en = LabelEncoder()
dataset['msShortQuestion6'] = (msShortQuestion6_en.fit_transform(dataset['msShortQuestion6']))
print("msShortQuestion6 : ","\n",dataset['msShortQuestion6'])