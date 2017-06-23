import os
import pandas as pd
from pandas import DataFrame

dataPath = "G:\\Log classifier\\keras\\data\\aclImdb_v1\\aclImdb"
testPath = dataPath + '\\test\\'
trainPath = dataPath + '\\train\\'
X_train = []
y_train = []
X_test = []
y_test = []

for filename in os.listdir(trainPath+'neg\\'):
	file = open(trainPath+'neg\\'+filename,'r',encoding="utf8")
	X_train.extend(file.readlines())
	y_train.append(0)
	file.close()

for filename in os.listdir(trainPath+'pos\\'):
	file = open(trainPath+'pos\\'+filename,'r',encoding="utf8")
	X_train.extend(file.readlines())
	y_train.append(1)
	file.close()

for filename in os.listdir(testPath+'neg\\'):
	file = open(testPath+'neg\\'+filename,'r',encoding="utf8")
	X_test.extend(file.readlines())
	y_test.append(0)
	file.close()

for filename in os.listdir(testPath+'pos\\'):
	file = open(testPath+'pos\\'+filename,'r',encoding="utf8")
	X_test.extend(file.readlines())
	y_test.append(1)
	file.close()

d = {
	'text': X_train,
	'label': y_train
}
df = pd.DataFrame(d)
df.to_csv('train_data2.csv')