# -*- coding: utf-8 -*-
# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdbCopy as imdb   ## it is changed to "imdbCopy as imdb" from "imdb" so that imdbcopy will load the data
											  ## this is done so that it will not download data everytime and used locally stored data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.preprocessing.text
# fix random seed for reproducibility
numpy.random.seed(7)

import os
import pandas as pd
from pandas import DataFrame
import random

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

print(type(X_train),type(y_train))
train = list(zip(X_train, y_train))
random.shuffle(train)
X_train = []
y_train = []
for item in train:
	X_train.append(item[0])
	y_train.append(item[1])
print(type(X_train),type(y_train))
X_train = [i.decode('UTF-8') for i in X_train]
top_words = 5000
for i in range(len(X_train)):
	if i==0: 
		print (X_train)
		print(type(X_train))
	X_train[i] = keras.preprocessing.text.one_hot(X_train[i], top_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")

for i in range(len(X_test)):
	X_test[i] = keras.preprocessing.text.one_hot(X_test[i], top_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")

X_train = sequence.pad_sequences(X_train, 500)
X_test = sequence.pad_sequences(X_test, 500)

# load the dataset but only keep the top n words, zero the rest

print ('start')
print (type(X_train))
print (X_train.shape)
print (X_train[0])
print ('viv')
print (X_test[0:2])
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

## to save the model so that we can used it for future purposes
# serialize model to JSON
model_json = model.to_json()
with open("lstmModel2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("lstmModel2.h5")
print("Saved model to disk")