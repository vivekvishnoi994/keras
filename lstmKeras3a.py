# -*- coding: utf-8 -*-
# LSTM for sequence classification in the IMDB dataset
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ##to ignore the gpu warnings
from keras.datasets import imdbCopy as imdb   ## it is changed to "imdbCopy as imdb" from "imdb" so that imdbcopy will load the data
											  ## this is done so that it will not download data everytime and used locally stored data
# fix random seed for reproducibility
numpy.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.preprocessing.text
from keras.backend import manual_variable_initialization 
#manual_variable_initialization(True)

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

r = random.random()
random.shuffle(X_train, lambda: r)
random.shuffle(y_train, lambda: r)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
top_words = 5000
max_review_length = 500

tokenizer = Tokenizer(num_words = top_words)
tokenizer.fit_on_texts(X_train+X_test)   ##texts is a list of text samples
import pickle
pickle.dump(tokenizer, open('lstmKeras3atokenizer.p', 'wb'))

sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen = max_review_length)

sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences, maxlen = max_review_length)

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

import pandas as pd
df = pd.read_csv('G:\\Log classifier\\keras\\train_data2.csv')
texts = list(df.iloc[:,2])

sequences = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(sequences, maxlen = max_review_length)
predictions = model.predict_classes(texts,verbose=0)
print (predictions)

match = 0
unmatch = 0
for i in range(25000):
	if predictions[i][0] == df.iloc[i,1]:
		match += 1
	else: unmatch += 1
print(match)
print(unmatch)


model.save('lstmModel3.h5',overwrite = True)  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
print("Saved model to disk")

from keras.models import load_model
loaded_model = load_model('lstmModel3.h5')
print("Loaded model from disk")
tokenizer = pickle.load(open('lstmKeras3atokenizer.p', 'rb'))
print ("Loaded tokenizer from disk")

import pandas as pd
df = pd.read_csv('G:\\Log classifier\\keras\\train_data2.csv')
texts = list(df.iloc[:,2])
sequences = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(sequences, maxlen = max_review_length)
predictions = loaded_model.predict_classes(texts,verbose=0)
print (predictions)

match = 0
unmatch = 0
for i in range(25000):
	if predictions[i][0] == df.iloc[i,1]:
		match += 1
	else: unmatch += 1
print(match)
print(unmatch)

