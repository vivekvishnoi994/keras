import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import io
import numpy as np
from keras.models import model_from_json
import keras.preprocessing.text
from keras.preprocessing import sequence
from keras.datasets import imdbCopy as imdb


## we need to load the model first which is saved by lstmKeras.py file
# load json and create model
json_file = open('lstmModel1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("lstmModel1.h5")
print("Loaded model from disk")

## after loading the model we can use it for future predictions
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# change the parameter of next line according to the data you are using and evaluate is for comparing only
#score = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

## to predict the results for new inputs
texts=[]
text = ['it was really bad movie','it is the best movie i ever watched','best movie ever made','this movie is not worth watching']
import pandas as pd
df = pd.read_csv('G:\\Log classifier\\keras\\train_data2.csv')
import keras.preprocessing.text
for i in range(len(df)):
	text = df.iloc[i,2]
	texts.append(keras.preprocessing.text.one_hot(text, 5000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "))
texts = sequence.pad_sequences(texts, 500)
predictions = loaded_model.predict_classes(texts,verbose=0)
match = 0
unmatch = 0
print (len(predictions))
print (predictions[0][0])
print (df.iloc[0,1])
print (predictions[50][0])
print (df.iloc[50,1])
print (type(predictions[0][0]))

for i in range(25000):
	if predictions[i][0] == df.iloc[i,1]:
		match += 1
	else: unmatch += 1

print(match)
print(unmatch)


