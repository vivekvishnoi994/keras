import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import io
import numpy as np
from keras.models import model_from_json
import keras.preprocessing.text
from keras.preprocessing import sequence
from keras.datasets import imdbCopy as imdb
#from keras.models import Sequential
#model = Sequential()
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


## we need to load the model first which is saved by lstmKeras.py file
# load json and create model
json_file = open('lstmModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("lstmModel.h5")
print("Loaded model from disk")

## after loading the model we can use it for future predictions
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# change the parameter of next line according to the data you are using and evaluate is for comparing only
#score = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

## to predict the results for new inputs
text = ["this is worst movie ever","this one is a good movie","this is not nice"]

tk = keras.preprocessing.text.Tokenizer(
		num_words = 2000,					#max no.of words in the dataset
		lower = True,
		split = " ")

#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(analyzer = "word",   
#                         tokenizer = None,    
#                         preprocessor = None, 
#                         stop_words = None,   
#                         max_features = 5000) 

#train_data_features = vectorizer.fit_transform(text)
#train_data_features = train_data_features.toarray()
#print (train_data_features)
#print ('close')

#text = "this movie is the worst movie"
#text = np.array(['this is excellent sentence'])
#print(text.shape)
#tk = keras.preprocessing.text.Tokenizer( nb_words=2000, lower=True,split=" ")

#predictions = loaded_model.predict(train_data_features)
#predictions = loaded_model.predict(np.array(sequence.pad_sequences(tk.texts_to_sequences(text))))
#predictions = loaded_model.predict(np.array(tk.fit_on_texts(text)))
#text = keras.preprocessing.text.one_hot(text, 500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
#from prepareData import load_data
#xx,yy,vocSize,ixToChar = load_data('testData.txt',10)


## Method 1
#You need to represent raw text data as numeric vector before training a neural network model. For this, you can use CountVectorizer or TfidfVectorizer provided by scikit-learn. After converting from raw text format to numeric vector representation, you can train a RNN/LSTM/CNN for text classification problem.
#from sklearn.datasets import fetch_20newsgroups
#categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
#twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
#print (twenty_train)
#from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform('testdata.txt')
#print(X_train_counts.shape)


## Method 2

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from keras.datasets import imdbCopy
#data = imdbCopy.get_word_index()
#vocabulary_to_load = data
#count_vect = CountVectorizer(vocabulary=vocabulary_to_load)
#count_vect._validate_vocabulary()
#tfidf_transformer = TfidfVectorizer()
#docs_test = ['this is good movie','this is worst movie']
#x_new_counts = count_vect.transform(docs_test)
#x_new_tfidf = tfidf_transformer.transfrom(x_new_tfidf)



## Method 3
text = ['it was really bad movie','it is the best movie i ever watched','best movie ever made','this movie is not worth watching']
import keras.preprocessing.text
for i in range(len(text)):
	text[i] = keras.preprocessing.text.one_hot(text[i], 5000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")

text = sequence.pad_sequences(text, 500)
predictions = loaded_model.predict_classes(text,verbose=0)
print(predictions)
f = open('predict.txt','rw')
f.write(predictions)
f.close()

