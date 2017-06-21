import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import io
import numpy as np
from keras.models import model_from_json
import keras.preprocessing.text
from keras.preprocessing import sequence
#from keras.models import Sequential
#model = Sequential()


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
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

## to predict the results for new inputs
text = "this is worst movie ever"
tk = keras.preprocessing.text.Tokenizer(
		num_words = 2000,					#max no.of words in the dataset
		lower = True,
		split = " ")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   
                         tokenizer = None,    
                         preprocessor = None, 
                         stop_words = None,   
                         max_features = 5000) 

train_data_features = vectorizer.fit_transform(text)
train_data_features = train_data_features.toarray()
print (train_data_features)
print ('close')


text = np.array(['this is excellent sentence'])
#print(text.shape)
tk = keras.preprocessing.text.Tokenizer( nb_words=2000, lower=True,split=" ")



print (tk)
print ('\n')
tk.fit_on_texts(text)					#method of keras's tokenizer (other methods are also there)
print (tk.fit_on_texts(text))
print ('\n')
print (tk.texts_to_sequences(text))
print ('\n')
print (np.array(tk.texts_to_sequences(text)))
predictions = loaded_model.predict(np.array(sequence.pad_sequences(tk.texts_to_sequences(text))))
print(predictions)
f = open('predict.txt','w')
f.write(predictions)
f.close()