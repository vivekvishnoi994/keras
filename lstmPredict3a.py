import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ##to ignore the gpu warnings

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
from keras.preprocessing import sequence
loaded_model = load_model('lstmModel3.h5')
print("Loaded model from disk")
tokenizer = pickle.load(open('lstmKeras3atokenizer.p', 'rb'))
print ("Loaded tokenizer from disk")

top_words = 5000
max_review_length = 500
import pandas as pd
df = pd.read_csv('G:\\Log classifier\\keras\\train_data2.csv')
#texts = list(df.iloc[:,2])
texts = ['this is a good movie','this movie is relly  good to watch','this does not goes as per the expeations','if you are planning to watch this movie then you are going to waste your time','believe me i never watch such a worst piece in my entire life','I never seen such a flop movie in my entire life','this movie really deserve an oscar','you wall never seen such a betiful piece in your entire life','this movie was pretty bad']
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

