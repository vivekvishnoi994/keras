#data = open(DATA_DIR, 'r').read()		#text ="I have a dream." so data will look like this
#chars = list(set(data))				#data = ['I',' ', 'h', 'a', 'v', 'e', ' ', 'a', ' ', 'd', 'r', 'e', 'a', 'm', '.']
#VOCAB_SIZE = len(chars)				#chars = ['I',' ', 'h', 'a', 'v', 'e', 'd', 'r', 'm', '.']

#ix_to_char = {ix:char for ix, char in enumerate(chars)}		#creating dictionaries for the data
#char_to_ix = {char:ix for ix, char in enumerate(chars)}		#thiis is used to convert back numbers into original characters

import numpy as np
data_dir = 'testdata.txt'

def load_data(data_dir, seq_length):				#default value for seq_length is 50
	data = open(data_dir, 'r').read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
	print(len(data), seq_length, int(len(data)/seq_length))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	X = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
	#print(X)
	y = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
	#print(y)
	for i in range(0, int(len(data)/seq_length)):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		#print(X_sequence)
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		#print(X_sequence_ix)
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		#print(input_sequence)
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence
			print(X)

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char


xx,yy,vocSize,ixToChar = load_data(data_dir,10)
