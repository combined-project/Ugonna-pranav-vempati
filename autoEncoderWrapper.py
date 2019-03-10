# autoEncoderWrapper.py
# author: Diego Magdaleno
# Wrapper function that takes autoEncoder.py and a sample and
# trains and examines the autoencoder.
# Python 3.6
# Linux


import argparse
import pandas as pd
import numpy as np
import autoEncoder as ae
from keras.models import load_model


def main():
	# Pass args from cmd.
	parser = argparse.ArgumentParser()
	parser.add_argument("-ip", "--inputfilepath",
			type=str, help="path to inputfile.")
	parser.add_argument("-b", "--batch_size", default=100,
			type=int, help="batch size of training data.")
	parser.add_argument("-e", "--epochs", default=100,
			type=int, help="number of epochs to train for.")
	parser.add_argument("-d", "--encoding_dim", default=3,
			type=int, help="encdoing dimensions.")
	parser.add_argument("-tr", "--training_mode", default=True,
			type=bool, help="Is the model being trained or Tested?")
	args = parser.parse_args()

	# Load info from args.
	filePath = args.inputfilepath
	batchSize = args.batch_size
	epochs = args.epochs
	encodingDim = args.encoding_dim
	training = args.training_mode

	# Load csv to data frame.
	df = pd.read_csv(filePath)

	# Check args.
	assert batchSize > 0 and batchSize <= df.shape[0], \
			"Batch size must be greater than 0 and less than or equal\
			to the size of the loaded in."
	assert epochs > 0, \
			"Number of epochs must be greater than 0."
	assert encodingDim > 0, \
			"Number of encoding dimensions must be greater than 0."

	# Either enter into training mode or test.
	if training:
		train(df, epochs, encodingDim, batchSize)
	else:
		# Acquire sample/ batch size from data frame.
		df_sample = df.sample(batchSize)
		test(df_sample)


# Train the autoencoder model.
# @param, df: the data frame containing the data to train the model.
# @param, epochs: the number of epochs to train the model for.
# @param, encodingDim: the number of encoding dimensions.
# @param, batchSize: the size of the batch to be fed into the model.
# @return, returns nothing.
def train(df, epochs, encodingDim, batchSize):
	ae.seedy(4)
	print("Generating model...")
	aeModel = ae.autoEncoder(encodingDim=encodingDim, data=df.as_matrix())
	aeModel.encoder_decoder()
	print("Fitting model...")
	aeModel.fit(batch_size=batchSize, epochs=epochs)
	print("Saving...")
	aeModel.save()
	print("Saved.")


# Test the existing autoencoder.
# @param, df: the data frame to test the model against.
# @return, returns nothing.
def test(df):
	# Check that a model's weights already exist. If they don't print
	# an error and exit the program.
	if not os.path.exists(r'./weights'):
		print("Error: No previous model has been saved. Exiting...")
		exit(1)
	else:
		# Otherwise, load the encoding and decoding weights and run the
		# data through the models, and see how the results compare to 
		# the actual values.
		encoder = load_model(r'./weights/encoder_weights.h5')
		decoder = load_model(r'./weights/decoder_weights.h5')

		inputs = df.as_matrix()
		x = encoder.predict(inputs)
		y = decoder.predict(x)

		print("Input: {}".format(inputs))
		print("Encoded: {}".format(x))
		print("Decoded: {}".format(y))


if __name__ == '__main__':
	main()