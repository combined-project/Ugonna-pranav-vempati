# autoEncoder.py
# Python autoencoder for the VAHNet dataset
# Python 3.6
# Linux

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import set_random_seed
import os


# Set the ranomd seed for the numpy random generator and the tensorflow
# backend
def seedy(s):
	np.random.seed(s)
	set_random_seed(s)


# Template for model.
class autoEncoder:
	def __init__(self, encodingDim=3, data=None):
		# Set the encoding dimenstion the dimension we're reducing the
		# numbers down to.
		self.encodingDim = encodingDim
		# If no sample data is passed through, it will create some on
		# its own
		if data is not None:
			self.x = data
		else:
			# Create some data to pass in. (Change Later)
			r = lambda: np.random.randint(1,3)
			# Create an array to pass in data. (Randomly generated
			# array in this case. Array of 3 random numbers.)
			self.x = np.array([[r(), r(), r()] for _ in range(1000)])
		# Print the data set we've generated.
		#print(self.x)


	# Encoder part of the model.
	def _encoder(self):
		# Define the input later.
		inputs = Input(shape=(self.x[0].shape))
		# Define hidden layer. (Dense, relu, size is the encoding
		# dimension). Call on inputs that come in from first layer.
		encoded = Dense(self.encodingDim, activation='relu')(inputs)
		# Define the full model, taking both the inputs layer and 
		# encoded layer.
		model = Model(inputs, encoded)
		self.encoder = model
		# Return the model.
		return model


	# Decoder part of the model.
	def _decoder(self):
		# Define the input later. Input here is the output of the
		# encoder layer.
		inputs = Input(shape=(self.encodingDim,))
		# Define output layer. (Dense, relu, size is 3 (same size
		# as the orignal input array)). Call on inputs that come in
		# from the encoded layer.
		decoded = Dense(3)(inputs)
		# Define the full model, taking both the input (encoded)
		# layer and  decoded layer.
		model = Model(inputs, decoded)
		self.decoder = model
		# Return the model.
		return model


	# Create full model.
	def encoder_decoder(self):
		# Initialize encoder, decoder models.
		ec = self._encoder()
		de = self._decoder()

		# Build the final model.
		# Input layer is the vector from the constructor.
		inputs = Input(shape=self.x[0].shape)
		# Run the input through the encoder model.
		ec_out = ec(inputs)
		# Run the output of the encoder model through the decoder
		# model.
		de_out = de(ec_out)
		# Define the full model.
		model = Model(inputs, de_out)

		# Return the model.
		self.model = model
		return model


	# Compile and fit the model.
	def fit(self, batch_size=10, epochs=300):
		# Compile the model. (Use Stocastic Gradient Descent. Loss is
		# Mean Squared Error).
		self.model.compile(optimizer='sgd', loss='mse')
		# Directory to store logs.
		logDir = './log/'
		# Create Tensorboard callback object. Be able to visualize loss
		# and save values to file over epochs.
		tbCallBack = keras.callbacks.TensorBoard(log_dir=logDir,
				histogram_freq=0, write_graph=True, write_images=True)
		# Fit the model. Here, the input should be the same as the
		# output, as that is the nature of autoencoders. Set the epochs
		# and batch size, as well as specify the callback object.
		self.model.fit(self.x, self.x,
						epochs=epochs,
						batch_size=batch_size,
						callbacks=[tbCallBack])


	# Save the model (weights).
	def save(self):
		# Check to see if a previous save exists.
		if not os.path.exists(r'./weights'):
			os.mkdir(r'./weights')
		else:
			self.encoder.save(r'./weights/encoder_weights.h5')
			self.decoder.save(r'./weights/decoder_weights.h5')
			self.model.save(r'./weights/ae_weights.h5')


# Main program.
if __name__ == '__main__':
	seedy(2)
	ae = autoEncoder(encodingDim=2)
	ae.encoder_decoder()
	ae.fit(batch_size=50, epochs=300)
	ae.save()