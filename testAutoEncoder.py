# testAutoEncoder.py
# Test the performance of autoEncoder.py
# Python 3.6
# Linux

from keras.models import load_model
import numpy as np

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')
autoencoder = load_model(r'./weights/ae_weights.h5')

inputs = np.array([[1, 2, 2]])
x = encoder.predict(inputs)
y = decoder.predict(x)

print("Input: {}".format(inputs))
print("Encoded: {}".format(x))
print("Decoded: {}".format(y))


autoencoder.compile(loss='mse', optimizer='sgd')
scores = autoencoder.evaluate(inputs, inputs, verbose=0)
print('{}'.format(round(scores*100), 3))
#print('%.2f%%' % (scores[1]*100))