import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import keras

from keras import layers
from keras import models
from keras import optimizers
from keras.models import Model
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
import time

def step_activation(x):
    threshold = 0.4
    cond = tf.less(x, tf.fill(value=threshold, dims=tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

def create_model(featureplan):

	if (featureplan=="mfcc.txt"):
		frame_shape = (800, 40)
	elif (featureplan=="pyannote_based.txt"):
		frame_shape = (800, 35)
	else:
		print ("Incompatible featureplan")
		raise

	## Network Architecture

	input_frame = keras.Input(frame_shape, name='main_input')

	bidirectional_1 = layers.Bidirectional(layers.LSTM(64, activation="tanh", return_sequences=True))(input_frame)
	bidirectional_2 = layers.Bidirectional(layers.LSTM(32, activation='tanh', return_sequences=True))(bidirectional_1)

	tdistributed_1 = layers.TimeDistributed(layers.Dense(40, activation='tanh'))(bidirectional_2)
	tdistributed_2 = layers.TimeDistributed(layers.Dense(10, activation='tanh'))(tdistributed_1)
	tdistributed_3 = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(tdistributed_2)

	

	model = Model(input_frame, tdistributed_3)

	rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0.0001, decay=0.00001)

	model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy"])

	return model
