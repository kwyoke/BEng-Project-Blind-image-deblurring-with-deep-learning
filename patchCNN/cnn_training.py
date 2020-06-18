import numpy
import cv2
#import PIL
import os, os.path
import matplotlib.pyplot as plt
#from PIL import Image
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dropout
from keras import callbacks
from keras.layers import Dense, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD,Adam
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras import backend as k


#parameters to define
num_classes = 73
num_img_perdir = 10000
batch_size = 128
train_dir = 'training_data/pascal_73/train/'
val_dir = 'training_data/pascal_73/validate/'
test_dir = 'training_data/pascal_73/test/'

weights_filename = 'weights_73.hdf5'
epochs = 50
learning_rate = 0.0001
#########################################################
# create generator
datagen = ImageDataGenerator(rescale=1./255)
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory(train_dir, class_mode='categorical', batch_size=batch_size, target_size=(30, 30))
val_it = datagen.flow_from_directory(val_dir, class_mode='categorical', batch_size=batch_size, target_size=(30, 30))
test_it = datagen.flow_from_directory(test_dir, class_mode='categorical', batch_size=batch_size, target_size=(30, 30))

###########################################################
# Defining the model
input_shape=(30,30,3)

model = Sequential()
model.add(Convolution2D(96, (7,7),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

###########################
decay = learning_rate / epochs
adam = Adam(lr=learning_rate)
model.load_weights("weights_73.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=["accuracy"])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
numpy.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


filepath=weights_filename
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Training
hist = model.fit_generator(train_it, steps_per_epoch=5703, validation_data=val_it, validation_steps=1140, epochs=epochs, callbacks=callbacks_list)

# Evaluating the model
score = model.evaluate_generator(test_it, steps=3421)
#score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
######################################################