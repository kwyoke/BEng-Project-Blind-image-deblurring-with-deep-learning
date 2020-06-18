import numpy
import cv2
import PIL
import os, os.path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
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

#k.set_image_dim_ordering('th')
num_classes = 73
batch_size = 128
weights_file = "weights_73.hdf5"
test_dir = "training_data/pascal_73/test/"

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
#evaluate model

epochs = 50
learning_rate = 0.0001
decay = learning_rate / epochs
adam = Adam(lr=learning_rate)
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=["accuracy", "top_k_categorical_accuracy"])


# create generator
datagen = ImageDataGenerator(rescale=1./255)
# prepare an iterators for each dataset
test_it = datagen.flow_from_directory('test73/', class_mode='categorical', batch_size=1, target_size=(30, 30), shuffle=False) #shuffle must be false, otherwise mess up prediction order
filenames = test_it.filenames


#evaluate model score
score = model.evaluate_generator(test_it, steps=len(filenames))
print(score)


##########################################################
#obtain predicted probabilities of each kernel for a given patch
predictions = model.predict_generator(test_it, steps=len(filenames)) #note that labels in predictions not same as true labels, so need to rectify  
classes = test_it.classes      
label_map = (test_it.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
classes = [label_map[k] for k in classes]
swap_ind = numpy.zeros(73, dtype=int)

for i in range(73):
	swap_ind[int(label_map[i])] = i

for i in range(len(filenames)):
	predictions[i] = predictions[i][swap_ind] #now predictions contain true kernel labels
	

##########################################
#plot predicted kernel probabilities as heatmap

from matplotlib import pyplot as plt

def generate_kernel_labels():
	sizes = []
	angles = []
	for i in range(13):
		sizes.append(2*i+1)
	for i in range(6):
		angles.append(i*30)

	allkernel_labels = []
	for i in range(len(angles)):
		for j in range(len(sizes)):
			if (i!=0 and j==0):
				pass
			else:
				label = (angles[i], sizes[j])
				allkernel_labels.append(label)
	
	return allkernel_labels

#PLOT HEATMAP
sizes = numpy.arange(1, 27, 2)
angles = numpy.arange(0, 180, 30)
heatmap = numpy.zeros((len(angles), len(sizes))) #size x angle
size_angle_labels = generate_kernel_labels()

num = 200
pred_prob = predictions[num]
print(classes[num])
for k in range(73):
	(angle, size) = size_angle_labels[k]
	r = int(numpy.argwhere(angles == angle))
	c = int(numpy.argwhere(sizes == size))
	
	heatmap[r][c] = pred_prob[k]
	if (size==1):
		heatmap[:][0] = pred_prob[k]

#print(heatmap)


fig, ax = plt.subplots()
im = ax.imshow(heatmap, cmap='hot')

# We want to show all ticks...
ax.set_yticks(numpy.arange(len(angles)))
ax.set_xticks(numpy.arange(len(sizes)))
# ... and label them with the respective list entries
ax.set_yticklabels(angles)
ax.set_xticklabels(sizes)
ax.set_xlabel('Kernel sizes')
ax.set_ylabel('Kernel angles')

ax.set_title("Motion kernel probabilties of a patch estimated by CNN")
fig.tight_layout()
plt.show()