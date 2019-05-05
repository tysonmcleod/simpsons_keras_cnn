from __future__ import absolute_import, division, print_function

## import tf and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import numpy as np # array operations
import matplotlib.pyplot as plt  #to show image
import os   # to iterate through directories and join paths
import cv2 # image operations
import random # shuffle the training data
import pickle # to just load the dataset??

# split the data
from sklearn.model_selection import train_test_split

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

# normalizing the data by scaling
X = X/255.0
#tf.cast(X, tf.int32)
#X = X.astype(np.int32)
#print(X.dtype)

# split the data
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)

# normalizing the data by scaling
#X_train = X_train / 255.0
#y_train = y_train / 255.0

# build the model
# flatten transforms the 2d array to a 1d array
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(120, 120,1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


### compile the model: add loss function -> what we want to minimize
### add optimizer -> how the model is updated based on the data it sees and its loss print_function
### metrics -> used to monitor training and testing steps, this uses the fraction of images that are correctly classifeid

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

'''
model = Sequential()

model.add(Conv2D(256, (3,3), input_shape = (X.shape[1:])))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

'''
### compile the model: add loss function -> what we want to minimize
### add optimizer -> how the model is updated based on the data it sees and its loss print_function
### metrics -> used to monitor training and testing steps, this uses the fraction of images that are correctly classifeid


## train with model.fit

model.fit(X, y, batch_size = 32, epochs=10, validation_split= 0.3)

## evaluate the accuracy
#test_loss, test_acc = model.evaluate(y_train, y_test)
#print('Test accuracy: ', test_acc)
