from __future__ import absolute_import, division, print_function

## import tf and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
#X, X_test, y, y_test = train_test_split(X,y, test_size=0.33)

# normalizing the data by scaling
#X_train = X_train / 255.0
#y_train = y_train / 255.0

# build the model
# flatten transforms the 2d array to a 1d array
model = keras.Sequential([
    ## c1
    keras.layers.Conv2D(32, (5,5), input_shape=(120,120,1)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Dropout(0.2),

    ## c2
    keras.layers.Conv2D(128, (3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Dropout(0.2),

    ## c3
    keras.layers.Conv2D(128, (3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Dropout(0.2),

    ## c4
    keras.layers.Conv2D(64, (3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Dropout(0.2),

    # flatten for dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


### compile the model: add loss function -> what we want to minimize
### add optimizer -> how the model is updated based on the data it sees and its loss print_function
### metrics -> used to monitor training and testing steps, this uses the fraction of images that are correctly classifeid

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# model summary
model.summary()


## data augmentation
datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True
)


## early stopping
# monitor val_loss , patience min number of epochs to wait before , mode (auto/min) what monitored
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')

## model checkpoint
#
#mcSave = keras.callbacks.ModelCheckpoint('weights.{epochs:02d}.{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only = True, mode='auto', period= 1)
datagen.fit(X)

## train with model.fit
model.fit(X, y, batch_size = 32, epochs=5, callbacks=[earlyStopping],validation_split= 0.2)
#model.fit_generator(datagen.flow(X, y,batch_size=32),X.shape[0] // 32,epochs=1,validation_data=(X_test, y_test),callbacks=[earlyStopping])
fname = "weights-test-cnn.hdf5"
model.save_weights(fname, overwrite = True)

#load weights => model.load_weights(fname)
