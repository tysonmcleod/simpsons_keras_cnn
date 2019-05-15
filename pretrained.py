from __future__ import absolute_import, division, print_function, unicode_literals

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

# loading data
X = pickle.load(open("X.pre-pickle","rb"))
y = pickle.load(open("y.pre-pickle","rb"))

# normalizing the data by scaling
X = X/255.0

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# normalizing the data by scaling
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.


train_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255)


datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2,
    shear_range = 0.2,
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)

#model.fit_generator(datagen.flow(X_train, y_train,batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=13, validation_data=val_datagen.flow(X_test, y_test, batch_size=32), validation_steps = len(y_train)/32, callbacks=[earlyStopping], shuffle=True)

datagen.fit(X_train)
val_datagen = ImageDataGenerator(rescale=1./255)

# create base model from pretrained
base_model = tf.keras.applications.MobileNetV2(input_shape=(96,96,3),
                                                include_top=False,
                                                weights='imagenet')


# freeze the conv base_model

base_model.trainable = False

#base_model.summary()
print(X.shape)

model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(64, activation=tf.nn.relu),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer ='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print(len(model.trainable_variables))
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
#model.fit_generator(datagen.flow(X_train, y_train,batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=40, validation_data=val_datagen.flow(X_test, y_test, batch_size=32), validation_steps = len(y_train)/32, callbacks=[earlyStopping], shuffle=True)
model.fit(X, y, batch_size = 32, epochs=40, callbacks=[earlyStopping],validation_split= 0.3)


#model.fit(X, y, batch_size = 32, epochs=40, callbacks=[earlyStopping],validation_split= 0.3)



# optimizer adam, sparse since not binary
#model.compile(optimizer ='adam',
#              loss = 'sparse_categorical_crossentropy',
#              metrics = ['accuracy'])

# early stopping
#earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

#model.fit_generator(datagen.flow(X_train, y_train,batch_size=16), steps_per_epoch=len(X_train) / 16, epochs=10, validation_data=val_datagen.flow(X_test, y_test, batch_size=32), callbacks=[earlyStopping], shuffle=True)
#xDD = 2
#model.fit_generator(datagen.flow(X_train, y_train,batch_size=xDD), epochs=50,steps_per_epoch=len(X_train) / xDD, validation_data=val_datagen.flow(X_test, y_test, batch_size=xDD), validation_steps = len(X_test)/xDD, callbacks=[earlyStopping])
#model.fit(X, y, batch_size = 32, epochs=40, callbacks=[earlyStopping],validation_split= 0.3)
# save the weights
#fname = "pre-weights-test-cnn.hdf5"
#model.save_weights(fname, overwrite = True)
