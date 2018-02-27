#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:20:40 2018

@author: virginiedo
"""

from create_dataset import create_feature_df
from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder


data, labels = create_feature_df('A','N')

lb = LabelEncoder()
labels = np_utils.to_categorical(lb.fit_transform(labels))
  
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

X_train = np.array(X_train.values)
X_test = np.array(X_test.values)  

print('Train sets created')

num_labels = 2
filter_size = 2


# build model
model = Sequential()

model.add(Dense(256, input_shape=X_train.shape))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

print('Neural network score : {}'.format(model.evaluate(X_test,y_test)))



