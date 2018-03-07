#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:46:50 2018

@author: virginiedo
"""

from keras.callbacks import ModelCheckpoint

from create_dataset import create_feature_df
from sklearn.model_selection import train_test_split

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
import pandas as pd


num_classes = 2

data, labels = create_feature_df('A','N')
labels = labels.values.ravel()

lb = LabelEncoder()
labels = np_utils.to_categorical(lb.fit_transform(labels))



#data = np.expand_dims(data, axis=0) 


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

X_train = np.array(X_train.values)
X_test = np.array(X_test.values)  


X_train = np.reshape(X_train,(X_train.shape[0],55,19,1))
X_test = np.reshape(X_test,(X_test.shape[0],55,19,1))

print('Train sets created')

num_classes = 2
# def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i):

def create_model():
    model = Sequential()
    #model.load_weights('my_model_weights.h5')

    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(55,19,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
    
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='Best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10, verbose=2, shuffle=True, callbacks=[checkpointer])
pd.DataFrame(hist.history).to_csv(path_or_buf='History.csv')


print('CNN score : {}'.format(model.evaluate(X_test,y_test)))



# skf = StratifiedKFold(n_splits=2,shuffle=True)
# target_train = target_train.reshape(size,)

# for i, (train_index, test_index) in enumerate(skf.split(X, target_train)):
	# print("TRAIN:", train_index, "TEST:", test_index)
	# X_train = X[train_index, :]
	# Y_train = Label_set[train_index, :]
	# X_val = X[test_index, :]
	# Y_val = Label_set[test_index, :]
	# model = None
	# model = create_model()
# train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i)