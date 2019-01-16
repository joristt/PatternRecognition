import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
"""
to run this kernel, pip install ultimate first from your custom packages
"""
from ultimate.mlp import MLP 
import tensorflow as tf
import gc

from keras.models import Sequential
from keras.layers import Dense, Activation

dataPath = "data/"

train = pd.read_csv(dataPath + "train_V2.csv", nrows=500000)
train["Id"] = [int(idd,16) for idd in train["Id"]]
train["groupId"] = [int(idd,16) for idd in train["groupId"]]
train["matchId"] = [int(idd,16) for idd in train["matchId"]]
train["matchType"] = [abs(hash(idd)) for idd in train["matchType"]]

del train["Id"]
del train["groupId"]
del train["matchId"]
del train["matchType"]


cols = train.columns
x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)
train.columns = cols


train, test = train_test_split(train)
train_lbls = train["winPlacePerc"]
test_lbls  = test["winPlacePerc"]
del train["winPlacePerc"]
del test["winPlacePerc"]

model = Sequential()
model.add(Dense(100, input_shape=(train.shape[1],), activation="relu", kernel_initializer='random_uniform'))
# model.add(Dense(50, activation="relu", kernel_initializer='random_uniform'))
model.add(Dense(1, activation="relu", kernel_initializer='random_uniform'))
# model.add(Activation('softmax'))
model.compile(loss='mean_absolute_error', optimizer='sgd')
model.fit(train, train_lbls, epochs=10)
model.evaluate(test, test_lbls)
model.predict(test)
