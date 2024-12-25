#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
#import tensorflow as tf 
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import losses
import numpy as np

from fer2013Dataset import FER2013Dataset


def create_model_from_config():
    config = json.load(open("config.json"))
    cnn_config = config['cnn']
    #print(cnn_config)
    model = Sequential()
    input_shape = (48,48,1)
    first_layer = True
    for lcfg in cnn_config['conv_layers']:
        print(lcfg)
        layer_type = lcfg['type']
        if layer_type ==  'conv2d':
            filters = lcfg['filters']
            
        

def create_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def train():
    model = create_model()
    model.summary()

    data = FER2013Dataset()

    X_train = []
    y_train = []
    for i in range(len(data)):
        x = data[i]
        pixels = x[1]
        emotion = x[0]
        X_train.append(pixels)
        y_train.append(emotion)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))

    model.fit(X_train, y_train, epochs=10)
    model.save("model.h5")
    print("done")


def test():
    model = create_model()
    model.load_weights("model.h5")

    data = FER2013Dataset()

    X_test = []
    y_test = []
    for i in range(len(data)):
        x = data[i]
        pixels = x[1]
        emotion = x[0]
        X_test.append(pixels)
        y_test.append(emotion)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(test_acc)
    print("done")

if __name__ == "__main__":
    create_model_from_config()


