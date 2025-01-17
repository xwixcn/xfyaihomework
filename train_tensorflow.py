#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
#import tensorflow as tf 
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import callbacks
import numpy as np

from fer2013Dataset import FER2013Dataset


def create_cnn_model_from_config(config = None):
    if config == None:
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
            kernel_size = lcfg['kernel_size']
            activation = lcfg['activation']
            if first_layer:
                model.add(layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape))
                first_layer = False
            else:
                model.add(layers.Conv2D(filters, kernel_size, activation=activation))
        elif layer_type == 'max_pooling2d':
            pool_size = lcfg['pool_size']
            model.add(layers.MaxPooling2D(pool_size))
        else:
            print("Unknown layer type")
            sys.exit(1)
    model.add(layers.Flatten())
    for lcfg in cnn_config['dense_layers']:
        units = lcfg['units']
        activation = lcfg['activation']
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(10))
    if 'optimizer' in config:
        optimizers = config['optimizer']
    else:
        optimizers = 'adam'

    model.compile(optimizer=optimizers,
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()
    return model

def create_mlp_model_from_config(config = None):
    if config == None:
        config = json.load(open("config.json"))
    mlp_config = config['mlp']

    model = Sequential()
    input_shape = (48,48,1)
    first_layer = True
    for lcfg in mlp_config['dense_layers']:
        units = lcfg['units']
        activation = lcfg['activation']
        if first_layer:
            model.add(layers.Dense(units, activation=activation, input_shape=input_shape))
            first_layer = False
        else:
            model.add(layers.Dense(units, activation=activation))
    model.add(layers.Flatten())
    model.add(layers.Dense(10))
    if 'optimizer' in config:
        optimizers = config['optimizer']
    else:
        optimizers = 'adam'

    model.compile(optimizer=optimizers,
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()
    return model

def train(logcallback = None, data = None, is_continue = False, config = None):
    config_type = config['type'].lower()
    if config_type == 'cnn':
        model = create_cnn_model_from_config(config)
    elif config_type == 'mlp':
        model = create_mlp_model_from_config(config)
    else:
        print(f"Unknown model type：{config['type']}")
        return

    if is_continue:
        model.load_weights("model.h5")
    model.summary()

    if data == None:
        data: FER2013Dataset = FER2013Dataset()

    data.split_by_config(config)
    data_train = data.get_train_data()
    if config_type == 'cnn':
        X_train, y_train = data.trans_to_cnn(data_train)
    elif config_type == 'mlp':
        X_train, y_train = data.trans_to_mlp(data_train)

    if logcallback == None:
        logcallback = MyLoggerCallback()
    history = model.fit(X_train, y_train, epochs=3, callbacks=[logcallback], verbose=0)
    print(history.history)
    model.save("model.h5")
    print("done")

class MyLoggerCallback(callbacks.Callback):
    def __init__(self):
        super(MyLoggerCallback, self).__init__()
        self.batchs = []
        self.losses = []
        self.accuracies = []
        self.log_per_batch = 10
        self.sender = None

    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            self.batchs.append(batch)
            self.losses.append(logs['loss'])
            self.accuracies.append(logs['accuracy'])
            output = "Batch:{}, Loss:{:.8},Acc:{:.8}".format(batch, logs['loss'], logs['accuracy'])
            print(output)
            if self.sender is not None:
                self.sender.addLogOutput(output)
                self.sender.setTrainPrecision(logs['accuracy'])
                self.sender.updateTrainPrecision()
            
    def setSender(self, sender):
        self.sender = sender
    
def test(callback = None, data = None, config = None):
    if config == None:
        config = json.load(open("config.json"))
        
    config_type = config['type'].lower()
    if config_type == 'cnn':
        model = create_cnn_model_from_config(config=config)
    elif config_type == 'mlp':
        model = create_mlp_model_from_config(config=config)
    else:
        print(f"Unknown model type：{config['type']}")
        return
    
    model.load_weights("model.h5")

    if data == None:
        data = FER2013Dataset()
        data.split_by_config(config)

    if callback == None:
        callback = MyLoggerCallback()

    data_test = data.get_test_data()
    if config_type == 'cnn':
        X_test, y_test = data.trans_to_cnn(data_test)
    elif config_type == 'mlp':
        X_test, y_test = data.trans_to_cnn(data_test)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0, callbacks=[callback])
    callback.sender.setTestPrecision(test_acc)
    callback.sender.updateTestPrecision()
    print(test_acc)
    print("done")

if __name__ == "__main__":
    create_cnn_model_from_config()
    #train()


