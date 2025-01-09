#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import csv
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class FER2013Dataset(Dataset):
    def __init__(self, csv_file="data/fer2013.csv", transform=None, usage_filter=None, max_samples=None, sender=None):
        self.max_samples = max_samples
        self.sender = sender
        self.data = self.load_data(csv_file)
        self.transform = transform
        self.usage_filter = usage_filter

    def load_data(self, csv_file):
        data = []
        cnt = 0
        if self.sender is not None:
            self.sender.addLogOutput("Loading data from " + csv_file)
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                cnt += 1
                if self.max_samples is not None and cnt > self.max_samples:
                    break
                row[0] = int(row['emotion'])
                row[1] = [int(p) for p in row['pixels'].split()]
                row[2] = row['Usage']
                data.append(row)
                if cnt % 1000 == 0:
                    msg = "Loaded " + str(cnt) + " samples"
                    if self.sender is not None:
                        self.sender.addLogOutput(msg)
                    print(msg)
        if self.sender is not None:
            self.sender.addLogOutput("Loaded " + str(cnt) + " samples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def split(self, test_size=0.2):
        train_data, test_data = train_test_split(self.data, test_size=test_size, shuffle=True)
        self.train_data = train_data
        self.test_data = test_data

    def split_by_config(self, config):
        test_size = 0.2
        if config is not None:
            if "test_size" in config:
                test_size = config["test_size"]
                test_size = float(test_size)
                if test_size < 0.0 or test_size > 1.0:
                    test_size = 0.2
        self.split(test_size)

    def trans_to_cnn(self, data):
        X = []
        y = []
        for i in range(len(data)):
            x = data[i]
            pixels = x[1]
            emotion = x[0]
            X.append(pixels)
            y.append(emotion)
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], 48, 48, 1))
        return X, y
    
    def trans_to_mlp(self, data):
        X = []
        y = []
        for i in range(len(data)):
            x = data[i]
            pixels = x[1]
            emotion = x[0]
            X.append(pixels)
            y.append(emotion)
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def get_train_data(self):
        if self.train_data is None:
            self.split()
        return self.train_data    
    
    def get_test_data(self):
        if self.test_data is None:
            self.split()
        return self.test_data
    
def pixel_to_image(pixels):
    X = np.zeros((48, 48), dtype=np.uint8)
    for i in range(48):
        for j in range(48):
            X[i, j] = pixels[i * 48 + j]
    return X

if __name__ == "__main__":
    data = FER2013Dataset()
    x = data[0]
    pixels = x[1]
    emotion = x[0]
    X = pixel_to_image(pixels)
    plt.imshow(X, cmap='gray')
    plt.title(emotion)
    plt.show()
    print("done")


    