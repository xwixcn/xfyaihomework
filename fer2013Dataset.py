#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import csv
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class FER2013Dataset(Dataset):
    def __init__(self, csv_file="data/fer2013.csv", transform=None, usage_filter=None):
        self.data = self.load_data(csv_file)
        self.transform = transform

    def load_data(self, csv_file):
        data = []
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                row[0] = int(row['emotion'])
                row[1] = [int(p) for p in row['pixels'].split()]
                row[2] = row['Usage']
                data.append(row)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
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

    