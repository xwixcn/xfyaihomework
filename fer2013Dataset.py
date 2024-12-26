#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import csv
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

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


    