# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

NUM_POS = 87
NUM_NEG = 130
pos_labels = [1]*NUM_POS
neg_labels = [-1]*NUM_NEG
pos_labels.extend(neg_labels)
pos_features = []
for k in range(1,NUM_POS+1):
    temp = [[0 for x in range(30)] for y in range(20)]
    with open("data/train8/"+ str(k)+".txt") as infile:
        rf = infile.readlines()
    for i in range(min(len(rf),20)):
        for j in range(min(len(rf[i]),30)):
            if rf[i][j] == '#':
                temp[i][j] = 1
    temp = np.array(temp)
    temp = temp.reshape(600)
    pos_features.append(temp)

neg_features = []
for k in range(1,NUM_NEG+1):
    temp = [[0 for x in range(30)] for y in range(20)]
    with open("data/trainOthers/"+ str(k)+".txt") as infile:
        rf = infile.readlines()
    for i in range(min(len(rf),20)):
        for j in range(min(len(rf[i]),30)):
            if rf[i][j] == '#':
                temp[i][j] = 1
    temp = np.array(temp)
    temp = temp.reshape(600)
    neg_features.append(temp)

pos_features = pd.DataFrame(pos_features)
neg_features = pd.DataFrame(neg_features)
features = pd.DataFrame()
features = features.append(pos_features)
features = features.append(neg_features)
features["LABEL"] = pos_labels
features= features.iloc[np.random.permutation(len(features))]
features.to_csv("features.csv", index=False)
