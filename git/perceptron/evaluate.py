# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

weights = pd.read_csv("weights.csv", names= ["weights"])
w = weights.weights.tolist()[1:]

def find_sum(x_vec, weights):
    ans = weights[0]
    for i in range(len(x_vec)):
        ans += x_vec[i]*weights[i+1]
    return ans

y_vec = [[0 for x in range(30)] for y in range(20)]
with open("data/test/img.txt") as infile:
    rf = infile.readlines()
for i in range(min(len(rf),20)):
    for j in range(min(len(rf[i]),30)):
        if rf[i][j] == '#':
            y_vec[i][j] = 1
y_vec = np.array(y_vec)
y_vec = y_vec.reshape(600)

output = 1 if find_sum(y_vec, w)>0 else -1

print('It is a 8') if output == 1 else -1