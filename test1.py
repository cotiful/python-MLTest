import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = pd.read_csv(
    '/Users/green_workspace/python_lab/bream_length.csv', header=None)
bream_weight = pd.read_csv(
    '/Users/green_workspace/python_lab/bream_weight.csv', header=None)
smelt_length = pd.read_csv(
    '/Users/green_workspace/python_lab/smelt_length.csv', header=None)
smelt_weight = pd.read_csv(
    '/Users/green_workspace/python_lab/smelt_weight.csv', header=None)

my_bream_length = bream_length.to_numpy()
my_bream_weight = bream_weight.to_numpy()
my_smelt_length = smelt_length.to_numpy()
my_smelt_weight = smelt_weight.to_numpy()

# print(my_bream_length)
# print(my_bream_weight)
# print(my_smelt_length)
# print(my_smelt_weight)

array_length = np.concatenate([my_bream_length, my_smelt_length])
array_weight = np.concatenate([my_bream_weight, my_smelt_weight])

fish_data = np.column_stack((array_length, array_weight))
# print(fish_data)

fish_target = [1]*35 + [0]*14

# print(fish_target)
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)

train_arr = np.array(fish_target)

train_input = fish_data[index[:35]]
train_target = train_arr[index[:35]]


# 테스트 데이터
test_input = fish_data[index[35:]]
test_target = train_arr[index[35:]]
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
