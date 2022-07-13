#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import string

random.seed(0)

height_list = [random.randrange(140, 200) for _ in range(1, 11)]
print(height_list)

height_series = pd.Series(height_list)
print(height_series)

weight_array = np.random.randint(45, 100, size=10)
weight_series = pd.Series(weight_array)
print(weight_series)

ser = pd.Series([1, 2, 3], name="some series")
print(ser)

val = [i for i in range(1, 11)]
labels = [string.ascii_letters[i] for i in range(10)] # range(len(val))
ser = pd.Series(val, index=labels)
print(ser)
print(ser.index)

list_1 = [i for i in range(1, 11)]
list_2 = [string.ascii_uppercase[i] for i in range(len(list_1))]
dict_1 = dict(zip(list_2, list_1))
ser = pd.Series(dict_1)
print(ser)

dict_2 = {string.ascii_letters[i]: i for i in range(len(string.ascii_letters))}
print(dict_2)

dict_1 = {string.ascii_letters[i]: i for i in range(3)}
print(dict_1)
a = pd.Series(dict_1, index=["a", "b", "c", "d"])
print(a)

# make Series from scalar
print(pd.Series(5, index=["a", "b", "c", "d"]))

ser = pd.Series(val, index=list("あいうえおかきくけこ"))
print(ser)
print(ser["あ"])
print(ser["あ" : "お"])
print(ser.iloc[0]) # index location same as ser["あ"]
print(ser[ser > 6])
print(ser + 3)
print(ser)
print(ser * 2)
print(ser / 2)

mylist = [i  for i in range(1, 6)]
print(mylist)
print([i + 2 for i in range(1, 6)])
for i in range(len(mylist)):
    mylist[i] += 2
print(mylist)

ser2 = pd.Series([6, 7, 8, 9, 10], index=list("あかさたな"))
ser3 = ser + ser2
print(ser3)

if __name__ == '__main__':
    sys.exit(0)

