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

val = [i for i in range(1, 11)]

a = pd.Series(val)
b = pd.Series(val, index=[0,1,2,3,4,5,6,7,8,9])
c = pd.Series(val, index=list("abcdefghij"))

print(f"a: {a.index}")
print(f"b: {b.index}")
print(f"c: {c.index}")

a = pd.Series(['a', 'b', 'c'])
b = pd.Series([1, 2, 3])
c = pd.Series([1.0, 2.0, 3.0])
d = pd.Series([True, False, True])
e = pd.Series(['a', 1, 1.0, True])

print(f"a: {a.dtype}")
print(f"b: {b.dtype}")
print(f"c: {c.dtype}")
print(f"d: {d.dtype}")
print(f"e: {e.dtype}")

a = pd.Series([1, 2, 3], index=[1, 2, 3])
b = pd.Series([1, 2, 3], index=list("abc"))

print(f"a:\n{a}")
print(f"b:\n{b}")

print(f"a[1]: {a.loc[1]}")
print(f"b[1]: {b.loc['a']}")

print(f"a[1]: {a.iloc[0:1]}")
print(f"b[1]: {b.loc['a':'c']}")

ser = pd.Series([1, 1, 3, 4, 'a'])
print(ser.size)

a = pd.Series([1, 2, 3])
b = pd.Series([1, 1, 3])

print(f"a: {a.is_unique}")
print(f"b: {b.is_unique}")

a = pd.Series([1, 2, 3])
b = pd.Series(['a', 'b', 'c'])
c = pd.Series(['a', 'a', 'b'], dtype='category')

print(f'a: {type(a.values), a.values}')
print(f'b: {type(b.values), b.values}')
print(f'c: {type(c.values), c.values}')

ser = pd.Series([1, 2, 3, 4, 5], index=list("abcde"))
print(ser)
ser['a'] = 10
ser['b':'d'] = [20, 30, 40]
ser['e'] = 50
print(ser)
ser['あ'] = 4
print(ser)

# ser2 = pd.Series([5, 6], index=['い', 'う'])
ser2 = pd.Series([5, 6], index=list('いう'))
ser_append = ser.append(ser2) # FutureWarning: The series.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
# ser_concat = ser.concat(ser2)
print(f'ser_append: \n{ser_append}')
# print(f'ser_concat: {ser_concat}')

ser = ser.append(ser2, ignore_index=True)
print(ser)

ser = pd.Series([1, 2, 3, 4, 5], index=list("abcde"))
del ser['a']
print(ser)

ser.drop(index=['b', 'c'], inplace=True)
print(ser)

ser = pd.Series([1, 1, 2, 2, 2, 3], index=list("abcdef"))
ser = ser.drop_duplicates(keep='first')
print(f'drop duplicate first: \n{ser}')

ser = ser.drop_duplicates(keep=False)
print(f'drop duplicate False: \n{ser}')

ser = pd.Series([1, np.nan, 3, 4, np.nan], index=list("abcde"))
print(ser)

ser = ser.isna()
print(ser)
print(ser[ser.isna()])

ser.dropna()
print(ser)

if __name__ == '__main__':
    sys.exit(0)

