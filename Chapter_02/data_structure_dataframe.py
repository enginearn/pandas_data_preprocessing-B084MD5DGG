#!/bin/bash/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import string

val = ['a', 1, 0.5]
df = pd.DataFrame(val)
print(df)

val = [[1, 2, 3], [4, 5, 6]]
df = pd.DataFrame(val, index=['a', 'b'], columns=['c', 'd', 'e'])
print(df)

val = [[1, 2, 3], [4, 5]]
df = pd.DataFrame(val)
print(df)

dictionary = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
df = pd.DataFrame(dictionary)
print(df)

age = pd.Series([10, 12, 9], index=list('ABC'))
sex = pd.Series(['M', 'F', 'F'], index=['C', 'A', 'D'])
df = pd.DataFrame({'age': age, 'sex': sex})
print(df)

nest_dict = {
    'age': {'A': 10, 'B': 12, 'C': 9},
    'sex': {'C': 'M', 'A': 'F', 'D': 'F'}
}

df = pd.DataFrame(nest_dict)
print(df)

val = [[1, 2], [4, 5]]
df1 = pd.DataFrame(val, columns=['A', 'B'], dtype=None)
print(df1.dtypes)

df2 = pd.DataFrame(val, columns=['A', 'B'], dtype=np.float64)
print(df2.dtypes)

df = pd.DataFrame({
    'math': [82, 93, 77, 82, 0],
    'english': [64, 48, 77, 61, 90],
    'chemistry': [70, 65, 60, 90, 100],
})

print(df)
