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
