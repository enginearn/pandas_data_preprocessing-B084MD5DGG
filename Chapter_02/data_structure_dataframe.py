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

print(f"df['math']:\n{df['math']}")

print(f"df['math', 'english]:\n{df[['math', 'english']]}")

print(f"{df[df['math'] >= 80]}")

val = [[1, 2, 3], [4, 5, 6]]
df = pd.DataFrame(val)
print(f"df.index:\n{df.index}")
print(f"df.columns:\n{df.columns}")
print(f"df.values:\n{df.values}")
print(df)

df.index = ['a', 'b']
print(f"df.index:\n{df.index}\n{df}")

df.columns = ['c', 'd', 'e']
print(f"df.columns:\n{df.columns}\n{df}")

# loc[row, column: column] got a slice of columns' data
print(f"df.loc\n{df.loc['a', :]}")

# loc[row, column] got a scaler data
print(f"{df.loc['b', 'e']}")

# iloc[row, column: column] got a slice of columns' data
print(f"{df.iloc[0, :]}")

# iloc[row, column] got a scaler data
print(f"{df.iloc[1, 2]}")

# shape: (number of rows, number of columns)
print(f"df.shape:\n{df.shape}")

# size: number of elements
print(f"df.size:\n{df.size}")

# ndim: number of dimensions
print(f"df.ndim:\n{df.ndim}")

# T: transpose
print(f"df.T:\n{df.T}")

# describe: summary statistics
print(f"df.describe:\n{df.describe()}")

# info: summary information
print(f"df.info:\n{df.info()}")

# head: first 5 rows
print(f"df.head:\n{df.head()}")

# tail: last 5 rows
print(f"df.tail:\n{df.tail()}")

df = pd.DataFrame({
    'math': [82, 93, 77, 82, 0],
    'english': [64, 48, 77, 61, 90],
    'chemistry': [70, 65, 60, 90, 100],
})

# sort_values: sort by values
print(f"df.sort_values:\n{df.sort_values(by='math')}")
print(f"{df.sort_values(by='math', ascending=False)}")

# sort_index: sort by index
print(f"df.sort_index:\n{df.sort_index()}")

# rename: rename columns
print(f"df.rename:\n{df.rename(columns={'math': 'Math', 'english': 'English'})}")

# rename: rename index
print(f"{df.rename(index={'a': 'A', 'b': 'B'})}")

# drop: drop columns
print(f"df.drop:\n{df.drop(columns=['math', 'english'])}")

# drop: drop index
print(f"{df.drop(index=[0, 1])}")

# dropna: drop NaN
print(f"df.dropna:\n{df.dropna()}")

# fillna: fill NaN
print(f"df.fillna:\n{df.fillna(0)}")

# isnull: check NaN
print(f"df.isnull:\n{df.isnull()}")

# notnull: check not NaN
print(f"df.notnull:\n{df.notnull()}")

# apply: apply function
print(f"df.apply:\n{df.apply(np.sum)}")

# apply: apply function
print(f"{df.apply(np.mean, axis=1)}")

# applymap: apply function to each element
print(f"df.applymap:\n{df.applymap(lambda x: x + 1)}")

# map: apply function to each element
print(f"df.map:\n{df['math'].map(lambda x: x + 1)}")

# replace: replace value
print(f"df.replace:\n{df.replace(0, np.nan)}")

# add: add two dataframes
print(f"df.add:\n{df.add(df)}")

# sub: subtract two dataframes
print(f"df.sub:\n{df.sub(df)}")

# mul: multiply two dataframes
print(f"df.mul:\n{df.mul(df)}")

# div: divide two dataframes
print(f"df.div:\n{df.div(df)}")

# add: add a scaler
print(f"df.add:\n{df.add(1)}")

# sub: subtract a scaler
print(f"df.sub:\n{df.sub(1)}")

# mul: multiply a scaler
print(f"df.mul:\n{df.mul(2)}")

# div: divide a scaler
print(f"df.div:\n{df.div(2)}")

# add: add a series
print(f"df.add:\n{df.add(df['math'], axis=0)}")

# sub: subtract a series
print(f"df.sub:\n{df.sub(df['math'], axis=0)}")

# mul: multiply a series
print(f"df.mul:\n{df.mul(df['math'], axis=0)}")

# div: divide a series
print(f"df.div:\n{df.div(df['math'], axis=0)}")

# add: add a series
print(f"df.add:\n{df.add(df['math'], axis='index')}")

# sub: subtract a series
print(f"df.sub:\n{df.sub(df['math'], axis='index')}")

# mul: multiply a series
print(f"df.mul:\n{df.mul(df['math'], axis='index')}")

# div: divide a series
print(f"df.div:\n{df.div(df['math'], axis='index')}")

# add: add a series
print(f"df.add:\n{df.add(df['math'], axis='columns')}")

# sub: subtract a series
print(f"df.sub:\n{df.sub(df['math'], axis='columns')}")

# mul: multiply a series
print(f"df.mul:\n{df.mul(df['math'], axis='columns')}")

# div: divide a series
print(f"df.div:\n{df.div(df['math'], axis='columns')}")

# add: add a series
print(f"df.add:\n{df.add(df['math'], axis='rows')}")

# sub: subtract a series
print(f"df.sub:\n{df.sub(df['math'], axis='rows')}")

# mul: multiply a series
print(f"df.mul:\n{df.mul(df['math'], axis='rows')}")

# div: divide a series
print(f"df.div:\n{df.div(df['math'], axis='rows')}")

# add: add a series
print(f"df.add:\n{df.add(df['math'], axis='index')}")

# sub: subtract a series
print(f"df.sub:\n{df.sub(df['math'], axis='index')}")

# mul: multiply a series
print(f"df.mul:\n{df.mul(df['math'], axis='index')}")

# div: divide a series
print(f"df.div:\n{df.div(df['math'], axis='index')}")

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
df.iloc[1, 1] = 100

print(f"df:\n{df}")

df['new1'] = 10
df['new2'] = df['new1'] + 1
print(f"{df}")

df.insert(0, 'insert1', [20, 21])
df.insert(1, 'insert2', df['insert1'] + 1)
print(f"{df}")

val = [i for i in range(7, 14)]
idx = [0, 1, 'new1', 'new2', 'insert1', 'insert2', 2]
series_add = pd.Series(val, index=idx, name='add')

# FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
df = df.append(series_add)
print(f"df.append:\n{df}")

df = pd.concat([df, series_add], axis=0)
print(f"pd.concat:\n{df}")

# drop: drop row by index
# axis=0: index label := row
# axis=1: column
df.drop(labels='add', axis=0, inplace=True)
print(f"{df}")

df.drop(labels=['new2', 'insert1', 2], axis=1, inplace=True)
print(f"{df}")

val = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [3, 5, 6], [1, 2, 3], [4, 5, 6]]
df = pd.DataFrame(val, columns=['a', 'b', 'c'])
print(f"df:\n{df}")

# duplicated: return boolean series
print(f"df.duplicated:\n{df.duplicated(keep='first')}")

# drop_duplicates: drop duplicate rows
df = df.drop_duplicates(keep='first')
print(f"df.drop_duplicates first:\n{df}")

df = df.drop_duplicates(keep='last')
print(f"df.drop_duplicates last:\n{df}")

val = [[1, 2, 3], [4, 5, np.nan], [1, np.nan, np.nan], [3, 5, 6], [7, 8, 9]]
df = pd.DataFrame(val, columns=list('ABC'))
print(f"df:\n{df}")
print(f"df.isna:\n{df.isna()}")
print(f"df.isna().sum:\n{df.notna().sum()}")

# dropna: drop rows with NaN
print(f"df.dropna:\n{df.dropna(axis=1)}")

print(f"df.dropna:\n{df.dropna(axis=0)}")

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
print(f"df:\n{df}")

print(f"df.index:\n{df.index}")
print(f"df.columns:\n{df.columns}")

idx = pd.Index([1, 2, 3, 4, 5])
print(f"idx:\n{idx}")

idx1 = pd.Index([1, 2, 3, 4, 5])
idx2 = pd.Index([1, 2, 3, 4, 5])
print(f"logical operation &: {idx1 & idx2}")
print(f"logical operation |: {idx1 | idx2}")

print(f"instercation: {idx1.intersection(idx2)}")
print(f"union: {idx1.union(idx2)}")

# RangeIndex: range of integers
# RangeIndex(start, stop, step)
idx = pd.RangeIndex(1, 501, 1)
print(idx)

df = pd.DataFrame(idx.values, index=idx)
print(f"df:\n{df}")

# DatetimeIndex: range of dates
# DatetimeIndex(freq, start, end)
idx_date = pd.date_range(start='2022-01-01', end='2022-01-31', freq='D',
                            tz='Asia/Tokyo', name='date')
print(f"idx_date:\n{idx_date}")

df_idx_date = pd.DataFrame(idx_date.value_counts(), index=idx_date.normalize())
print(f"df:\n{df_idx_date}")

# PeriodIndex: range of periods
# PeriodIndex(freq, start, end)
idx_period = pd.period_range(start='2020-01-01', end='2022-01-31', freq='D')
print(f"idx_period:\n{idx_period}")

df_idx_period = pd.DataFrame(idx_period.value_counts(), index=idx_period)
print(f"df:\n{df_idx_period}")

# TimedeltaIndex: range of time deltas
# TimedeltaIndex(freq, start, end)
idx_timedelta = pd.timedelta_range(start='1 days', end='31 days', freq='D')
print(f"idx_timedelta:\n{idx_timedelta}")

df_idx_timedelta = pd.DataFrame(idx_timedelta.value_counts(), index=idx_timedelta)
print(f"df:\n{df_idx_timedelta}")

# CategoricalIndex: range of categories
# CategoricalIndex(categories, ordered)
idx_category = pd.CategoricalIndex(['a', 'b', 'c', 'd', 'e'])
print(f"idx_category:\n{idx_category}")

df_idx_category = pd.DataFrame(idx_category.value_counts(), index=idx_category)
print(f"df:\n{df_idx_category}")

# MultiIndex: range of multiple indexes
# MultiIndex.from_product(iterables, names)
idx_multi = pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2, 3]], names=['first', 'second'])
print(f"idx_multi:\n{idx_multi}")

df_idx_multi = pd.DataFrame(idx_multi.value_counts(), index=idx_multi)
print(f"df:\n{df_idx_multi}")

# get data from web page and save as csv file
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = pd.read_csv(url, header=None, skipinitialspace=True, na_values='?')
df.to_csv('adult.csv', index=False)

print(f"df adult.data:\n{df.iloc[0:5]}")


cols = \
"""
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: countries whose came from; Country of origin.
income: >50k <=50k.
"""

col_name = [col.split(':')[0] for col in cols.split('\n') if col != '']
print(f"col_name:\n{col_name}")

df.columns = col_name
print(f"df:\n{df.iloc[0:5]}")
print(f"df:\n{df.iloc[25:28]}")
