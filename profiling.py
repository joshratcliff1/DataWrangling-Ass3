import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import datetime

# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv('Dataset_generation/data_wrangling_rl1_2020_u7199704.csv')


'datasets_test/clean/A-1000.csv','datasets_test/little/A-1000.csv',
'datasets_test/very/A-1000.csv', 'Dataset_generation/data_wrangling_rl1_2020_u7199704.csv'

'''Final Checks - Printing Values'''
print(df)
print(df.isnull().sum(axis = 0))
print(df.state.describe())
# print(df['state'].value_counts())