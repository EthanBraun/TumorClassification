import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

trainSplit = 0.9

# Create dataframe from csv
data = pd.read_csv('../data/data.csv')

# Shuffle data
data = data.reindex(np.random.permutation(data.index))

# Split into train and test sets
splitIdx = int(len(data) * trainSplit)
train, test = data[:splitIdx], data[splitIdx:]



#y = data.diagnosis
#x = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
