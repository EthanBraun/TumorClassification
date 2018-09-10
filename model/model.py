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

# Split into x and y
dropCols = ['id', 'diagnosis', 'Unnamed: 32']
trainY, testY = train.diagnosis, test.diagnosis
trainX, testX = train.drop(dropCols, axis=1), test.drop(dropCols, axis=1) 

# Generate synthetic train data to balance classes
trainX, trainY = ADASYN().fit_sample(trainX, trainY)

