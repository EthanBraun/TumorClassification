import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from keras.models import *
from keras.layers import *

# Hyper-parameters
trainSplit = 0.9
aeOuterDim = 30
aeInnerDim = 10

# Creates and returns densely-connected encoder and autoencoder networks
def createAutoencoder():
	mIn = Input(shape=(aeOuterDim, ))
	enc = Dense(aeInnerDim, activation='linear')(mIn)
	mOut = Dense(aeOuterDim, activation='sigmoid')(enc)

	encoder, autoencoder = Model(mIn, enc), Model(mIn, mOut)
	autoencoder.compile(loss='mse', optimizer='nadam')
	
	return encoder, autoencoder


# -- Preprocessing --

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

# Min-max scale x data to (0, 1)
scaler = MinMaxScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Convert labels to more useful format
convertLabels = lambda y: [[1., 0.] if l == 'M' else [0., 1.] for l in y]
trainY, testY = convertLabels(trainY), convertLabels(testY)


# -- Feature Extraction --

# Create encoder and autoencoder networks
encoder, autoencoder = createAutoencoder()

