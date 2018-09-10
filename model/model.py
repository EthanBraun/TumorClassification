import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import *

# Hyper-parameters
trainSplit = 0.9
aeOuterDim = 30
aeInnerDim = 20
aeEpochs = 500

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
#convertLabels = lambda y: [[1., 0.] if l == 'M' else [0., 1.] for l in y]
convertLabels = lambda y: [1. if l == 'M' else 0. for l in y]
trainY, testY = convertLabels(trainY), convertLabels(testY)


# -- Feature Extraction --

# Create encoder and autoencoder networks
encoder, autoencoder = createAutoencoder()

# Fit trainX to itself with autoencoder
print('\nFitting autoencoder\n')
autoencoder.fit(trainX, trainX, epochs=aeEpochs, validation_split=0.1, verbose=0)

# Encode features to reduce dimensionality
trainX = encoder.predict(trainX)
testX = encoder.predict(testX)


# -- Visualization --

# Create dataframe for visualization
cols = ['enc_' + str(i) for i in range(aeInnerDim)]
#revertLabels = lambda y: ['M' if l == [1., 0.] else 'B' for l in y]
revertLabels = lambda y: ['M' if l == 1. else 'B' for l in y]
xData = pd.DataFrame(trainX, columns=cols)
yData = pd.DataFrame(revertLabels(trainY), columns=['diagnosis'])
visData = pd.concat([xData, yData], axis=1)
visData = pd.melt(visData, id_vars='diagnosis', var_name='features', value_name='value')

# Visualize encoded features with plot 
sns.set(style='whitegrid')
ax = sns.violinplot(data=visData, x='features', y='value', hue='diagnosis', split=True)
plt.show(ax)


# -- Model Training --

# Train classifier on encoded features of training set
xgbParams = {'maxDepth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
xgbRounds = 20
dTrain = xgb.DMatrix(trainX, label=trainY)
xgbTree = xgb.train(xgbParams, dTrain, xgbRounds) 


# -- Model Prediction --

# Determine accuracy of predictions with test set
dTest = xgb.DMatrix(testX)
preds = xgbTree.predict(dTest)

intPreds = [int(p + 0.5) for p in preds]
matchingPreds = [1. if p == y else 0. for p, y in zip(intPreds, testY)]
print('\nPrediction accuracy: ' + str(sum(matchingPreds) / len(matchingPreds)))
