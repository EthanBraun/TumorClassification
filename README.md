# Tumor Classification

### Overview

In this repo I construct a supervised learning model to classify whether a tumor is benign or malignant based on 30 features (e.g. mean tumor radius, mean tumor perimeter, standard deviation of gray-scale texture values, etc.)

The data set I used was obtained through Kaggle, and can be found [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

### Model Specifics

Given that the class labels in the data set are imblanaced with 357 examples of benign tumors and 212 examples of malignant tumors, I used ADASYN (adaptive synthetic sampling) to oversample the minority class and avoid introducing bias into the model.

After balancing the classes in the training set, I train an autoencoder (with latent dim = 20) to reduce the dimensionality of the data and extract richer features. After sufficiently training the autoencoder, I encode the features of the train and test sets with the first half of the network.

Once the features have been encoded, I utilize XGB (extreme gradient-boosted trees) to fit the training data and predict the labels of the test set.

### Results

Over 20 trials with randomly sampled train and test sets, my model achieves 95.965% prediction accuracy.