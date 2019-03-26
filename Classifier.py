import os
import numpy as np
import pandas as pd 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import fix_yahoo_finance as fix 
import pandas_datareader.data as pdr


import matplotlib.pyplot as plt
#%matplotlob inline


class Stock_Classifier():

    def __init__(self, n_splits = 3):
        #n_splits = no of cross validation splits to perform on classifier
        # Initialises Linear Reg classisfier, Cross Validator and Standard Scaler

        self.n_splits = n_splits
        self.clf =  Pipeline([('scl', StandardScaler()), ('clf', LinearRegression())])

        self.cv = TimeSeriesSplit(n_splits=self.n_splits)

    def get_cv_splits(self, X, y):
         #Return the training, testing split as an array from cross validator
         # X = Numpy array of features values
         # y = Numpy array of target values

        train_splits = []
        test_splits = []

        for train_index, test_index in self.cv.split(X):
            train_splits.append(train_index)
            test_splits.append(test_index)
        
        return train_splits, test_splits


    def train(self, X, y, split_indexes):
        for set_index in split_indexes:
            self.clf.fit(X[set_index], y[set_index])
    
    def score(self, X, y, split_indexes):

        scores = []
        for set_index in split_indexes:
            scores.append(self.clf.score[X[set_index], y[set_index]])
        return scores

    def full_set_train(self, X, y):
        #Training the classifier on full set of data:

        self.clf.fit(X, y)

    def query(self, X):
        return self.clf.predict(X)

    def get_splits(self):
        #Return array of CV splits
        return self.n_splits
        