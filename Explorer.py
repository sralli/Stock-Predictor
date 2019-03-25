import pandas as pd 
import numpy as np 
import datetime 
from dateutil.relativedelta import relativedelta
from Classifier import Stock_Classifier

class Stock_Explorer():

    #Interact with single stock data to evaluate it and return predicted values

    def __inti__(self, series, symbol, rolling_window_size = 10, prediction_window_size = 14, return_date_range = 100):
        # series = Pandas Series of stocks data. Dates are index
        # Symbol = string of symbol to be evaluated
        # rolling_window_size = window size of rolling stats
        # Amount of days to be predicted for the next few days
        # # Range, rnage of dates to be evaluated. 
    
        self.symbol = symbol
        self.stock_clf = Stock_Classifier()
        self.series = series
        self.rolling_window_size = rolling_window_size
        self.prediction_window_size = prediction_window_size
        self.return_date_range = return_date_range

    
    def get_json(self):
        # Returning dict of stock exploration
        """
        Data returned: Symbol, BB-Ratios, Momentum, SMA_Ratios, SMA, Lower BB
        Upper BB, Normed_Data, Target_Vals, Stats, Prediction_Size, Train_Test_data, 
        Dates
        """

        final_json = {}

        if len(self.series) <= self.stock_clf.get_splits():
            final_json['error'] = "Error with data"
            return final_json
        
        #Split and train: 

        self.__get_X_y()
        train_test_data = self.__train_and_test()

        #Predict: 

        self.__predict_vals()

        