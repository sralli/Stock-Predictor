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
%matplotlob inline


asdasdas