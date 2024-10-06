import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# loading data
company = 'AAPL'
# apple stock

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2024, 10, 1)

data = web.DataReader(company, 'yahoo', start, end)
