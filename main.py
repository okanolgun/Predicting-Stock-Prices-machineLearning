import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# loading data
company = 'TSLA'
# stock name

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2024, 10, 1)

import yfinance as yf

data = yf.download(company, start=start, end=end)

# preparing data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range (prediction_days, len(scaled_data)) :
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# At each step, the 60-day closing prices are added to the x_train list,
# and the closing price of that day is added to the y_train list.
# This data set is made more efficient by converting it into NumPy arrays
# and reshaped in accordance with the LSTM model.

# building the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Dense layer for final prediction

# Our model is a neural network built with LSTM layers used to capture sequential
# relationships in time series data, with Dropout layers added at each step to reduce
# overfitting. The model tries to predict the next closing price
# by looking at previous closing prices

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# The model is compiled with the Adam optimizer and the loss function is the mean squared error.
# This aims to minimize the error between the predicted and actual closing prices.
# The model is then trained on x_train (60 days of historical prices) and y_train (next day's closing price).
# Over 25 epochs, the data is processed in mini-batches of 32 at a time,
# updating the model's weights and learning to predict closing prices in the time series.


''' testing model on the existing data '''

# loading test data
test_start = dt.datetime(2012,1,1)
test_end = dt.datetime.now()


test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# model inputs
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# make predictions on test data
# make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Burada x_test yerine predicted_prices kullanılıyor


# plot the test predictions
plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")

plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()
