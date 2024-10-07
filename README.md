# Predicting-Stock-Prices-machineLearning-neuralNetwork

In this project, we created a LSTM (Long Short-Term Memory) model to predict Tesla (TSLA) and Apple (APPL) stock prices. First, we downloaded the closing prices between 2020-2024 using the yfinance library and normalized them with Min-Max Scale. Then, we prepared the training data by establishing the relationship between the 60-day closing prices (x_train) and the next day closing price of this period (y_train). 

The model is defined as a neural network with three LSTM layers and Dropout layers between each layer to prevent over-learning and is trained with the Adam optimization algorithm. The trained model makes predictions on the test data for the period 2020-2024, and the actual and predicted prices are visualized with matplotlib. 

Finally, the model predicts the future closing price using the past 60-day data, and this value is presented to the user by subtracting it from the scale. 

