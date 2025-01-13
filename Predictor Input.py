import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.layers import Input
from keras.models import load_model
from datetime import datetime, timedelta

#--------------------------------------------LOAD MODELS -------------------------------------------------------------------#

dailymodel = load_model(r'C:\Users\Pelos\OneDrive\Desktop\Projects\Stock Predictor\stock_predictor_model.keras')
weeklymodel = load_model(r'C:\Users\Pelos\OneDrive\Desktop\Projects\Stock Predictor\weekly_stock_algorithm.keras')

#--------------------------------------------GATHERING TICKER & API DATA ---------------------------------------------------#

today = datetime.today().strftime('%Y-%m-%d')
ticker = input("What ticker would you like to model? ")
data = yf.download(ticker, start="2023-01-01", end=today)
close_prices = data["Close"].values

#-------------------------------------------SCALING DATA -------------------------------------------------------------------#

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled = scaler.fit_transform(close_prices.reshape (-1,1))

#-------------------------------------------- DAILY DATA PREPROCESSING -----------------------------------------------------#

x_daily = []

x_daily = scaled[-10:]
x_daily = np.array(x_daily)
x_daily = x_daily.reshape(1,10,1)

#-------------------------------------------- RUNNING PREDICTIONS ------------------------------------------------------------#

daily_prediction = dailymodel.predict(x_daily)
daily_prediction = scaler.inverse_transform(daily_prediction.reshape(1,-1))


#-------------------------------------------- WEEKLY DATA PREPROCESSING ------------------------------------------------------#

x_weekly = []
x_weekly = scaled[-15:]
x_weekly = np.array(x_weekly)
x_weekly = x_weekly.reshape(1,15,1)

#------------------------------------------- RUNNING WEEKLY PREDICTION --------------------------------------------------------#

weekly_prediction = weeklymodel.predict(x_weekly)
weekly_prediction = scaler.inverse_transform(weekly_prediction.reshape(1,-1))
trend_prices = np.concatenate((close_prices[-15:].reshape(-1), weekly_prediction[-1]))

#------------------------------------------- PLOTTING WEEKLY PREDICTION -------------------------------------------------------#

datelist = []
for i in range (0,22):
    datelist.append((datetime.today() - timedelta(days=15) + timedelta(days=i)).strftime('%m-%d'))
plt.plot(datelist,trend_prices)
plt.title(ticker + " 3 Week Close Predictions")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()