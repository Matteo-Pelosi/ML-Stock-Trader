import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def run_prediction(ticker):

    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start="2023-01-01", end=today)
    close_prices = data["Close"].values
    

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
    plt.legend()
    plt.grid(True)

    if (datetime.today().strftime('%m-%d')) in datelist:
        today_index = datelist.index(datetime.today().strftime('%m-%d'))

    plt.axvline(x=today_index,color="r", linestyle="--", label="Today")

    st.write("Tomorrow Prediction",daily_prediction)
    st.pyplot(plt)

#--------------------------------------------LOAD MODELS -------------------------------------------------------------------#
dailymodel = load_model(r'C:\Users\Pelos\OneDrive\Desktop\Projects\Stock Predictor\stock_predictor_model.keras')
weeklymodel = load_model(r'C:\Users\Pelos\OneDrive\Desktop\Projects\Stock Predictor\weekly_stock_algorithm.keras')

#--------------------------------------------GATHERING TICKER & API DATA ---------------------------------------------------#
st.title("Stock Price Predictions")
st.sidebar.header("Ticker Input")
ticker= st.sidebar.text_input("Enter Stock Ticker: ")
if ticker:
    run_prediction(ticker)
