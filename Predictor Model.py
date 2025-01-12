
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.layers import Input
# ---------------------- GATHERING STOCK DATA FROM API -----------------

api_key = '91SGPKQ9J5R8NB7O'
ts = TimeSeries(key = api_key, output_format = 'pandas')
row_data,metadata = ts.get_daily(symbol = 'AAPL', outputsize='full')
close_prices = row_data['4. close'].values
print(close_prices.shape)

# ---------------------- SCALING DATA ----------------------------

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled = scaler.fit_transform(close_prices.reshape (-1,1))

#------------------------ CHOP DATA INTO 10-DAY SEQUENCES -------------------

x,y = [],[] #create two arrays

for i in range (10,len(scaled)):
    x.append(scaled[i-10:i,0])  #add the last ten (from i but not including i) prices from the first(only) column
    y.append(scaled[i,0]) #add the ith entry (next days price)

x,y = np.array(x) , np.array(y) # make then numpy arrays to work with keras

#-------------------------RESHAPING FOR LSTM ---------------

x = x.reshape(x.shape[0],x.shape[1],1) #essentially making it 3D b/c LSTM needs to know number of features (only 1 - closing price)

#--------------------------TRAIN TEST SPLIT -----------------------



x_training , x_testing , y_training, y_testing = train_test_split(x,y,test_size = 0.2, random_state=42)

#------------------------CREATE LSTM MODEL --------------------



model = Sequential()
model.add(Input(shape=(x_training.shape[1], 1)))
model.add(LSTM(50,return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
daily_history = model.fit(x_training,y_training, epochs = 10, batch_size = 32)


#-------------------------LOSS MONITORING -------------------------
plt.plot(daily_history.history['loss'])
plt.title('Daily Model Loss Per Epoch')
plt.ylabel('Loss (Mean Abs. Percentage)') 
plt.xlabel('Epoch') 
plt.show()
#-------------------------EVALUATE-----------------------------

test_loss = model.evaluate(x_testing,y_testing)
print('Test Loss: ', test_loss)

model.save('stock_predictor_model.keras')

predicted_prices = model.predict(x_testing)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(1,-1))
actual_prices = scaler.inverse_transform(y_testing.reshape(1,-1))

print("Predicted Prices: ", predicted_prices[-1])
print("Actual Prices: ",actual_prices[-1])


# // PREPARING WEEKLY DATA///

train_days = 15
predict_days = 7

x_weekly = []
y_weekly = []

for i in range(train_days+predict_days, len(scaled)):
    x_weekly.append(scaled[i-train_days-predict_days:i-predict_days,0])
    y_weekly.append(scaled[i-predict_days:i,0])



x_weekly = np.array(x_weekly)
y_weekly = np.array(y_weekly)
x_weekly = x_weekly.reshape(x_weekly.shape[0] , x_weekly.shape[1] , 1)

X_train_weekly, X_test_weekly, y_train_weekly, y_test_weekly = train_test_split(x_weekly, y_weekly, test_size=0.2, random_state=42)

# //WEEKLY MODEL BUILD AND TRAIN//

weekly_model = Sequential()
weekly_model.add(LSTM(100 , return_sequences=True , input_shape = (X_train_weekly.shape[1],1)))
weekly_model.add(Dropout(0.2))
weekly_model.add(LSTM(100))
weekly_model.add(Dropout(0.2))
weekly_model.add(Dense ( 128, activation = 'relu' ))
weekly_model.add(Dense ( 64, activation = 'relu'))
weekly_model.add(Dense ( 32, activation = 'relu'))
weekly_model.add(Dense (7))
weekly_model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
history = weekly_model.fit(X_train_weekly , y_train_weekly , epochs = 10 , batch_size = 32)

weekly_model.save('weekly_stock_algorithm.keras')

# plotting loss per epoch (to monitor overfitting) #

plt.plot(history.history['loss'])
plt.title('Weekly Model Loss Per Epoch')
plt.ylabel('Loss (Mean Abs. Percentage)') 
plt.xlabel('Epoch') 
plt.show()

# Printing the final model loss and comparison values #

print("Weekly Model Loss: ", weekly_model.evaluate(X_test_weekly,y_test_weekly))

weekly_predicted_prices = weekly_model.predict(X_test_weekly)
weekly_predicted_prices = scaler.inverse_transform(weekly_predicted_prices.reshape(7,-1))
weekly_actual_prices = scaler.inverse_transform(y_test_weekly.reshape(7,-1))

print("Predicted Prices: ", weekly_predicted_prices[-1])
print("Actual Prices: ", weekly_actual_prices[-1])


