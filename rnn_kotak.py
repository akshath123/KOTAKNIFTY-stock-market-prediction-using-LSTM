
# Importing the libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the training set
training_set = pd.read_csv('NSE-KOTAKNIFTY.csv')
training_set = training_set.iloc[:, 1:2].values
training_set_cpy = training_set

# Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs 
X_train = training_set[0:1706]
Y_train = training_set[1:1707]

# Reshaping 
X_train = np.reshape(X_train, (1706, 1, 1))

from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM

# Initializing the RNN
regressor = Sequential()

# Adding the input layer and the LSTM Layer 
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer 
regressor.add(Dense(units = 1)) 

# Compiling the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 

# Fitting the RNN to the training set 
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200) 

# Importing the test data 
test_set = pd.read_csv('kotak_test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

# Making predections
inputs = real_stock_price 
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (202, 1, 1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results 
plt.plot(real_stock_price, color = 'red', label = 'Real Kotak NIFTY Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Kotak NIFTY Stock Price')
plt.title('Kotak NIFTY Price Predection')
plt.xlabel('Time')
plt.ylabel('Kotak NIFTY Stock Price')
plt.legend()
plt.show() 

inp = X_train
inp = sc.transform(inp)
inp = np.reshape(inp, (1706, 1, 1))

predicted_train_set = regressor.predict(inp)
predicted_train_set = sc.inverse_transform(predicted_train_set)

# Visualizing the training 
plt.figure() 
plt.plot(training_set_cpy , color = 'red', label = 'Real Kotak NIFTY Stock Price')
plt.plot(predicted_train_set, color = 'blue', label = 'Predicted Kotak NIFTY Stock Price')
plt.title('Kotak NIFTY Price Predection')
plt.xlabel('Time')
plt.ylabel('Kotak NIFTY Stock Price')
plt.legend()
plt.show() 

# Evaluating the RNN
import math 
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse/800)
