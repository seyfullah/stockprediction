# https://www.youtube.com/watch?v=QIUxPv5PJOY
# https://randerson112358.medium.com/stock-price-prediction-using-python-machine-learning-e82a039ac2bb

# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import time
import datetime

start = time.time()
print("start:" + str(datetime.datetime.now()))

plt.style.use('fivethirtyeight')

stock = 'AAPL'
# Get the stock quote
df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')
# Show the data
print(df)
print(df.shape)

# Visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])  # Converting the dataframe to a numpy array
dataset = data.values  # Get /Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]  # Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
model.save(stock + "-60-LSTM")

# Test data set
test_data = scaled_data[training_data_len - 60:, :]  # Create the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len:,
         :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling

# Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

# Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions  # Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predicted prices
print(valid)

# Get the quote
quote = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')
# Create a new dataframe
new_df = quote.filter(['Close'])
# Get teh last 60 day closing price
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append teh past 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("pred_price")
print(pred_price)

# Get the quote
quote = web.DataReader(stock, data_source='yahoo', start='2019-12-18', end='2019-12-18')
print("quote['Close']")
print(quote['Close'])

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("end:  " + str(datetime.datetime.now()))
print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))