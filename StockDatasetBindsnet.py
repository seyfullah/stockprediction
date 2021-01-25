from __future__ import print_function, division
import math
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from typing import Optional, Dict

from bindsnet.encoding import Encoder, NullEncoder
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class StockDatasetBindsnet(Dataset):
    __doc__ = (
        """BindsNET stock price dataset wrapper:
    
        The stock of __getitem__ is a dictionary containing the price, 
        label (increasing, decreasing, not changing),
        and their encoded versions if encoders were provided.
        \n\n"""
    )

    def __init__(self,
                 csv_file,
                 price_encoder: Optional[Encoder] = None,
                 label_encoder: Optional[Encoder] = None,
                 train=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        # language=rst
        """
        Constructor for the BindsNET dataset wrapper.
        For details on the dataset you're interested in visit

        :param csv_file (string):  Path to the csv file with annotations.
        :param price_encoder: Spike encoder for use on the price
        :param label_encoder: Spike encoder for use on the label
        :param train: train
        """
        self.df = pd.read_csv(csv_file)
        self.train = train
        self.get_technical_indicators()

        # Creating a new dataframe with only the 'Close' column
        data = self.df.filter(['close'])
        # Converting the dataframe to a numpy array
        dataset = data.values

        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        # here we are Scaling the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Creating the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        # Spliting the data into x_train and y_train data sets
        x_train = []
        y_train = []
        dim = 32
        self.window_size = dim * dim
        for i in range(self.window_size, len(train_data)):
            x_train.append(train_data[i - self.window_size:i, 0])
            y_train.append(train_data[i, 0])

        # Here we are Converting x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(x_train), np.array(y_train)

        # Creating the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        # Spliting the data into x_train and y_train data sets
        x_train = []
        y_train = []
        for i in range(self.window_size, len(train_data)):
            x_train.append(train_data[i - self.window_size:i, 0])
            y_train.append(train_data[i, 0])

        # Here we are Converting x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(x_train), np.array(y_train)

        # here we are testing data set
        test_data = scaled_data[training_data_len - self.window_size:, :]
        # Creating the x_test and y_test data sets
        x_test = []
        y_test = dataset[training_data_len:, :]
        # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
        for i in range(dim*dim, len(test_data)):
            x_test.append(test_data[i - self.window_size:i, 0])

        # here we are converting x_test to a numpy array
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

        # Allow the passthrough of None, but change to NullEncoder
        if price_encoder is None:
            price_encoder = NullEncoder()

        if label_encoder is None:
            label_encoder = NullEncoder()

        self.price_encoder = price_encoder
        self.label_encoder = label_encoder

    def __len__(self):
        # return len(self.df) - self.window_size
        if self.train:
            return len(self.x_train) - self.window_size
        return len(self.x_test) - 1

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.df.iloc[idx]

        close = item['close']
        ma7 = item['ma7']
        ma21 = item['ma21']
        # MACD = item['MACD']
        upper_band = item['upper_band']
        lower_band = item['lower_band']
        ema = item['ema']
        # price = {'close': close,
        #          'ma7': ma7,
        #          'ma21': ma21,
        #          'MACD': MACD,
        #          '20sd': m20sd,
        #          'upper_band': upper_band,
        #          'lower_band': lower_band,
        #          'ema': ema,
        #          }
        # price = torch.FloatTensor([close, ma7, ma21, MACD, m20sd, upper_band, lower_band, ema])
        # price = torch.FloatTensor([close, ma7, ma21, ema])
        # price = torch.FloatTensor([close])
        # label = torch.FloatTensor([0])
        # if idx > 0:  # Not changing
        #     if close > self.df.iloc[idx - 1]['close']:
        #         label = torch.FloatTensor([1])  # Increasing
        #     elif close < self.df.iloc[idx - 1]['close']:
        #         label = torch.FloatTensor([2])  # Decreasing
        # output = {
        #     "price": price,
        #     "label": label,
        #     "encoded_price": self.price_encoder(price),
        #     "encoded_label": self.label_encoder(label),
        # }
        if self.train:
            x = torch.FloatTensor(self.x_train[idx])
            y = torch.as_tensor(self.y_train[idx], dtype=torch.float)
            x_encoded = self.price_encoder(x)
            y_encoded = self.label_encoder(y)
            output = {
                "price": x,
                "label": y,
                "encoded_price": x_encoded,
                "encoded_label": y_encoded,
            }
        else:
            x = torch.FloatTensor(self.x_test[idx])
            y = torch.as_tensor(self.y_test[idx], dtype=torch.float)
            x_encoded = self.price_encoder(x)
            y_encoded = self.label_encoder(y)
            output = {
                "price": x,
                "label": y,
                "encoded_price": x_encoded,
                "encoded_label": y_encoded,
            }
        return output

    def get_technical_indicators(self):
        # Create 7 and 21 days Moving Average
        self.df['ma7'] = self.df['close'].rolling(window=7).mean().fillna(0)
        self.df['ma21'] = self.df['close'].rolling(window=21).mean().fillna(0)
        self.df['ma50'] = self.df['close'].rolling(window=50).mean().fillna(0)
        self.df['Daily Return'] = self.df['close'].pct_change()

        s = pd.Series(self.df['close'])

        # Create MACD
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        self.df['MACD'] = macd.fillna(0)  #

        # Create Bollinger Bands
        self.df['20sd'] = s.rolling(20).std().fillna(0)  #
        self.df['upper_band'] = self.df['ma21'] + (self.df['20sd'] * 2)
        self.df['lower_band'] = self.df['ma21'] - (self.df['20sd'] * 2)
        self.df['upper_band'] = self.df['upper_band'].fillna(0)
        self.df['lower_band'] = self.df['lower_band'].fillna(0)

        # Create Exponential moving average
        self.df['ema'] = self.df['close'].ewm(span=20, adjust=False).mean().fillna(0)

        # Create Momentum

        return self.df

    def get_feature_importance_data(self):
        data = self.df.copy()
        y = data['close']
        X = data.iloc[:, [4, 7, 8, 9, 10, 11, 12, 13]]
        train_samples = int(X.shape[0] * 0.65)

        X_train = X.iloc[:train_samples]
        X_test = X.iloc[train_samples:]

        y_train = y.iloc[:train_samples]
        y_test = y.iloc[train_samples:]

        return (X_train, y_train), (X_test, y_test)

    def plot_technical_indicators(self, last_days):
        plt.figure(figsize=(16, 10), dpi=100)
        shape_0 = self.df.shape[0] + (2014 - 1970) * 365
        xmacd_ = shape_0 - last_days
        shape_0 += 200

        self.df = self.df.iloc[-last_days:, :]
        x_ = range(3, self.df.shape[0])
        x_ = list(self.df.index)

        # Plot first subplot
        plt.subplot(2, 1, 1)
        plt.plot(self.df['ma7'], label='MA 7', color='g', linestyle='--')
        plt.plot(self.df['close'], label='Closing Price', color='b')
        plt.plot(self.df['ma21'], label='MA 21', color='r', linestyle='--')
        plt.plot(self.df['upper_band'], label='Upper Band', color='c')
        plt.plot(self.df['lower_band'], label='Lower Band', color='c')
        plt.fill_between(x_, self.df['lower_band'], self.df['upper_band'], alpha=0.35)
        plt.title('Technical indicators for ' + self.df.iloc[0, 6] + ' - last {} days.'.format(last_days))
        plt.ylabel('USD')
        plt.legend()

        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.plot(self.df['MACD'], label='MACD', linestyle='-.')
        plt.title('MACD')
        plt.legend()

        plt.show()
