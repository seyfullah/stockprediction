from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class StockDataset(Dataset):
    """Stock self.df."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.df = pd.read_csv(csv_file)
        self.get_technical_indicators()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        date = self.df.iloc[idx, 0]
        open = self.df.iloc[idx, 1]
        high = self.df.iloc[idx, 2]
        low = self.df.iloc[idx, 3]
        close = self.df.iloc[idx, 4]
        volume = self.df.iloc[idx, 5]
        Name = self.df.iloc[idx, 6]
        last7DaysMean = self.df.iloc[idx-7:idx, 4].mean()
        if math.isnan(last7DaysMean):
            last7DaysMean = 0.0
        last7WeeksMean = self.df.iloc[idx-7*7:idx, 4].mean()
        if math.isnan(last7WeeksMean):
            last7WeeksMean = 0.0
        stock = {'date': date, 
                'open': open, 
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,                                                
                'Name': Name,
                'last7DaysMean': last7DaysMean,
                'last7WeeksMean': last7WeeksMean}

        return stock

    def get_technical_indicators(self):
        # Create 7 and 21 days Moving Average
        self.df['ma7'] = self.df['close'].rolling(window=7).mean()
        self.df['ma21'] = self.df['close'].rolling(window=21).mean()
        s = pd.Series(self.df['close'])

        # Create MACD
        self.df['26ema'] = pd.Series.ewm(self.df['close'], span=26)
        self.df['12ema'] = pd.Series.ewm(self.df['close'], span=12)
        # self.df['MACD'] = (self.df['12ema'] - self.df['26ema'])
        # EMA_12 = pd.Series(self.df['close'].ewm(span=12, min_periods=12).mean())
        # EMA_26 = pd.Series(self.df['close'].ewm(span=26, min_periods=26).mean())
        # self.df['MACD'] = pd.Series(EMA_12 - EMA_26)
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1-exp2
        self.df['MACD'] = macd
        # exp3 = macd.ewm(span=9, adjust=False).mean()

        # Create Bollinger Bands
        self.df['20sd'] = s.rolling(20).std()
        self.df['upper_band'] = self.df['ma21'] + (self.df['20sd'] * 2)
        self.df['lower_band'] = self.df['ma21'] - (self.df['20sd'] * 2)

        # Create Exponential moving average
        # self.df['ema'] = self.df['close'].ewm(com=0.5).mean()
        self.df['ema'] = self.df['close'].ewm(span=20, adjust=False).mean()

        # Create Momentum
        # self.df['momentum'] = self.df['close'] - 1

        return self.df

    def get_feature_importance_data(self):
        data = self.df.copy()
        y = data['close']
        X = data.iloc[:, [4, 7,8,9,10,11,12,13,14,15]]
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
        # plt.plot(self.df['momentum'], label='Momentum', color='b', linestyle='-')
        # plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
        # plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
        plt.title('MACD')
        plt.legend()

        plt.show()
