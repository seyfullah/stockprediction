from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from StockDataset import StockDataset
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

stock_frame = pd.read_csv('AAPL_data.csv')

n = 1
date = stock_frame.iloc[n, 0]
prices = stock_frame.iloc[n, 1:5]
prices = np.asarray(prices)
prices = prices.astype('float').reshape(-1, 2)

stock_dataset = StockDataset(csv_file='AAPL_data.csv')

dates = []
closes = []
for i, stock in enumerate(stock_dataset):
    dates.append(stock['date'])
    closes.append(stock['close'])

# Visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(closes)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
plt.show()

answer = input("Devam?")