from __future__ import print_function, division

# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from StockDataset import StockDataset

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

stock_frame = pd.read_csv('AAPL_data.csv')

n = 1
date = stock_frame.iloc[n, 0]
prices = stock_frame.iloc[n, 1:5]
prices = np.asarray(prices)
prices = prices.astype('float').reshape(-1, 2)

stock_dataset = StockDataset(csv_file='AAPL_data.csv')

stock_dataset.plot_technical_indicators(len(stock_dataset.df))

# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = stock_dataset.get_feature_importance_data()

print(len(X_test_FI))

answer = input("Devam mı?")
