# https://towardsdatascience.com/aifortrading-2edd6fac689d
# https://github.com/borisbanushev/stockpredictionai
# https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md
# https://www.similarweb.com/website/apple.com/#similarSites

from utils import *

import time
import numpy as np


import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas_datareader as web

import warnings

import time
import datetime
import stockstats

warnings.filterwarnings("ignore")

start = time.time()
print("start:" + str(datetime.datetime.now()))
stock = 'AAPL'


def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')


# dataset_ex_df = pd.read_csv('data/panel_data_close.csv', header=0, parse_dates=[0], date_parser=parser)
df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')
# print(dataset_ex_df[['Date', 'GS']].head(3))
print(df.iloc[:, 3].head(3))
data = df.filter(['Close']).values.ravel().tolist()
print('There are {} number of days in the dataset.'.format(len(data)))

# Visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
# plt.vlines(datetime.date(2016, 4, 20), 0, 100, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
plt.show()

num_training_days = int(df.shape[0] * .7)
print(
    'Number of training days: {}. Number of test days: {}.'.format(num_training_days, df.shape[0] - num_training_days))


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    s = pd.Series(dataset['Close'])

    # Create MACD
    dataset['26ema'] = pd.Series.ewm(dataset['Close'], span=26)
    dataset['12ema'] = pd.Series.ewm(dataset['Close'], span=12)
    # dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
    EMA_12 = pd.Series(dataset['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(dataset['Close'].ewm(span=26, min_periods=26).mean())
    dataset['MACD'] = pd.Series(EMA_12 - EMA_26)

    # Create Bollinger Bands
    dataset['20sd'] = s.rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    # Create Exponential moving average
    # dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    dataset['ema'] = dataset['Close'].ewm(span=20, adjust=False).mean()

    # Create Momentum
    dataset['momentum'] = dataset['Close'] - 1

    return dataset


dataset_TI_df = get_technical_indicators(df)
print(dataset_TI_df.head())


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0] + (2014 - 1970) * 365
    xmacd_ = shape_0 - last_days
    shape_0 += 200

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Close'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for ' + stock + ' - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.title('MACD')
    plt.legend()

    plt.show()


plot_technical_indicators(dataset_TI_df, 400)

# 3.3. Fundamental analysis
# just import bert
# import bert

# data_FT = dataset_ex_df[['Date', 'GS']]
data_FT = df.iloc[:, 3]
# TODO sorunlu
# close_fft = np.fft.fft(np.asarray(data_FT['GS'].tolist()))
# close_fft = np.fft.fft(np.asarray(data_FT.tolist()))
close_fft = np.fft.fft(np.asarray(data_FT))
fft_df = pd.DataFrame({'fft': close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(14, 7), dpi=100)
# fft_list = np.asarray(fft_df['fft'].tolist())
fft_list = np.asarray(fft_df['fft'])
for num_ in [3, 6, 9, 100]:
    fft_list_m10 = np.copy(fft_list);
    fft_list_m10[num_:-num_] = 0
    ifft = np.fft.ifft(fft_list_m10)
    plt.plot(pd.Series(ifft, index=data_FT.keys()), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT, label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 3: ' + stock + ' (Close) stock prices & Fourier transforms')
plt.legend()
plt.show()

from collections import deque

items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df) / 2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Figure 4: Components of Fourier transforms')
plt.show()

# 3.5. ARIMA as a feature

from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

# series = data_FT['GS']
series = data_FT
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show()

from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# Plot the predicted (from ARIMA) and real prices

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on ' + stock + ' stock')
plt.legend()
plt.show()


# print('Total dataset has {} samples, and {} features.'.format(dataset_total_df.shape[0], dataset_total_df.shape[1]))


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['Close']
    X = data.iloc[:, 1:]

    train_samples = int(X.shape[0] * 0.65)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)


# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)

# from mxnet import nd, autograd, gluon
# from mxnet.gluon import nn, rnn
# import mxnet as mx

# context = mx.cpu();
# model_ctx = mx.cpu()
# mx.random.seed(1719)

# regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=150, base_score=0.7, colsample_bytree=1, learning_rate=0.05)


# xgbModel = regressor.fit(X_train_FI, y_train_FI,
#                          eval_set=[(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
#                          verbose=False)
#
# eval_result = regressor.evals_result()
#
# training_rounds = range(len(eval_result['validation_0']['rmse']))
#
# plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
# plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
# plt.xlabel('Iterations')
# plt.ylabel('RMSE')
# plt.title('Training Vs Validation Error')
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(8, 8))
# plt.xticks(rotation='vertical')
# plt.bar([i for i in range(len(xgbModel.feature_importances_))],
#         xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
# plt.title('Figure 6: Feature importance of the technical indicators.')
# plt.show()

# 3.8. Extracting high-level features with Stacked Autoencoders

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))


def relu(x):
    return max(x, 0)


def lrelu(x):
    return max(0.01 * x, x)


plt.figure(figsize=(15, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

ranges_ = (-10, 3, .25)

plt.subplot(1, 2, 1)
plt.plot([i for i in np.arange(*ranges_)], [relu(i) for i in np.arange(*ranges_)], label='ReLU', marker='.')
plt.plot([i for i in np.arange(*ranges_)], [gelu(i) for i in np.arange(*ranges_)], label='GELU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.title('Figure 7: GELU as an activation function for autoencoders')
plt.ylabel('f(x) for GELU and ReLU')
plt.xlabel('x')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([i for i in np.arange(*ranges_)], [lrelu(i) for i in np.arange(*ranges_)], label='Leaky ReLU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.ylabel('f(x) for Leaky ReLU')
plt.xlabel('x')
plt.title('Figure 8: LeakyReLU')
plt.legend()

plt.show()

batch_size = 64
# n_batches = VAE_data.shape[0] / batch_size
# VAE_data = VAE_data.values
#
# train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days, :-1]}, \
#                                label={'label': VAE_data[:num_training_days, -1]}, batch_size=batch_size)
# test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:, :-1]}, \
#                               label={'label': VAE_data[num_training_days:, -1]}, batch_size=batch_size)

# model_ctx = mx.cpu()


# class VAE(gluon.HybridBlock):
#     def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, \
#                  batch_size=100, act_type='relu', **kwargs):
#         self.soft_zero = 1e-10
#         self.n_latent = n_latent
#         self.batch_size = batch_size
#         self.output = None
#         self.mu = None
#         super(VAE, self).__init__(**kwargs)

#         with self.name_scope():
#             self.encoder = nn.HybridSequential(prefix='encoder')

#             for i in range(n_layers):
#                 self.encoder.add(nn.Dense(n_hidden, activation=act_type))
#             self.encoder.add(nn.Dense(n_latent * 2, activation=None))

#             self.decoder = nn.HybridSequential(prefix='decoder')
#             for i in range(n_layers):
#                 self.decoder.add(nn.Dense(n_hidden, activation=act_type))
#             self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

#     def hybrid_forward(self, F, x):
#         h = self.encoder(x)
#         # print(h)
#         mu_lv = F.split(h, axis=1, num_outputs=2)
#         mu = mu_lv[0]
#         lv = mu_lv[1]
#         self.mu = mu

#         eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
#         z = mu + F.exp(0.5 * lv) * eps
#         y = self.decoder(z)
#         self.output = y

#         KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
#         logloss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
#         loss = -logloss - KL

#         return loss


# n_hidden = 400  # neurons in each layer
# n_latent = 2
# n_layers = 3  # num of dense layers in encoder and decoder respectively
# n_output = VAE_data.shape[1] - 1
#
# net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size,
#           act_type='gelu')

# net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
# net.hybridize()
# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})
#
# print(net)

#
# n_epoch = 150
# print_period = n_epoch // 10
# start = time.time()
#
# training_loss = []
# validation_loss = []
# for epoch in range(n_epoch):
#     epoch_loss = 0
#     epoch_val_loss = 0
#
#     train_iter.reset()
#     test_iter.reset()
#
#     n_batch_train = 0
#     for batch in train_iter:
#         n_batch_train +=1
#         data = batch.data[0].as_in_context(mx.cpu())
#
#         with autograd.record():
#             loss = net(data)
#         loss.backward()
#         trainer.step(data.shape[0])
#         epoch_loss += nd.mean(loss).asscalar()
#
#     n_batch_val = 0
#     for batch in test_iter:
#         n_batch_val +=1
#         data = batch.data[0].as_in_context(mx.cpu())
#         loss = net(data)
#         epoch_val_loss += nd.mean(loss).asscalar()
#
#     epoch_loss /= n_batch_train
#     epoch_val_loss /= n_batch_val
#
#     training_loss.append(epoch_loss)
#     validation_loss.append(epoch_val_loss)
#
#     """if epoch % max(print_period, 1) == 0:
#         print('Epoch {}, Training loss {:.2f}, Validation loss {:.2f}'.\
#               format(epoch, epoch_loss, epoch_val_loss))"""


# # We want the PCA to create the new components to explain 80% of the variance
# pca = PCA(n_components=.8)
#
# x_pca = StandardScaler().fit_transform(vae_added_df)
#
# principalComponents = pca.fit_transform(x_pca)
#
# principalComponents.n_components_

end = time.time()
print('Training completed in {} seconds.'.format(int(end - start)))

# dataset_total_df['Date'] = dataset_ex_df['Date']
#
# vae_added_df = mx.nd.array(dataset_total_df.iloc[:, :-1].values)
#
# print('The shape of the newly created (from the autoencoder) features is {}.'.format(vae_added_df.shape))
#


end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("end:  " + str(datetime.datetime.now()))
print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
