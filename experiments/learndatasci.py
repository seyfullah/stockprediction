# https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/

import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data

# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL', 'MSFT', '^GSPC']

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('INPX', 'yahoo', start_date, end_date)
# panel_data.to_frame().head(9)
print(panel_data.head(9))

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data['Close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')

print(all_weekdays)

# DatetimeIndex(['2010-01-01', '2010-01-04', '2010-01-05', '2010-01-06',
#               '2010-01-07', '2010-01-08', '2010-01-11', '2010-01-12',
#               '2010-01-13', '2010-01-14',
#               ...
#               '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22',
#               '2016-12-23', '2016-12-26', '2016-12-27', '2016-12-28',
#               '2016-12-29', '2016-12-30'],
#              dtype='datetime64[ns]', length=1826, freq='B')

print(close.head(10))

print(close.describe())

# Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
# msft = close.loc[:, 'MSFT']
msft = close
# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=20).mean()
long_rolling_msft = msft.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(msft.index, msft, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()
plt.show()

# https://www.learndatasci.com/tutorials/python-finance-part-2-intro-quantitative-trading-strategies/
import seaborn as sns
import numpy as np
import math
import matplotlib.dates as mdates

sns.set(style='darkgrid', context='talk', palette='Dark2')

# data = pd.read_pickle('./data.pkl')
data = msft
print("data.head(10)")
print(data.head(10))

# Calculating the short-window moving average

short_rolling = data.rolling(window=20).mean()
print("Calculating the short-window moving average")
print(short_rolling.head())

# Calculating the long-window moving average
long_rolling = data.rolling(window=100).mean()
print("Calculating the long-window moving average")
print(long_rolling.tail())

# Relative returns
returns = data.pct_change(1)
print("Relative returns")
print(returns.head())

# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
log_returns = np.log(data).diff()
print("Log returns")
print(log_returns.head())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,12))

#for c in log_returns:
#    if not math.isnan(c):
#        ax1.plot(log_returns.index, log_returns[c].cumsum(), label=str(c))

ax1.set_ylabel('Cumulative log returns')
ax1.legend(loc='best')

#for c in log_returns:
#    ax2.plot(log_returns.index, 100*(np.exp(log_returns[c].cumsum()) - 1), label=str(c))

ax2.set_ylabel('Total relative returns (%)')
ax2.legend(loc='best')

plt.show()

# Last day returns. Make this a column vector
r_t = log_returns.tail(1).transpose()
print("Last day returns.")
print(r_t)

# # Weights as defined above
# weights_vector = pd.DataFrame(1 / 3, index=r_t.index, columns=r_t.columns)
# print("Weights as defined above")
# print(weights_vector)

# # Total log_return for the portfolio is:
# portfolio_log_return = weights_vector.transpose().dot(r_t)
# print("Total log_return for the portfolio is:")
# print(portfolio_log_return)
#
# weights_matrix = pd.DataFrame(1 / 3, index=data.index, columns=data.columns)
# print("weights_matrix.tail()")
# print(weights_matrix.tail())
#
# log_returns.head()
#
# # Initially the two matrices are multiplied. Note that we are only interested in the diagonal,
# # which is where the dates in the row-index and the column-index match.
# temp_var = weights_matrix.dot(log_returns.transpose())
# temp_var.head().iloc[:, 0:5]
#
# # The numpy np.diag function is used to extract the diagonal and then
# # a Series is constructed using the time information from the log_returns index
# portfolio_log_returns = pd.Series(np.diag(temp_var), index=log_returns.index)
# portfolio_log_returns.tail()
#
# total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,12))
#
# ax1.plot(portfolio_log_returns.index, portfolio_log_returns.cumsum())
# ax1.set_ylabel('Portfolio cumulative log returns')
#
# ax2.plot(total_relative_returns.index, 100 * total_relative_returns)
# ax2.set_ylabel('Portfolio total relative returns (%)')
#
# plt.show()
#
# # Calculating the time-related parameters of the simulation
# days_per_year = 52 * 5
# total_days_in_simulation = data.shape[0]
# number_of_years = total_days_in_simulation / days_per_year
#
# # The last data point will give us the total portfolio return
# total_portfolio_return = total_relative_returns[-1]
# # Average portfolio return assuming compunding of returns
# average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1
#
# print('Total portfolio return is: ' +
#       '{:5.2f}'.format(100 * total_portfolio_return) + '%')
# print('Average yearly return is: ' +
#       '{:5.2f}'.format(100 * average_yearly_return) + '%')

# https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/

start_date = '2015-01-01'
end_date = '2016-12-31'

# fig, ax = plt.subplots(figsize=(16,9))
#
# ax.plot(data.loc[start_date:end_date, :].index, data.loc[start_date:end_date, 'MSFT'], label='Price')
# ax.plot(long_rolling.loc[start_date:end_date, :].index, long_rolling.loc[start_date:end_date, 'MSFT'], label = '100-days SMA')
# ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date, 'MSFT'], label = '20-days SMA')
#
# ax.legend(loc='best')
# ax.set_ylabel('Price in $')
my_year_month_fmt = mdates.DateFormatter('%m/%y')
# ax.xaxis.set_major_formatter(my_year_month_fmt)
#
# plt.show()

# Using Pandas to calculate a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
ema_short = data.ewm(span=20, adjust=False).mean()

fig, ax = plt.subplots(figsize=(15,9))

ax.plot(data.loc[start_date:end_date, :].index, data.loc[start_date:end_date, 'MSFT'], label='Price')
ax.plot(ema_short.loc[start_date:end_date, :].index, ema_short.loc[start_date:end_date, 'MSFT'], label = 'Span 20-days EMA')
ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date, 'MSFT'], label = '20-days SMA')

ax.legend(loc='best')
ax.set_ylabel('Price in $')
ax.xaxis.set_major_formatter(my_year_month_fmt)

# Taking the difference between the prices and the EMA timeseries
trading_positions_raw = data - ema_short
trading_positions_raw.tail()

# Taking the sign of the difference to determine whether the price or the EMA is greater and then multiplying by 1/3
trading_positions = trading_positions_raw.apply(np.sign) * 1/3
trading_positions.tail()

# Lagging our trading signals by one day.
trading_positions_final = trading_positions.shift(1)

ig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

ax1.plot(data.loc[start_date:end_date, :].index, data.loc[start_date:end_date, 'MSFT'], label='Price')
ax1.plot(ema_short.loc[start_date:end_date, :].index, ema_short.loc[start_date:end_date, 'MSFT'], label = 'Span 20-days EMA')

ax1.set_ylabel('$')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(my_year_month_fmt)

ax2.plot(
    trading_positions_final.loc[start_date:end_date, :].index,
    trading_positions_final.loc[start_date:end_date, 'MSFT'],
    label='Trading position')

ax2.set_ylabel('Trading position')
ax2.xaxis.set_major_formatter(my_year_month_fmt)

# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
asset_log_returns = np.log(data).diff()
asset_log_returns.head()

strategy_asset_log_returns = trading_positions_final * asset_log_returns
strategy_asset_log_returns.tail()

# Get the cumulative log-returns per asset
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()

# Transform the cumulative log returns to relative returns
cum_strategy_asset_relative_returns = np.exp(cum_strategy_asset_log_returns) - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

for c in asset_log_returns:
    ax1.plot(cum_strategy_asset_log_returns.index, cum_strategy_asset_log_returns[c], label=str(c))

ax1.set_ylabel('Cumulative log-returns')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(my_year_month_fmt)

for c in asset_log_returns:
    ax2.plot(cum_strategy_asset_relative_returns.index,
             100*cum_strategy_asset_relative_returns[c], label=str(c))

ax2.set_ylabel('Total relative returns (%)')
ax2.legend(loc='best')
ax2.xaxis.set_major_formatter(my_year_month_fmt)

# Total strategy relative returns. This is the exact calculation.
cum_relative_return_exact = cum_strategy_asset_relative_returns.sum(axis=1)

# Get the cumulative log-returns per asset
cum_strategy_log_return = cum_strategy_asset_log_returns.sum(axis=1)

# Transform the cumulative log returns to relative returns. This is the approximation
cum_relative_return_approx = np.exp(cum_strategy_log_return) - 1

fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cum_relative_return_exact.index, 100*cum_relative_return_exact, label='Exact')
ax.plot(cum_relative_return_approx.index, 100*cum_relative_return_approx, label='Approximation')

ax.set_ylabel('Total cumulative relative returns (%)')
ax.legend(loc='best')
ax.xaxis.set_major_formatter(my_year_month_fmt)

def print_portfolio_yearly_statistics(portfolio_cumulative_relative_returns, days_per_year = 52 * 5):

    total_days_in_simulation = portfolio_cumulative_relative_returns.shape[0]
    number_of_years = total_days_in_simulation / days_per_year

    # The last data point will give us the total portfolio return
    total_portfolio_return = portfolio_cumulative_relative_returns[-1]
    # Average portfolio return assuming compunding of returns
    average_yearly_return = (1 + total_portfolio_return)**(1/number_of_years) - 1

    print('Total portfolio return is: ' + '{:5.2f}'.format(100*total_portfolio_return) + '%')
    print('Average yearly return is: ' + '{:5.2f}'.format(100*average_yearly_return) + '%')

print_portfolio_yearly_statistics(cum_relative_return_exact)

# Define the weights matrix for the simple buy-and-hold strategy
simple_weights_matrix = pd.DataFrame(1/3, index = data.index, columns=data.columns)

# Get the buy-and-hold strategy log returns per asset
simple_strategy_asset_log_returns = simple_weights_matrix * asset_log_returns

# Get the cumulative log-returns per asset
simple_cum_strategy_asset_log_returns = simple_strategy_asset_log_returns.cumsum()

# Transform the cumulative log returns to relative returns
simple_cum_strategy_asset_relative_returns = np.exp(simple_cum_strategy_asset_log_returns) - 1

# Total strategy relative returns. This is the exact calculation.
simple_cum_relative_return_exact = simple_cum_strategy_asset_relative_returns.sum(axis=1)

fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cum_relative_return_exact.index, 100*cum_relative_return_exact, label='EMA strategy')
ax.plot(simple_cum_relative_return_exact.index, 100*simple_cum_relative_return_exact, label='Buy and hold')

ax.set_ylabel('Total cumulative relative returns (%)')
ax.legend(loc='best')
ax.xaxis.set_major_formatter(my_year_month_fmt)
plt.show()

print_portfolio_yearly_statistics(simple_cum_relative_return_exact)
