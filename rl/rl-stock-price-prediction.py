# https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/
import math
import time

import numpy as np
import pandas_datareader as web

from Agent import Agent

import time
import datetime

start = time.time()
print("start:" + str(datetime.datetime.now()))


def formatPrice(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))


def getStockDataVec(key):
    vec = []
    lines = open(key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        # print(line)
        # print(float(line.split(",")[4]))
        vec.append(float(line.split(",")[4]))
        # print(vec)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


#stock_name = 'HDB'  # input("Enter stock_name: ")
window_size = '60'  # input("Enter window_size: ")
episode_count = '1'  # input("Enter Episode_count: ")
stock = 'AAPL'
#stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
agent = Agent(window_size)
#data1 = getStockDataVec(stock_name)
df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')
data = df.filter(['Close']).values.ravel().tolist()
l = len(data) - 1
batch_size = 32
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    print(str(range(l)) + " kere dönecek.")
    for t in range(l):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1:  # buy
            agent.inventory.append(data[t])
            print(str(t) + " Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print(str(t) + " Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    if e % 1 == 0:
        agent.model.save(stock + "-" + str(window_size) + "-" + str(e))
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("end:  " + str(datetime.datetime.now()))
print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
