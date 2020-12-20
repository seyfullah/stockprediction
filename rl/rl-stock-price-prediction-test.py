# https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/
import math
import time

import numpy as np
from keras.models import load_model

import Agent

start = time.time()

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


stock = 'AAPL'
stock_name = 'HDB'#input("Enter Stock_name, Model_name")
model_name = '0'#input()
model_name = stock + "-64-0"
model = load_model(model_name)
window_size = model.layers[0].input.shape.as_list()[1]
agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
print(data)
l = len(data) - 1
batch_size = 32
state = getState(data, 0, window_size + 1)
print(state)
total_profit = 0
agent.inventory = []
print(l)
for t in range(l):
    action = agent.act(state)
    print(action)
    # sit
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0
    if action == 1: # buy
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))
    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    if done:
        print("--------------------------------")
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")
        print ("Total profit is:",formatPrice(total_profit))

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))