from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import gym
import numpy as np
import torch
import random

from bindsnet.encoding import Encoder, NullEncoder
from bindsnet.environment import Environment
import numpy as np
import pandas as pd
from collections import deque

import gym
from gym import spaces
from gym.utils import seeding


class StockEnvironment(gym.Env):
    """Number guessing game
    The object of the game is to guess within 1% of the randomly chosen priceYesterday
    within 200 time steps
    After each step the agent is provided with one of four possible observations
    which indicate where the guess is in relation to the randomly chosen priceYesterday
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target
    The rewards are:
    0 if the agent's guess is outside of 1% of the target
    1 if the agent's guess is inside 1% of the target
    The episode terminates after the agent guesses within 1% of the target or
    200 steps have been taken
    The agent will need to use a memory of previously submitted actions and observations
    in order to efficiently explore the available actions
    The purpose is to have agents optimise their exploration parameters (e.g. how far to
    explore from previous actions) based on previous experience. Because the goal changes
    each episode a state-value or action-value function isn't able to provide any additional
    benefit apart from being able to tell whether to increase or decrease the next guess.
    The perfect agent would likely learn the bounds of the action space (without referring
    to them explicitly) and then follow binary tree style exploration towards to goal priceYesterday
    """

    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        # self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.model = load_model(model_name) if is_eval else self._model()
        self.df = pd.read_csv("AAPL_data.csv")

        self.action_space = spaces.Discrete(3) # sit, buy, sell
        self.action_space.n = 3
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([1000000]),
                                       dtype=np.float32)

        self.priceYesterday = 0
        self.index = 0
        # self.lenRecords = len(self.df)
        self.observation = 0

        self.seed()
        self.reset()

    def seed1(self, seed=None):
        if torch.is_tensor(self.index):
            self.index = self.index.tolist()

        close = self.df.iloc[self.index, 4]
        self.index += 1
        return self.env.seed(self, close)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        # action is sit, buy, sell

        reward = 0
        done = False

        # if not self.is_eval and random.random() <= self.epsilon:
        #     # return random.randrange(self.action_space.n)
        #     return self.observation, reward, done, {"priceYesterday": self.priceYesterday, "guesses": self.index}
        # options = self.model.predict(state)
        # return np.argmax(options[0])
        if torch.is_tensor(self.index):
            self.index = self.index.tolist()

        close = self.df.iloc[self.index, 4]
        self.observation = torch.FloatTensor([close])

        # if (self.observation - self.priceYesterday):
        #     reward = 1
        #     done = True

        return self.observation, reward, done, {"priceYesterday": self.priceYesterday, "index": self.index}

    def reset(self):
        self.priceYesterday = 0
        self.index = 0
        self.observation = 0
        return self.observation
