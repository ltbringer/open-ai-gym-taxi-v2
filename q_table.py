import datetime
import numpy as np
import pandas as pd


class QTable(object):
    def __init__(
        self, 
        observation_space=500,
        action_space=6, 
        alpha=0.5, 
        gamma=0.9
    ):
        self.alpha             = alpha
        self.gamma             = gamma
        self.observation_space = observation_space
        self.action_space      = action_space
        self.__q               = np.zeros(self.observation_space * self.action_space)\
                                    .reshape((self.observation_space, self.action_space))

    def q(self, state=None, action=None):
        if state is None:
            return self.__q
        if action is None:
            return self.__q[state]
        return self.__q[state][action]
    
    def update_q(self, state, action, value):
        self.__q[state][action] = value

    def max_q(self, state):
        return np.max(self.__q[state])

    def old_value(self, state, action):
        return (1 - self.alpha) * self.q(state, action)

    def discounted_reward(self, state):
        return self.gamma * self.max_q(state)

    def sarsa_max_update(self, s, a, r, new_s):
        new_value = self.old_value(s, a) + (self.alpha * (r + self.discounted_reward(new_s) - self.q(s, a)))
        self.update_q(s, a, new_value)
        
    def save(self, score):
        timestamp = datetime.datetime.now().timestamp()
        timestamp_12_digit = int(timestamp * 1000)
        df = pd.DataFrame(self.__q)
        df.to_csv("alpha_{}_gamma_{}_score_{}__{}.csv".format(self.alpha, self.gamma, score, timestamp_12_digit))