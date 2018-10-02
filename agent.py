import numpy as np
from collections import defaultdict
from q_table import QTable


class Agent:

    def __init__(
        self, 
        observation_space=500,
        action_space=6, 
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9999,
        epsilon_min=0.01
    ):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA               = action_space
        self.possible_actions = np.arange(self.nA)
        self.epsilon_decay    = epsilon_decay
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.q_table          = QTable(
            observation_space=observation_space,
            action_space=action_space, 
            alpha=alpha, 
            gamma=gamma
        )

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def epsilon_greedy(self, state):
        policy                  = np.ones(self.nA) * (self.epsilon/self.nA)
        best_action_idx         = np.argmax(self.q_table.q(state))
        policy[best_action_idx] = (1 - self.epsilon) + (self.epsilon / self.nA)
        return policy
        
    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action_probabilities  = self.epsilon_greedy(state)
        return np.random.choice(self.possible_actions, p=action_probabilities)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.q_table.sarsa_max_update(state, action, reward, next_state)