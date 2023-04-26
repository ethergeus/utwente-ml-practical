import numpy as np
import random


class State:
    def __init__(self, state):
        self.state = state
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        return self.state == other.state


class QTable:
    def __init__(self,  actions: list, alpha=.1, gamma=.9, epsilon=.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}

    def epsilon_greedy(self, state):
        r = np.random.uniform(0, 1)

        if r < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.eval_greedy(state)
    
    def init_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}

    def update_q(self, state, action, reward, next_state):
        self.q_table[state][action] += self.alpha * (reward + self.gamma * (self.eval_greedy(next_state) - self.q_table[state][action]))

    def eval_greedy(self, state):
        return max(self.q_table[state], key=lambda a: self.q_table[state][a])