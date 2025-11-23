import random
import numpy as np

class RLAgent:
    """Base class for all reinforcement learning agents."""
    
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, a) for a in self.actions]
        return self.actions[np.argmax(qs)]
