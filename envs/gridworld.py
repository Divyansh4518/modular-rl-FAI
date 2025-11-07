import numpy as np
from envs.base_env import GameEnv

class GridWorld(GameEnv):
    def __init__(self, width=5, height=5, start=(0,0), goal=(4,4), cliffs=[]):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.cliffs = cliffs
        self.state = start
        self.actions = ['U', 'D', 'L', 'R']

    def reset(self):
        self.state = self.start
        return self.state

    def get_actions(self, state=None):
        return self.actions

    def step(self, action):
        x, y = self.state
        if action == 'U': y = max(0, y-1)
        elif action == 'D': y = min(self.height-1, y+1)
        elif action == 'L': x = max(0, x-1)
        elif action == 'R': x = min(self.width-1, x+1)

        next_state = (x, y)
        reward = -1
        done = False

        if next_state in self.cliffs:
            reward = -100
            next_state = self.start
        elif next_state == self.goal:
            reward = 100
            done = True

        self.state = next_state
        return next_state, reward, done, {}
