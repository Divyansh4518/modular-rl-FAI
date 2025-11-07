from agents.base_agent import RLAgent

class QLearningAgent(RLAgent):
    """Q-Learning agent implementing off-policy TD control."""
    
    def learn(self, state, action, reward, next_state):
        q = self.get_Q(state, action)
        next_qs = [self.get_Q(next_state, a) for a in self.actions]
        target = reward + self.gamma * max(next_qs)
        self.Q[(state, action)] = q + self.alpha * (target - q)
