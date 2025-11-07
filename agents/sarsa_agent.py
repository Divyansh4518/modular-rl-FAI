from agents.base_agent import RLAgent

class SARSAAgent(RLAgent):
    """SARSA (State-Action-Reward-State-Action) learning agent."""
    
    def learn(self, state, action, reward, next_state, next_action):
        q = self.get_Q(state, action)
        target = reward + self.gamma * self.get_Q(next_state, next_action)
        self.Q[(state, action)] = q + self.alpha * (target - q)
