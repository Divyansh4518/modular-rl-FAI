from agents.base_agent import RLAgent

class ExpectedSARSAAgent(RLAgent):
    """
    Expected SARSA agent.
    
    Instead of using the next action's Q-value (SARSA) or the max Q-value (Q-Learning),
    Expected SARSA uses the expected Q-value over all possible next actions,
    weighted by the policy probability.
    
    This makes it more stable than SARSA while being less aggressive than Q-Learning.
    """
    
    def learn(self, state, action, reward, next_state):
        """
        Update Q-value using Expected SARSA rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.get_Q(state, action)
        
        # Calculate expected Q-value for next state
        # This accounts for the exploration policy (epsilon-greedy)
        next_q_values = [self.get_Q(next_state, a) for a in self.actions]
        
        if not next_q_values:
            expected_q = 0
        else:
            # Expected value under epsilon-greedy policy:
            # - With probability (1-epsilon), take greedy action (max Q)
            # - With probability epsilon, take random action (average Q)
            max_q = max(next_q_values)
            avg_q = sum(next_q_values) / len(next_q_values)
            expected_q = (1 - self.epsilon) * max_q + self.epsilon * avg_q
        
        # Calculate target and update
        target = reward + self.gamma * expected_q
        self.Q[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def learn_deterministic(self, state, action, reward, next_state):
        """
        Alternative learning method assuming a deterministic (greedy) policy.
        
        This is simpler and just averages all next Q-values equally,
        which can be useful for environments where all actions are equally likely.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.get_Q(state, action)
        
        # Simple average of all next Q-values
        next_q_values = [self.get_Q(next_state, a) for a in self.actions]
        expected_q = sum(next_q_values) / len(next_q_values) if next_q_values else 0
        
        # Calculate target and update
        target = reward + self.gamma * expected_q
        self.Q[(state, action)] = current_q + self.alpha * (target - current_q)
