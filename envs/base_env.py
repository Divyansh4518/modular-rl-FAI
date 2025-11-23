class GameEnv:
    """Base class for all game environments."""
    
    def reset(self):
        """Reset the environment to initial state and return it."""
        raise NotImplementedError

    def step(self, action):
        """
        Execute the action.
        Returns (next_state, reward, done, info)
        """
        raise NotImplementedError

    def get_actions(self, state=None):
        """Return possible actions from current state."""
        raise NotImplementedError
