"""Agents package for the RL framework."""

from .base_agent import RLAgent
from .sarsa_agent import SARSAAgent
from .qlearning_agent import QLearningAgent
from .expected_sarsa_agent import ExpectedSARSAAgent

# DQN requires PyTorch - only import if available
try:
    from .dqn_agent import DQNAgent, train_dqn
    __all__ = ['RLAgent', 'SARSAAgent', 'QLearningAgent', 'ExpectedSARSAAgent', 'DQNAgent', 'train_dqn']
except ImportError:
    __all__ = ['RLAgent', 'SARSAAgent', 'QLearningAgent', 'ExpectedSARSAAgent']
