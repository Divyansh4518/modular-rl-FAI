"""Utility functions for the RL framework."""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=50):
    """
    Calculate moving average of a list of values.
    
    Args:
        data: List of numerical values
        window_size: Size of the moving window
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]


def plot_rewards(rewards_dict, window_size=50, title="Training Progress"):
    """
    Plot rewards for multiple agents.
    
    Args:
        rewards_dict: Dictionary mapping agent names to reward lists
        window_size: Size of moving average window
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for agent_name, rewards in rewards_dict.items():
        smoothed = moving_average(rewards, window_size)
        plt.plot(smoothed, label=agent_name, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Moving Average)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_policy(agent, width, height):
    """
    Print the learned policy for a grid world.
    
    Args:
        agent: Trained agent
        width: Width of the grid
        height: Height of the grid
    """
    action_symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
    
    print("\nLearned Policy:")
    for y in range(height):
        row = []
        for x in range(width):
            state = (x, y)
            qs = [agent.get_Q(state, a) for a in agent.actions]
            best_action = agent.actions[np.argmax(qs)]
            row.append(action_symbols[best_action])
        print(' '.join(row))


def print_value_function(agent, width, height):
    """
    Print the value function for a grid world.
    
    Args:
        agent: Trained agent
        width: Width of the grid
        height: Height of the grid
    """
    print("\nValue Function:")
    for y in range(height):
        row = []
        for x in range(width):
            state = (x, y)
            qs = [agent.get_Q(state, a) for a in agent.actions]
            value = max(qs) if qs else 0
            row.append(f"{value:6.1f}")
        print(' '.join(row))
