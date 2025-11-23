"""
Interactive single-agent testing script for all environments.
Allows detailed analysis of specific agents on Maze, GridWorld, or Tic-Tac-Toe.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.maze import Maze, create_simple_maze, create_complex_maze
from envs.gridworld import GridWorld
from envs.tic_tac_toe import TicTacToe
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.expected_sarsa_agent import ExpectedSARSAAgent

# --- Helper Functions ---

def get_agent(agent_type, env, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Factory function to create agents."""
    actions = env.get_actions()
    if agent_type.lower() == 'sarsa':
        return SARSAAgent(actions, alpha, gamma, epsilon)
    elif agent_type.lower() == 'expected_sarsa':
        return ExpectedSARSAAgent(actions, alpha, gamma, epsilon)
    else:
        return QLearningAgent(actions, alpha, gamma, epsilon)

def train_generic(env, agent, episodes, max_steps=200):
    """Generic training loop for Maze/GridWorld."""
    rewards = []
    steps_list = []
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if isinstance(agent, SARSAAgent):
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)
            
            state = next_state
            
        rewards.append(total_reward)
        steps_list.append(steps)
        
        if (ep + 1) % (episodes // 10) == 0:
            print(f"Episode {ep+1}/{episodes} - Avg Reward: {np.mean(rewards[-50:]):.1f}")
            
    return rewards, steps_list

def train_ttt(env, agent, episodes):
    """Training loop for Tic-Tac-Toe."""
    results = []
    initial_epsilon = agent.epsilon
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False
        
        # Decay epsilon
        agent.epsilon = max(0.05, initial_epsilon * (0.99995 ** ep))
        
        while not done:
            next_state, reward, done, info = env.step(action)
            
            if isinstance(agent, SARSAAgent):
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)
            
            state = next_state
            
        if info.get('winner') == 'X': results.append(1)
        elif info.get('winner') == 'O': results.append(-1)
        else: results.append(0)
        
        if (ep + 1) % (episodes // 10) == 0:
            win_rate = (results[-1000:].count(1) / 1000) * 100
            print(f"Episode {ep+1}/{episodes} - Win Rate: {win_rate:.1f}%")
            
    return results

# --- Visualization Functions ---

def plot_learning(data, title, ylabel):
    """Simple moving average plot."""
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(data, alpha=0.3, label='Raw')
    
    window = max(1, len(data) // 50)
    ma = [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
    plt.plot(ma, label=f'Moving Avg (w={window})', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"single_agent_{title.lower().replace(' ', '_')}.png"
    save_path = os.path.join('results', filename)
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")

# --- Main Logic ---

def run_maze_test():
    print("\n--- Maze Configuration ---")
    print("1. Simple (5x5)")
    print("2. Default (7x7)")
    print("3. Complex (10x10)")
    c = input("Choice [2]: ").strip() or '2'
    
    if c == '1': env = create_simple_maze()
    elif c == '3': env = create_complex_maze()
    else: env = Maze()
    
    agent_type = input("Agent (qlearning/sarsa/expected_sarsa) [qlearning]: ").strip() or 'qlearning'
    episodes = int(input("Episodes [500]: ").strip() or 500)
    
    agent = get_agent(agent_type, env)
    print(f"\nTraining {agent_type} on Maze...")
    rewards, steps = train_generic(env, agent, episodes)
    
    plot_learning(rewards, f"Maze {agent_type} Rewards", "Total Reward")
    plot_learning(steps, f"Maze {agent_type} Steps", "Steps to Goal")

def run_gridworld_test():
    print("\n--- GridWorld Configuration ---")
    print("Using Standard Cliff Walking (5x5)")
    env = GridWorld(width=5, height=5, start=(0, 4), goal=(4, 4), cliffs=[(1, 4), (2, 4), (3, 4)])
    
    agent_type = input("Agent (qlearning/sarsa/expected_sarsa) [qlearning]: ").strip() or 'qlearning'
    episodes = int(input("Episodes [500]: ").strip() or 500)
    
    agent = get_agent(agent_type, env)
    print(f"\nTraining {agent_type} on GridWorld...")
    rewards, steps = train_generic(env, agent, episodes)
    
    plot_learning(rewards, f"GridWorld {agent_type} Rewards", "Total Reward")
    plot_learning(steps, f"GridWorld {agent_type} Steps", "Steps to Goal")

def run_ttt_test():
    print("\n--- Tic-Tac-Toe Configuration ---")
    opp = input("Opponent (random/optimal) [random]: ").strip() or 'random'
    env = TicTacToe(opponent=opp)
    
    agent_type = input("Agent (qlearning/sarsa) [qlearning]: ").strip() or 'qlearning'
    episodes = int(input("Episodes [10000]: ").strip() or 10000)
    
    agent = get_agent(agent_type, env, epsilon=1.0) # Start with high exploration
    print(f"\nTraining {agent_type} on Tic-Tac-Toe vs {opp}...")
    results = train_ttt(env, agent, episodes)
    
    # Calculate win rate
    win_rates = []
    window = 1000
    for i in range(len(results)):
        start = max(0, i - window + 1)
        win_rates.append(results[start:i+1].count(1) / len(results[start:i+1]) * 100)
        
    plot_learning(win_rates, f"TTT {agent_type} vs {opp} Win Rate", "Win Rate (%)")

if __name__ == "__main__":
    print("="*60)
    print("INTERACTIVE SINGLE AGENT TESTER")
    print("="*60)
    print("1. Maze")
    print("2. GridWorld")
    print("3. Tic-Tac-Toe")
    
    choice = input("\nSelect Environment (1-3): ").strip()
    
    if choice == '1':
        run_maze_test()
    elif choice == '2':
        run_gridworld_test()
    elif choice == '3':
        run_ttt_test()
    else:
        print("Invalid choice.")

