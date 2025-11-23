"""
Comprehensive testing script for GridWorld environment.
Trains agents and generates policy/value visualizations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.expected_sarsa_agent import ExpectedSARSAAgent

def train_agent(env, agent, episodes=500, max_steps=200, verbose=True):
    """Train an agent and return metrics."""
    rewards = []
    steps_list = []
    successes = []
    
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
        successes.append(1 if done else 0)
        
        if verbose and (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} - Reward: {np.mean(rewards[-100:]):.1f}, Success: {np.mean(successes[-100:])*100:.1f}%")
            
    return {'rewards': rewards, 'steps': steps_list, 'successes': successes}

def visualize_grid_policy(env, agent, title="Learned Policy"):
    """Visualize policy on the grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    grid = np.zeros((env.height, env.width))
    ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
    
    # Draw grid lines
    for x in range(env.width + 1):
        ax.axvline(x - 0.5, color='black', lw=1)
    for y in range(env.height + 1):
        ax.axhline(y - 0.5, color='black', lw=1)
        
    # Mark start, goal, cliffs
    ax.text(env.start[0], env.start[1], 'S', ha='center', va='center', fontsize=20, color='green', fontweight='bold')
    ax.text(env.goal[0], env.goal[1], 'G', ha='center', va='center', fontsize=20, color='red', fontweight='bold')
    
    for cliff in env.cliffs:
        rect = plt.Rectangle((cliff[0]-0.5, cliff[1]-0.5), 1, 1, facecolor='black', alpha=0.5)
        ax.add_patch(rect)
        ax.text(cliff[0], cliff[1], 'XXX', ha='center', va='center', color='white', fontsize=10)

    # Draw arrows
    action_map = {'U': (0, -0.3), 'D': (0, 0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}
    
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state == env.goal or state in env.cliffs:
                continue
                
            q_values = [agent.get_Q(state, a) for a in agent.actions]
            best_action = agent.actions[np.argmax(q_values)]
            dx, dy = action_map[best_action]
            
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def visualize_value_heatmap(env, agent, title="Value Function"):
    """Visualize value function as heatmap."""
    value_grid = np.zeros((env.height, env.width))
    
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state in env.cliffs:
                value_grid[y, x] = -100 # Visual floor
            elif state == env.goal:
                value_grid[y, x] = 100
            else:
                qs = [agent.get_Q(state, a) for a in agent.actions]
                value_grid[y, x] = max(qs) if qs else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(value_grid, cmap='RdYlGn')
    plt.colorbar(im, label='Max Q-Value')
    
    # Mark cliffs
    for cliff in env.cliffs:
        ax.text(cliff[0], cliff[1], 'X', ha='center', va='center', color='white', fontweight='bold')
        
    ax.set_title(title)
    plt.tight_layout()
    return fig

def run_test(episodes=500):
    print("="*60)
    print("GRIDWORLD COMPREHENSIVE TEST")
    print("="*60)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Setup environment (Standard Cliff Walking)
    env = GridWorld(width=5, height=5, start=(0, 4), goal=(4, 4), 
                   cliffs=[(1, 4), (2, 4), (3, 4)])
    
    print(f"Grid: {env.width}x{env.height}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Cliffs: {env.cliffs}")
    
    agents = {
        'Q-Learning': QLearningAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1),
        'SARSA': SARSAAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1),
        'Expected SARSA': ExpectedSARSAAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1)
    }
    
    results = {}
    
    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        results[name] = train_agent(env, agent, episodes=episodes)
        
        # Visualize Policy
        fig = visualize_grid_policy(env, agent, title=f"{name} Policy")
        plt.savefig(os.path.join('results', f"gridworld_{name.lower().replace(' ', '_')}_policy.png"))
        plt.close()
        
        # Visualize Value
        fig = visualize_value_heatmap(env, agent, title=f"{name} Value Function")
        plt.savefig(os.path.join('results', f"gridworld_{name.lower().replace(' ', '_')}_value.png"))
        plt.close()
        
    # Comparison Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        rewards = res['rewards']
        ma = [np.mean(rewards[max(0, i-50):i+1]) for i in range(len(rewards))]
        plt.plot(ma, label=name)
    plt.title("Average Rewards (Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, res in results.items():
        steps = res['steps']
        ma = [np.mean(steps[max(0, i-50):i+1]) for i in range(len(steps))]
        plt.plot(ma, label=name)
    plt.title("Steps to Goal (Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', "gridworld_comparison.png"))
    print("\nSaved plots in 'results' directory:")
    print("- gridworld_comparison.png")
    print("- gridworld_*_policy.png")
    print("- gridworld_*_value.png")
    print("\nTest Complete!")

if __name__ == "__main__":
    episodes = int(input("Enter episodes [default=500]: ").strip() or 500)
    run_test(episodes)
