"""
Main testing script for Maze environment.
Combines quick test presets and comprehensive comparison testing.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.maze import Maze, create_simple_maze, create_complex_maze
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.expected_sarsa_agent import ExpectedSARSAAgent

def train_agent(env, agent, episodes=1000, max_steps=200, verbose=True):
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
            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps_list[-100:])
            success_rate = np.mean(successes[-100:]) * 100
            print(f"Episode {ep + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Success: {success_rate:.1f}%")
    
    # Calculate cumulative success rate
    window = 50
    success_rate_curve = []
    for i in range(len(successes)):
        start_idx = max(0, i - window + 1)
        success_rate_curve.append(np.mean(successes[start_idx:i+1]) * 100)
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'success_rate': success_rate_curve,
        'successes': successes
    }

def plot_comparison(results_dict, maze_name="Maze"):
    """Create comparison plots."""
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'RL Agents Performance Comparison on {maze_name}', fontsize=16, fontweight='bold')
    
    colors = {'Q-Learning': '#1f77b4', 'SARSA': '#ff7f0e', 'Expected SARSA': '#2ca02c'}
    
    # 1. Rewards
    ax1 = axes[0, 0]
    for name, res in results_dict.items():
        # Moving average
        rewards = res['rewards']
        ma = [np.mean(rewards[max(0, i-50):i+1]) for i in range(len(rewards))]
        ax1.plot(ma, label=name, color=colors.get(name), alpha=0.8)
    ax1.set_title('Learning Curve: Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (Moving Avg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Steps
    ax2 = axes[0, 1]
    for name, res in results_dict.items():
        steps = res['steps']
        ma = [np.mean(steps[max(0, i-50):i+1]) for i in range(len(steps))]
        ax2.plot(ma, label=name, color=colors.get(name), alpha=0.8)
    ax2.set_title('Learning Efficiency: Steps to Goal')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps (Moving Avg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Success Rate
    ax3 = axes[1, 0]
    for name, res in results_dict.items():
        ax3.plot(res['success_rate'], label=name, color=colors.get(name), alpha=0.8)
    ax3.set_title('Success Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_ylim([0, 105])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Metrics Bar Chart
    ax4 = axes[1, 1]
    names = list(results_dict.keys())
    final_rewards = [np.mean(res['rewards'][-100:]) for res in results_dict.values()]
    final_steps = [np.mean(res['steps'][-100:]) for res in results_dict.values()]
    final_success = [np.mean(res['successes'][-100:]) * 100 for res in results_dict.values()]
    
    x = np.arange(len(names))
    width = 0.25
    
    ax4.bar(x - width, final_rewards, width, label='Avg Reward', color='#1f77b4', alpha=0.8)
    ax4.bar(x, final_steps, width, label='Avg Steps', color='#ff7f0e', alpha=0.8)
    ax4.bar(x + width, final_success, width, label='Success %', color='#2ca02c', alpha=0.8)
    
    ax4.set_title('Final Performance (Last 100 Episodes)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join('results', 'maze_agents_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved as '{save_path}'")
    # plt.show() # Uncomment to show interactively

def run_test(maze_choice='2', episodes=500, max_steps=200):
    """Run the full comparison test."""
    print("="*60)
    print("MAZE ENVIRONMENT - RL AGENTS TESTING")
    print("="*60)
    
    # Setup Environment
    if maze_choice == '1':
        env = create_simple_maze()
        maze_name = "Simple Maze (5x5)"
    elif maze_choice == '3':
        env = create_complex_maze()
        maze_name = "Complex Maze (10x10)"
    else:
        env = Maze()
        maze_name = "Default Maze (7x7)"
        
    print(f"\nSelected: {maze_name}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    
    # Setup Agents
    alpha, gamma, epsilon = 0.1, 0.9, 0.1
    agents = {
        'Q-Learning': QLearningAgent(env.get_actions(), alpha, gamma, epsilon),
        'SARSA': SARSAAgent(env.get_actions(), alpha, gamma, epsilon),
        'Expected SARSA': ExpectedSARSAAgent(env.get_actions(), alpha, gamma, epsilon)
    }
    
    # Train
    results = {}
    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        results[name] = train_agent(env, agent, episodes, max_steps)
        
    # Plot
    plot_comparison(results, maze_name)
    print("\nTest Complete!")

if __name__ == "__main__":
    print("\nSelect Test Mode:")
    print("1. Quick Test (Simple Maze, 200 ep)")
    print("2. Standard Test (Default Maze, 500 ep)")
    print("3. Comprehensive Test (Complex Maze, 1000 ep)")
    print("4. Custom (Enter arguments manually)")
    
    choice = input("\nEnter choice (1-4) [default=2]: ").strip() or '2'
    
    if choice == '1':
        run_test('1', 200, 100)
    elif choice == '3':
        run_test('3', 1000, 300)
    elif choice == '4':
        m = input("Maze (1/2/3): ")
        e = int(input("Episodes: "))
        s = int(input("Max Steps: "))
        run_test(m, e, s)
    else:
        run_test('2', 500, 200)
