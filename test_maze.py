"""
Testing script for training all RL agents on the Maze environment.

This script trains Q-Learning, SARSA, and Expected SARSA agents on a maze
and generates comprehensive analysis graphs for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from envs.maze import Maze, create_simple_maze, create_complex_maze
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.expected_sarsa_agent import ExpectedSARSAAgent


def train_agent(env, agent, episodes=1000, max_steps=200, verbose=True):
    """
    Train an agent in the given environment.
    
    Args:
        env: Environment instance (Maze)
        agent: Agent instance
        episodes: Number of training episodes
        max_steps: Maximum steps per episode (to prevent infinite loops)
        verbose: Whether to print progress
        
    Returns:
        dict containing:
            - rewards: List of total rewards per episode
            - steps: List of steps taken per episode
            - success_rate: List of success rates over time
    """
    rewards_per_episode = []
    steps_per_episode = []
    successes = []  # Track whether goal was reached
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Update agent based on agent type
            if isinstance(agent, SARSAAgent):
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            elif isinstance(agent, ExpectedSARSAAgent):
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)
            elif isinstance(agent, QLearningAgent):
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)
            
            state = next_state
        
        # Record metrics
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        successes.append(1 if done else 0)
        
        # Print progress
        if verbose and (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            success_rate = np.mean(successes[-100:]) * 100
            print(f"Episode {ep + 1}/{episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Steps: {avg_steps:.1f}, "
                  f"Success Rate: {success_rate:.1f}%")
    
    # Calculate cumulative success rate for plotting
    window = 50
    success_rate = []
    for i in range(len(successes)):
        start_idx = max(0, i - window + 1)
        success_rate.append(np.mean(successes[start_idx:i+1]) * 100)
    
    return {
        'rewards': rewards_per_episode,
        'steps': steps_per_episode,
        'success_rate': success_rate,
        'successes': successes
    }


def moving_average(data, window_size=50):
    """Calculate moving average for smoother plots."""
    if len(data) < window_size:
        window_size = len(data)
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        result.append(np.mean(data[start_idx:i+1]))
    return result


def plot_comparison(results_dict, maze_name="Maze"):
    """
    Create comprehensive comparison plots for all agents.
    
    Args:
        results_dict: Dictionary mapping agent names to their results
        maze_name: Name of the maze for the title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'RL Agents Performance Comparison on {maze_name}', 
                 fontsize=16, fontweight='bold')
    
    colors = {
        'Q-Learning': '#1f77b4',
        'SARSA': '#ff7f0e',
        'Expected SARSA': '#2ca02c'
    }
    
    # Plot 1: Rewards over time (moving average)
    ax1 = axes[0, 0]
    for agent_name, results in results_dict.items():
        rewards_ma = moving_average(results['rewards'], window_size=50)
        ax1.plot(rewards_ma, label=agent_name, 
                color=colors.get(agent_name, None), alpha=0.8, linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward (Moving Avg)', fontsize=12)
    ax1.set_title('Learning Curve: Rewards', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps per episode (moving average)
    ax2 = axes[0, 1]
    for agent_name, results in results_dict.items():
        steps_ma = moving_average(results['steps'], window_size=50)
        ax2.plot(steps_ma, label=agent_name, 
                color=colors.get(agent_name, None), alpha=0.8, linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps per Episode (Moving Avg)', fontsize=12)
    ax2.set_title('Learning Efficiency: Steps to Goal', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rate over time
    ax3 = axes[1, 0]
    for agent_name, results in results_dict.items():
        ax3.plot(results['success_rate'], label=agent_name, 
                color=colors.get(agent_name, None), alpha=0.8, linewidth=2)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('Success Rate (Rolling Average)', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Final performance comparison (bar chart)
    ax4 = axes[1, 1]
    agent_names = list(results_dict.keys())
    final_rewards = [np.mean(results['rewards'][-100:]) for results in results_dict.values()]
    final_steps = [np.mean(results['steps'][-100:]) for results in results_dict.values()]
    final_success = [np.mean(results['successes'][-100:]) * 100 for results in results_dict.values()]
    
    x = np.arange(len(agent_names))
    width = 0.25
    
    bars1 = ax4.bar(x - width, final_rewards, width, label='Avg Reward (last 100)', 
                    color='#1f77b4', alpha=0.8)
    bars2 = ax4.bar(x, final_steps, width, label='Avg Steps (last 100)', 
                    color='#ff7f0e', alpha=0.8)
    bars3 = ax4.bar(x + width, final_success, width, label='Success Rate % (last 100)', 
                    color='#2ca02c', alpha=0.8)
    
    ax4.set_xlabel('Agent', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Final Performance Metrics (Last 100 Episodes)', 
                  fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agent_names, fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('maze_agents_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'maze_agents_comparison.png'")
    plt.show()


def print_summary(results_dict):
    """Print detailed summary statistics for all agents."""
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY (Last 100 Episodes)")
    print("="*70)
    
    for agent_name, results in results_dict.items():
        rewards = results['rewards'][-100:]
        steps = results['steps'][-100:]
        successes = results['successes'][-100:]
        
        print(f"\n{agent_name}:")
        print(f"  Average Reward:    {np.mean(rewards):7.2f} (±{np.std(rewards):.2f})")
        print(f"  Average Steps:     {np.mean(steps):7.1f} (±{np.std(steps):.1f})")
        print(f"  Success Rate:      {np.mean(successes)*100:7.1f}%")
        print(f"  Min Reward:        {np.min(rewards):7.2f}")
        print(f"  Max Reward:        {np.max(rewards):7.2f}")
    
    print("\n" + "="*70)


def main(maze_choice='2', episodes=1000, max_steps=200):
    """
    Main testing function.
    
    Args:
        maze_choice: '1' for simple, '2' for default, '3' for complex
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
    """
    print("="*70)
    print("MAZE ENVIRONMENT - RL AGENTS TESTING")
    print("="*70)
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.9      # Discount factor
    epsilon = 0.1    # Exploration rate
    
    # Choose maze based on input
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
    print("\nMaze Layout:")
    env.render()
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Max Steps per Episode: {max_steps}")
    print(f"  Learning Rate (α): {alpha}")
    print(f"  Discount Factor (γ): {gamma}")
    print(f"  Exploration Rate (ε): {epsilon}")
    
    # Create agents
    agents = {
        'Q-Learning': QLearningAgent(env.get_actions(), alpha=alpha, gamma=gamma, epsilon=epsilon),
        'SARSA': SARSAAgent(env.get_actions(), alpha=alpha, gamma=gamma, epsilon=epsilon),
        'Expected SARSA': ExpectedSARSAAgent(env.get_actions(), alpha=alpha, gamma=gamma, epsilon=epsilon)
    }
    
    # Train all agents
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\n{'-'*70}")
        print(f"Training {agent_name}...")
        print(f"{'-'*70}")
        results[agent_name] = train_agent(env, agent, episodes=episodes, 
                                          max_steps=max_steps, verbose=True)
    
    # Print summary
    print_summary(results)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(results, maze_name=maze_name)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    maze_choice = '2'  # Default maze
    episodes = 1000
    max_steps = 200
    
    if len(sys.argv) > 1:
        maze_choice = sys.argv[1]
    if len(sys.argv) > 2:
        episodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        max_steps = int(sys.argv[3])
    
    print(f"\nUsage: python test_maze.py [maze_choice] [episodes] [max_steps]")
    print(f"  maze_choice: 1=Simple, 2=Default, 3=Complex (default: 2)")
    print(f"  episodes: Number of training episodes (default: 1000)")
    print(f"  max_steps: Max steps per episode (default: 200)")
    print(f"\nRunning with: maze={maze_choice}, episodes={episodes}, max_steps={max_steps}\n")
    
    main(maze_choice=maze_choice, episodes=episodes, max_steps=max_steps)
