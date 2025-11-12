"""
Individual agent testing script for detailed analysis of a single agent.

This script allows you to test a single agent in detail and visualize
its learning process, Q-table heatmap, and policy.
"""

import numpy as np
import matplotlib.pyplot as plt
from envs.maze import Maze, create_simple_maze, create_complex_maze
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.expected_sarsa_agent import ExpectedSARSAAgent


def train_single_agent(env, agent, episodes=500, max_steps=200, verbose=True):
    """Train a single agent and collect detailed metrics."""
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
            next_state, reward, done, info = env.step(action)
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
        
        if verbose and (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} - "
                  f"Reward: {total_reward:.0f}, Steps: {steps}, "
                  f"Success Rate: {np.mean(successes[-50:])*100:.1f}%")
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'successes': successes
    }


def visualize_policy(env, agent, title="Learned Policy"):
    """Visualize the learned policy as arrows on the maze."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze
    maze_display = np.ones((env.height, env.width, 3))
    for i in range(env.height):
        for j in range(env.width):
            if env.maze[i, j] == 1:  # Wall
                maze_display[i, j] = [0.2, 0.2, 0.2]  # Dark gray
            else:  # Path
                maze_display[i, j] = [0.9, 0.9, 0.9]  # Light gray
    
    # Mark start and goal
    maze_display[env.start[0], env.start[1]] = [0.3, 0.8, 0.3]  # Green
    maze_display[env.goal[0], env.goal[1]] = [0.8, 0.3, 0.3]    # Red
    
    ax.imshow(maze_display)
    
    # Draw policy arrows
    action_arrows = {
        'U': (0, -0.3),
        'D': (0, 0.3),
        'L': (-0.3, 0),
        'R': (0.3, 0)
    }
    
    for i in range(env.height):
        for j in range(env.width):
            if env.maze[i, j] == 0 and (i, j) != env.goal:
                state = (i, j)
                # Get best action
                q_values = [agent.get_Q(state, a) for a in agent.actions]
                best_action = agent.actions[np.argmax(q_values)]
                
                dx, dy = action_arrows[best_action]
                ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.1,
                        fc='blue', ec='blue', alpha=0.7, linewidth=2)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Labels
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Start'),
        Patch(facecolor='red', alpha=0.5, label='Goal'),
        Patch(facecolor='gray', label='Path'),
        Patch(facecolor='black', label='Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig


def visualize_value_function(env, agent, title="Value Function (Max Q)"):
    """Visualize the value function as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate max Q-value for each state
    value_map = np.full((env.height, env.width), np.nan)
    
    for i in range(env.height):
        for j in range(env.width):
            if env.maze[i, j] == 0:  # Only for valid positions
                state = (i, j)
                q_values = [agent.get_Q(state, a) for a in agent.actions]
                value_map[i, j] = max(q_values) if q_values else 0
    
    # Create masked array to hide walls
    value_map_masked = np.ma.masked_invalid(value_map)
    
    # Plot heatmap
    im = ax.imshow(value_map_masked, cmap='RdYlGn', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Q-Value', rotation=270, labelpad=20, fontsize=12)
    
    # Mark start and goal
    ax.plot(env.start[1], env.start[0], 'g*', markersize=20, label='Start')
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=20, label='Goal')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # Labels
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_learning_curves(results, agent_name):
    """Plot detailed learning curves for a single agent."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{agent_name} - Detailed Learning Analysis', 
                 fontsize=16, fontweight='bold')
    
    episodes = range(1, len(results['rewards']) + 1)
    
    # Plot 1: Raw rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, results['rewards'], alpha=0.3, color='blue')
    window = 50
    ma = [np.mean(results['rewards'][max(0, i-window):i+1]) 
          for i in range(len(results['rewards']))]
    ax1.plot(episodes, ma, color='blue', linewidth=2, label='Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Rewards over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps per episode
    ax2 = axes[0, 1]
    ax2.plot(episodes, results['steps'], alpha=0.3, color='orange')
    ma_steps = [np.mean(results['steps'][max(0, i-window):i+1]) 
                for i in range(len(results['steps']))]
    ax2.plot(episodes, ma_steps, color='orange', linewidth=2, label='Moving Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rate
    ax3 = axes[1, 0]
    success_rate = [np.mean(results['successes'][max(0, i-window):i+1]) * 100
                    for i in range(len(results['successes']))]
    ax3.plot(episodes, success_rate, color='green', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title(f'Success Rate (Rolling Window: {window})')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution (last 100 episodes)
    ax4 = axes[1, 1]
    last_rewards = results['rewards'][-100:]
    ax4.hist(last_rewards, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(last_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(last_rewards):.1f}')
    ax4.set_xlabel('Total Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution (Last 100 Episodes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def test_agent(agent_type='qlearning', maze_choice='2', episodes=500):
    """
    Test a single agent with detailed visualization.
    
    Args:
        agent_type: 'qlearning', 'sarsa', or 'expected_sarsa'
        maze_choice: '1' (simple), '2' (default), '3' (complex)
        episodes: Number of training episodes
    """
    print("="*70)
    print(f"INDIVIDUAL AGENT TESTING - {agent_type.upper()}")
    print("="*70)
    
    # Create environment
    if maze_choice == '1':
        env = create_simple_maze()
        maze_name = "Simple Maze"
    elif maze_choice == '3':
        env = create_complex_maze()
        maze_name = "Complex Maze"
    else:
        env = Maze()
        maze_name = "Default Maze"
    
    print(f"\nEnvironment: {maze_name}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    env.render()
    
    # Create agent
    if agent_type.lower() == 'sarsa':
        agent = SARSAAgent(env.get_actions())
        agent_name = "SARSA"
    elif agent_type.lower() == 'expected_sarsa':
        agent = ExpectedSARSAAgent(env.get_actions())
        agent_name = "Expected SARSA"
    else:
        agent = QLearningAgent(env.get_actions())
        agent_name = "Q-Learning"
    
    print(f"\nTraining {agent_name} for {episodes} episodes...")
    print("-"*70)
    
    # Train agent
    results = train_single_agent(env, agent, episodes=episodes)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final 100 episodes:")
    print(f"  Avg Reward: {np.mean(results['rewards'][-100:]):.2f}")
    print(f"  Avg Steps:  {np.mean(results['steps'][-100:]):.1f}")
    print(f"  Success Rate: {np.mean(results['successes'][-100:])*100:.1f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Learning curves
    fig1 = plot_learning_curves(results, agent_name)
    plt.savefig(f'{agent_type}_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {agent_type}_learning_curves.png")
    
    # Policy visualization
    fig2 = visualize_policy(env, agent, f"{agent_name} - Learned Policy")
    plt.savefig(f'{agent_type}_policy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {agent_type}_policy.png")
    
    # Value function
    fig3 = visualize_value_function(env, agent, f"{agent_name} - Value Function")
    plt.savefig(f'{agent_type}_value_function.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {agent_type}_value_function.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    agent_type = 'qlearning'
    maze_choice = '2'
    episodes = 500
    
    if len(sys.argv) > 1:
        agent_type = sys.argv[1]
    if len(sys.argv) > 2:
        maze_choice = sys.argv[2]
    if len(sys.argv) > 3:
        episodes = int(sys.argv[3])
    
    print("\nUsage: python test_single_agent.py [agent] [maze] [episodes]")
    print("  agent: qlearning, sarsa, or expected_sarsa (default: qlearning)")
    print("  maze: 1=Simple, 2=Default, 3=Complex (default: 2)")
    print("  episodes: Number of training episodes (default: 500)")
    print(f"\nRunning: agent={agent_type}, maze={maze_choice}, episodes={episodes}\n")
    
    test_agent(agent_type=agent_type, maze_choice=maze_choice, episodes=episodes)
