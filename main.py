from envs.gridworld import GridWorld
from agents.sarsa_agent import SARSAAgent
from agents.qlearning_agent import QLearningAgent

def train(env, agent, episodes=500):
    """
    Train an agent in the given environment.
    
    Args:
        env: Environment instance
        agent: Agent instance (SARSAAgent or QLearningAgent)
        episodes: Number of training episodes
        
    Returns:
        List of total rewards per episode
    """
    rewards_per_episode = []
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if isinstance(agent, SARSAAgent):
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            else:  # Q-Learning
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)

            state = next_state

        rewards_per_episode.append(total_reward)
        
        # Print progress every 100 episodes
        if (ep + 1) % 100 == 0:
            avg_reward = sum(rewards_per_episode[-100:]) / 100
            print(f"Episode {ep + 1}/{episodes} - Avg Reward (last 100): {avg_reward:.2f}")
    
    return rewards_per_episode


if __name__ == "__main__":
    # Create environment with cliffs
    env = GridWorld(cliffs=[(1,4),(2,4),(3,4)])

    # Create agents
    sarsa = SARSAAgent(env.get_actions())
    qlearn = QLearningAgent(env.get_actions())

    # Train agents
    print("Training SARSA agent...")
    sarsa_rewards = train(env, sarsa)
    
    print("\nTraining Q-Learning agent...")
    qlearn_rewards = train(env, qlearn)

    # Plot comparison
    import matplotlib.pyplot as plt
    
    # Calculate moving average for smoother plots
    def moving_average(data, window_size=50):
        if len(data) < window_size:
            return data
        return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]
    
    sarsa_ma = moving_average(sarsa_rewards)
    qlearn_ma = moving_average(qlearn_rewards)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_ma, label='SARSA', alpha=0.8)
    plt.plot(qlearn_ma, label='Q-Learning', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Moving Average)')
    plt.title("SARSA vs Q-Learning in GridWorld")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n=== Training Complete ===")
    print(f"SARSA - Final 100 episodes avg reward: {sum(sarsa_rewards[-100:])/100:.2f}")
    print(f"Q-Learning - Final 100 episodes avg reward: {sum(qlearn_rewards[-100:])/100:.2f}")
