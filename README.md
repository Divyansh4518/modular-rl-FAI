# 🎮 Modular Reinforcement Learning Framework

A clean, extensible reinforcement learning framework for experimenting with different environments and learning algorithms. Built with simplicity and modularity in mind—plug in any game and any RL algorithm!

## 📁 Project Structure

```
rl_framework/
│
├── envs/                           # Game environments
│   ├── __init__.py
│   ├── base_env.py                # Abstract base class for environments
│   ├── gridworld.py               # GridWorld with cliffs
│   ├── maze.py                    # Maze navigation with walls
│   └── tic_tac_toe.py             # Tic-Tac-Toe with multiple opponents
│
├── agents/                         # Learning algorithms
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base class for agents
│   ├── sarsa_agent.py             # SARSA (on-policy TD)
│   ├── qlearning_agent.py         # Q-Learning (off-policy TD)
│   ├── expected_sarsa_agent.py    # Expected SARSA (hybrid)
│   └── dqn_agent.py               # Deep Q-Network (neural network)
│
├── main.py                         # Training script with SARSA vs Q-Learning demo
├── utils.py                        # Visualization and utility functions
│
├── test_maze.py                    # Compare all agents on maze (NEW!)
├── quick_test_maze.py              # Preset testing configurations (NEW!)
├── test_single_agent.py            # Individual agent analysis (NEW!)
├── batch_test.py                   # Automated batch testing (NEW!)
│
├── MAZE_TESTING_README.md          # Testing suite documentation (NEW!)
├── TESTING_SCRIPTS_README.md       # Quick reference guide (NEW!)
├── MAZE_TESTING_GUIDE.md           # Comprehensive testing guide (NEW!)
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

Make sure you have the required dependencies:

```bash
# Core dependencies
pip install numpy matplotlib

# Optional: For Deep Q-Network (DQN) agent
pip install torch
```

### Run the Demo

```bash
cd rl_framework
python main.py
```

This will train both SARSA and Q-Learning agents on a GridWorld environment with cliffs and display a comparison plot.

## 🧪 NEW: Comprehensive Maze Testing Suite

We've added a complete testing framework for evaluating RL agents on maze environments!

### Quick Start with Testing Suite

```bash
# Compare all agents on default maze
python test_maze.py 2 500 200

# Quick preset test
python quick_test_maze.py 2

# Detailed analysis of single agent
python test_single_agent.py qlearning 2 500

# Automated batch testing
python batch_test.py 2
```

### What You Get

- ✅ **4 testing scripts** for different use cases
- ✅ **Automatic visualization** with professional plots
- ✅ **3 maze difficulties** (Simple 5×5, Default 7×7, Complex 10×10)
- ✅ **Multiple visualization types** (comparisons, policies, value functions)
- ✅ **Detailed metrics** (rewards, steps, success rates)
- ✅ **Batch processing** for comprehensive analysis

### Documentation

- 📘 **MAZE_TESTING_README.md** - Complete testing suite overview
- 📗 **TESTING_SCRIPTS_README.md** - Individual script documentation  
- 📙 **MAZE_TESTING_GUIDE.md** - Comprehensive usage guide

See the testing documentation for detailed usage instructions!

## ✨ What's Included

This framework comes with **3 fully implemented environments** and **4 learning algorithms** out of the box:

**Environments:**
- ✅ GridWorld (classic cliff navigation)
- ✅ Maze (customizable mazes with walls)
- ✅ Tic-Tac-Toe (with multiple opponent types)

**Algorithms:**
- ✅ SARSA (on-policy TD)
- ✅ Q-Learning (off-policy TD)
- ✅ Expected SARSA (hybrid approach)
- ✅ DQN (deep reinforcement learning)

**Features:**
- 🔌 Plug-and-play architecture
- 📊 Built-in visualization tools
- 💾 Model save/load (DQN)
- 🎮 Human vs Agent play mode
- 📈 Training progress tracking

## 🎯 Usage Examples

### Basic Training

```python
from envs.gridworld import GridWorld
from agents.qlearning_agent import QLearningAgent

# Create environment
env = GridWorld(width=5, height=5, cliffs=[(1,4), (2,4), (3,4)])

# Create agent
agent = QLearningAgent(
    actions=env.get_actions(),
    alpha=0.1,      # Learning rate
    gamma=0.9,      # Discount factor
    epsilon=0.1     # Exploration rate
)

# Train
for episode in range(500):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
```

### Training on Different Environments

```python
# Maze Environment
from envs.maze import create_simple_maze, create_complex_maze
env = create_simple_maze()
env.render()  # Visualize the maze

# Tic-Tac-Toe Environment
from envs.tic_tac_toe import TicTacToe
env = TicTacToe(opponent='optimal')  # 'random', 'optimal', or 'none'

# Train any agent on any environment
from agents.expected_sarsa_agent import ExpectedSARSAAgent
agent = ExpectedSARSAAgent(env.get_actions())
# ... training loop ...
```

### Using Different Algorithms

```python
from agents import SARSAAgent, QLearningAgent, ExpectedSARSAAgent

# Compare multiple algorithms
algorithms = {
    'SARSA': SARSAAgent(env.get_actions()),
    'Q-Learning': QLearningAgent(env.get_actions()),
    'Expected SARSA': ExpectedSARSAAgent(env.get_actions())
}

results = {}
for name, agent in algorithms.items():
    results[name] = train(env, agent, episodes=500)

# Plot comparison
from utils import plot_rewards
plot_rewards(results, title="Algorithm Comparison")
```

### Deep Q-Network (DQN) Example

```python
from agents.dqn_agent import DQNAgent, train_dqn
from envs.maze import create_simple_maze

env = create_simple_maze()

# DQN requires state dimension (for neural network input)
agent = DQNAgent(
    state_dim=2,  # (row, col) for maze
    action_list=env.get_actions(),
    hidden_dims=[128, 128],
    alpha=0.001,
    epsilon_decay=0.995
)

# Train with dedicated DQN training function
rewards = train_dqn(env, agent, episodes=1000)

# Save trained model
agent.save('models/dqn_maze.pth')
```

### Visualize Results

```python
from utils import print_policy, print_value_function, plot_rewards

# Show learned policy (for tabular agents)
print_policy(agent, width=5, height=5)

# Show value function
print_value_function(agent, width=5, height=5)

# Plot training progress
plot_rewards({'Q-Learning': rewards}, window_size=50)
```

## 🧩 How to Extend

### Adding a New Environment

Create a new file in `envs/` that inherits from `GameEnv`:

```python
# envs/maze.py
from envs.base_env import GameEnv

class Maze(GameEnv):
    def __init__(self, maze_layout):
        """Initialize your environment."""
        self.maze = maze_layout
        self.state = None
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def reset(self):
        """Reset to initial state and return it."""
        self.state = self.start_position
        return self.state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        """
        # Implement your game logic here
        next_state = self._compute_next_state(action)
        reward = self._compute_reward(next_state)
        done = self._check_terminal(next_state)
        
        self.state = next_state
        return next_state, reward, done, {}
    
    def get_actions(self, state=None):
        """Return available actions."""
        return self.actions
```

**Required Methods:**
- `reset()` → Returns initial state
- `step(action)` → Returns `(next_state, reward, done, info)`
- `get_actions(state)` → Returns list of valid actions

**Then import it in `envs/__init__.py`:**

```python
from .maze import Maze
__all__ = ['GameEnv', 'GridWorld', 'Maze']
```

### Adding a New Agent

Create a new file in `agents/` that inherits from `RLAgent`:

```python
# agents/monte_carlo_agent.py
from agents.base_agent import RLAgent

class MonteCarloAgent(RLAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        self.episode_history = []
    
    def learn(self, episode_data):
        """
        Implement your learning algorithm.
        
        Args:
            episode_data: List of (state, action, reward) tuples
        """
        # Your learning logic here
        G = 0
        for state, action, reward in reversed(episode_data):
            G = reward + self.gamma * G
            q = self.get_Q(state, action)
            self.Q[(state, action)] = q + self.alpha * (G - q)
```

**Key Methods to Override:**
- `learn(...)` → Update Q-values based on experience
- Optionally override `choose_action(state)` for custom exploration

**Then import it in `agents/__init__.py`:**

```python
from .monte_carlo_agent import MonteCarloAgent
__all__ = ['RLAgent', 'SARSAAgent', 'QLearningAgent', 'MonteCarloAgent']
```

## � Available Environments

### 1. GridWorld ✅
Classic grid navigation with cliffs and goal positions.
```python
from envs.gridworld import GridWorld
env = GridWorld(width=5, height=5, cliffs=[(1,4),(2,4),(3,4)])
```

### 2. Maze ✅
Navigate through mazes with walls and obstacles.
```python
from envs.maze import create_simple_maze, create_complex_maze
env = create_simple_maze()  # 5x5 maze
env = create_complex_maze()  # 10x10 challenging maze

# Custom maze
maze_layout = [[0,0,1,0], [0,1,0,0], [0,0,0,1], [1,0,0,0]]
env = Maze(maze_layout, start=(0,0), goal=(3,3))
```

### 3. Tic-Tac-Toe ✅
Play against different opponents.
```python
from envs.tic_tac_toe import TicTacToe, play_human_vs_agent

# Train against random opponent
env = TicTacToe(opponent='random')

# Train against optimal opponent
env = TicTacToe(opponent='optimal')

# Self-play (for two-agent training)
from envs.tic_tac_toe import TicTacToeSelfPlay
env = TicTacToeSelfPlay()

# Play against trained agent
play_human_vs_agent(trained_agent)
```

## 🧠 Available Algorithms

### 1. SARSA (State-Action-Reward-State-Action) ✅
**On-policy TD control** - Learns from actions actually taken.
```python
from agents.sarsa_agent import SARSAAgent
agent = SARSAAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1)
```

### 2. Q-Learning ✅
**Off-policy TD control** - Learns optimal policy regardless of actions taken.
```python
from agents.qlearning_agent import QLearningAgent
agent = QLearningAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1)
```

### 3. Expected SARSA ✅
**Hybrid approach** - Uses expected value over all possible actions.
```python
from agents.expected_sarsa_agent import ExpectedSARSAAgent
agent = ExpectedSARSAAgent(env.get_actions(), alpha=0.1, gamma=0.9, epsilon=0.1)
```

### 4. Deep Q-Network (DQN) ✅
**Neural network-based** - Handles complex state spaces with function approximation.
```python
from agents.dqn_agent import DQNAgent, train_dqn

agent = DQNAgent(
    state_dim=2,              # Input size for neural network
    action_list=env.get_actions(),
    hidden_dims=[128, 128],   # Network architecture
    alpha=0.001,              # Learning rate
    gamma=0.99,               # Discount factor
    epsilon=1.0,              # Initial exploration
    epsilon_decay=0.995,      # Decay rate
    batch_size=64,            # Replay batch size
    buffer_size=10000         # Replay buffer size
)

# Train with dedicated function
rewards = train_dqn(env, agent, episodes=1000)
```

## 📊 Algorithm Comparison

| Algorithm | Type | Best For | Pros | Cons |
|-----------|------|----------|------|------|
| **SARSA** | On-policy | Safe exploration | Stable, conservative | Slower convergence |
| **Q-Learning** | Off-policy | Optimal learning | Fast, optimal policy | Can be unstable |
| **Expected SARSA** | Hybrid | Balanced approach | Stable + efficient | Moderate complexity |
| **DQN** | Deep RL | Complex states | Handles high dimensions | Requires more data |

## 📊 Training Tips

### Hyperparameter Tuning

```python
# Learning rate (alpha): Controls how much new information overrides old
alpha = 0.1  # Start with 0.1, decrease for stability

# Discount factor (gamma): How much to value future rewards
gamma = 0.9  # 0.9-0.99 for most tasks

# Exploration rate (epsilon): Probability of random action
epsilon = 0.1  # Start with 0.1-0.3, decay over time
```

### Epsilon Decay

```python
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

for episode in range(num_episodes):
    agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
    # ... training code
```

## 🎨 Visualization

The framework includes utilities for analyzing learned policies:

```python
from utils import print_policy, print_value_function, plot_rewards

# Print policy as arrows
print_policy(agent, width=5, height=5)
# Output:
# → → → → ↓
# ↑ ← ← ↓ ↓
# ↑ ← ← ← ↓
# ↑ → → → ↓
# → → → → G

# Print value function
print_value_function(agent, width=5, height=5)

# Plot multiple agents
plot_rewards({
    'SARSA': sarsa_rewards,
    'Q-Learning': qlearn_rewards
}, window_size=50, title="Algorithm Comparison")
```

## 🔬 Comparing Algorithms

```python
from main import train

algorithms = {
    'SARSA': SARSAAgent(env.get_actions()),
    'Q-Learning': QLearningAgent(env.get_actions()),
    'Expected SARSA': ExpectedSARSAAgent(env.get_actions())
}

results = {}
for name, agent in algorithms.items():
    print(f"Training {name}...")
    results[name] = train(env, agent, episodes=500)

plot_rewards(results, title="Algorithm Comparison")
```

## 📝 Design Philosophy

This framework follows key principles:

1. **Separation of Concerns**: Environments don't know about agents, agents don't know about specific environments
2. **Duck Typing**: As long as an environment implements `reset()`, `step()`, and `get_actions()`, any agent can use it
3. **Minimal Dependencies**: Only NumPy and Matplotlib required
4. **Easy to Extend**: Clear base classes and simple interfaces
5. **Educational**: Clean, readable code for learning RL concepts

## 🎯 Future Enhancements

Ideas for extending the framework:

1. **More environments**: CartPole, MountainCar, Blackjack, Custom games
2. **Advanced algorithms**: Actor-Critic, A3C, PPO, DDPG
3. **Multi-agent scenarios**: Competitive/cooperative games
4. **Model-based RL**: Add planning and world models
5. **Better visualization**: Live training plots, heatmaps, policy animations
6. **Hyperparameter optimization**: Auto-tuning with grid search or Bayesian optimization

## � Quick Reference

### Environment Interface
Every environment must implement:
```python
reset()                    # → initial_state
step(action)               # → (next_state, reward, done, info)
get_actions(state=None)    # → list of valid actions
```

### Agent Interface (Tabular)
Q-Learning, SARSA, Expected SARSA:
```python
choose_action(state)                        # Choose action using ε-greedy
learn(state, action, reward, next_state)    # Update Q-values
get_Q(state, action)                        # Get Q-value
```

### Agent Interface (DQN)
Deep Q-Network:
```python
choose_action(state)                                    # Choose action
store_experience(state, action, reward, next_state, done)  # Store in buffer
learn()                                                 # Train on batch
decay_epsilon()                                         # Update exploration
save(filepath) / load(filepath)                         # Persistence
```

### Utility Functions
```python
# Visualization
plot_rewards(rewards_dict, window_size, title)
print_policy(agent, width, height)
print_value_function(agent, width, height)

# Training
train(env, agent, episodes)           # For tabular agents
train_dqn(env, agent, episodes)       # For DQN agent
```

## �📚 Resources

- **Sutton & Barto**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- **OpenAI Gym**: [gym.openai.com](https://gym.openai.com)
- **Deep RL Course**: [Hugging Face Deep RL](https://huggingface.co/learn/deep-rl-course)
- **Q-Learning Tutorial**: Understanding the Bellman equation
- **DQN Paper**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

## 🤝 Contributing

Feel free to extend this framework! The modular design makes it easy to:
- Add new environments in `envs/`
- Implement new algorithms in `agents/`
- Create visualization tools in `utils.py`

**To contribute:**
1. Fork the repository
2. Create your feature in the appropriate module
3. Test with existing environments/agents
4. Submit a pull request

## 📄 License

Free to use for educational and research purposes.

---

**Built for learning, designed for experimentation.** 🚀🧠

Happy Reinforcement Learning! 🎮

---

**Happy Learning!** 🚀🧠🎮
