# 🎮 Modular Reinforcement Learning Framework

A clean, extensible reinforcement learning framework for experimenting with different environments and learning algorithms. Built with simplicity and modularity in mind—plug in any game and any RL algorithm!

## 📁 Project Structure

```
rl_framework/
│
├── envs/                      # Game environments
│   ├── __init__.py
│   ├── base_env.py           # Abstract base class for environments
│   └── gridworld.py          # GridWorld implementation
│
├── agents/                    # Learning algorithms
│   ├── __init__.py
│   ├── base_agent.py         # Abstract base class for agents
│   ├── sarsa_agent.py        # SARSA algorithm
│   └── qlearning_agent.py    # Q-Learning algorithm
│
├── main.py                    # Training script with SARSA vs Q-Learning demo
├── utils.py                   # Visualization and utility functions
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

Make sure you have the required dependencies:

```bash
pip install numpy matplotlib
```

### Run the Demo

```bash
cd rl_framework
python main.py
```

This will train both SARSA and Q-Learning agents on a GridWorld environment with cliffs and display a comparison plot.

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

### Visualize Results

```python
from utils import print_policy, print_value_function, plot_rewards

# Show learned policy
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

## 🎲 Example Environments to Add

### Tic-Tac-Toe

```python
# envs/tic_tac_toe.py
from envs.base_env import GameEnv

class TicTacToe(GameEnv):
    def __init__(self):
        self.board = [[' ']*3 for _ in range(3)]
        self.actions = [(i, j) for i in range(3) for j in range(3)]
    
    def reset(self):
        self.board = [[' ']*3 for _ in range(3)]
        return tuple(map(tuple, self.board))
    
    def step(self, action):
        row, col = action
        self.board[row][col] = 'X'
        
        # Check win/draw conditions
        if self._check_win('X'):
            return self._get_state(), 1, True, {}
        elif self._is_full():
            return self._get_state(), 0, True, {}
        
        # Opponent moves
        self._opponent_move()
        
        if self._check_win('O'):
            return self._get_state(), -1, True, {}
        
        return self._get_state(), 0, False, {}
    
    def get_actions(self, state=None):
        return [(i, j) for i in range(3) for j in range(3) 
                if self.board[i][j] == ' ']
```

### Cliff Walking

```python
# envs/cliff_walking.py
from envs.base_env import GameEnv

class CliffWalking(GameEnv):
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = (0, 0)
        self.goal = (0, 11)
        self.cliff_states = [(0, i) for i in range(1, 11)]
        # ... implement reset, step, get_actions
```

## 🧠 Example Agents to Add

### Deep Q-Network (DQN)

```python
# agents/dqn_agent.py
import torch
import torch.nn as nn

class DQNAgent(RLAgent):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        # Initialize neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Implement learning with experience replay
```

### Expected SARSA

```python
# agents/expected_sarsa_agent.py
from agents.base_agent import RLAgent

class ExpectedSARSAAgent(RLAgent):
    def learn(self, state, action, reward, next_state):
        q = self.get_Q(state, action)
        
        # Calculate expected value
        next_qs = [self.get_Q(next_state, a) for a in self.actions]
        expected_q = sum(next_qs) / len(next_qs)
        
        target = reward + self.gamma * expected_q
        self.Q[(state, action)] = q + self.alpha * (target - q)
```

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

## 🎯 Next Steps

1. **Add more environments**: Maze, Tic-Tac-Toe, CartPole, etc.
2. **Implement advanced algorithms**: DQN, Actor-Critic, PPO
3. **Add experience replay**: For more stable learning
4. **Multi-agent support**: Competitive/cooperative scenarios
5. **Save/load models**: Persist trained agents

## 📚 Resources

- **Sutton & Barto**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- **OpenAI Gym**: [gym.openai.com](https://gym.openai.com)
- **Q-Learning Tutorial**: Understanding the Bellman equation

## 🤝 Contributing

Feel free to extend this framework! The modular design makes it easy to:
- Add new environments in `envs/`
- Implement new algorithms in `agents/`
- Create visualization tools in `utils.py`

## 📄 License

Free to use for educational and research purposes.

---

**Happy Learning!** 🚀🧠🎮
