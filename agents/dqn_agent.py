import random
import numpy as np
from collections import deque, namedtuple

# Try to import PyTorch, provide helpful error message if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. DQN agent will not be available.")
    print("Install PyTorch with: pip install torch")


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    
    Simple feedforward network with configurable hidden layers.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network (DQN) agent with experience replay and target network.
    
    Key features:
    - Neural network function approximation
    - Experience replay for breaking correlation
    - Target network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(self, state_dim, action_list, hidden_dims=[64, 64], 
                 alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64, 
                 target_update_freq=10):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state vector
            action_list: List of possible actions
            hidden_dims: List of hidden layer sizes
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network (in episodes)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent. Install with: pip install torch")
        
        self.state_dim = state_dim
        self.actions = action_list
        self.action_dim = len(action_list)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.episode_count = 0
        
        # Create Q-network and target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def state_to_tensor(self, state):
        """
        Convert state to tensor.
        
        Args:
            state: State (can be tuple, list, or array)
            
        Returns:
            PyTorch tensor
        """
        if isinstance(state, (tuple, list)):
            # Flatten nested structures
            state = self._flatten_state(state)
        
        state_array = np.array(state, dtype=np.float32)
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
    
    def _flatten_state(self, state):
        """Flatten a nested state structure into a 1D list."""
        flat = []
        for item in state:
            if isinstance(item, (tuple, list)):
                flat.extend(self._flatten_state(item))
            else:
                # Convert characters to numbers for board games
                if isinstance(item, str):
                    if item == 'X':
                        flat.append(1.0)
                    elif item == 'O':
                        flat.append(-1.0)
                    else:
                        flat.append(0.0)
                else:
                    flat.append(float(item))
        return flat
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Chosen action
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            return self.actions[action_idx]
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        action_idx = self.actions.index(action)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def learn(self):
        """
        Sample from replay buffer and update Q-network.
        
        Returns:
            Loss value if learning occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.cat([self.state_to_tensor(e.state) for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.cat([self.state_to_tensor(e.next_state) for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, filepath)
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']


def train_dqn(env, agent, episodes=1000, max_steps=1000, verbose=True):
    """
    Training loop for DQN agent.
    
    Args:
        env: Environment instance
        agent: DQNAgent instance
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        
    Returns:
        List of total rewards per episode
    """
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.learn()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay epsilon and update target network
        agent.decay_epsilon()
        
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes} - "
                  f"Avg Reward: {avg_reward:.2f} - "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_per_episode
