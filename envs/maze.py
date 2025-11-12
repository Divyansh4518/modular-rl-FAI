import numpy as np
from envs.base_env import GameEnv

class Maze(GameEnv):
    """
    A configurable maze environment where the agent must navigate from start to goal.
    
    The maze is represented as a 2D grid where:
    - 0 = open path
    - 1 = wall
    - Agent can move in 4 directions (Up, Down, Left, Right)
    """
    
    def __init__(self, maze_layout=None, start=None, goal=None):
        """
        Initialize the maze environment.
        
        Args:
            maze_layout: 2D list/array where 0=path, 1=wall. If None, uses default maze.
            start: Tuple (row, col) for start position. If None, uses (0, 0).
            goal: Tuple (row, col) for goal position. If None, uses bottom-right corner.
        """
        if maze_layout is None:
            # Default 7x7 maze
            self.maze = np.array([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        else:
            self.maze = np.array(maze_layout)
        
        self.height, self.width = self.maze.shape
        
        # Set start and goal positions
        self.start = start if start is not None else (0, 0)
        self.goal = goal if goal is not None else (self.height - 1, self.width - 1)
        
        # Validate start and goal are not walls
        if self.maze[self.start] == 1:
            raise ValueError("Start position cannot be a wall!")
        if self.maze[self.goal] == 1:
            raise ValueError("Goal position cannot be a wall!")
        
        self.state = self.start
        self.actions = ['U', 'D', 'L', 'R']
        self.action_effects = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def reset(self):
        """Reset the environment to the start position."""
        self.state = self.start
        return self.state
    
    def get_actions(self, state=None):
        """Return available actions."""
        return self.actions
    
    def step(self, action):
        """
        Execute an action and return the result.
        
        Args:
            action: One of ['U', 'D', 'L', 'R']
            
        Returns:
            next_state: New position (row, col)
            reward: Reward for the transition
            done: Whether the episode has ended
            info: Additional information dictionary
        """
        row, col = self.state
        d_row, d_col = self.action_effects[action]
        
        # Calculate next position
        next_row = row + d_row
        next_col = col + d_col
        
        # Check if next position is valid (within bounds and not a wall)
        if (0 <= next_row < self.height and 
            0 <= next_col < self.width and 
            self.maze[next_row, next_col] == 0):
            # Valid move
            next_state = (next_row, next_col)
        else:
            # Invalid move (hit wall or boundary) - stay in place
            next_state = self.state
        
        # Calculate reward
        if next_state == self.goal:
            reward = 100  # Large positive reward for reaching goal
            done = True
        elif next_state == self.state and (next_row, next_col) != self.state:
            # Tried to move into a wall
            reward = -1  # Small penalty for hitting wall
            done = False
        else:
            # Normal move
            reward = -1  # Small penalty to encourage shorter paths
            done = False
        
        self.state = next_state
        
        info = {
            'wall_hit': next_state == self.state and (next_row, next_col) != self.state
        }
        
        return next_state, reward, done, info
    
    def render(self):
        """Print a visual representation of the maze with the agent's position."""
        print("\n" + "=" * (self.width * 2 + 1))
        for row in range(self.height):
            row_str = "|"
            for col in range(self.width):
                if (row, col) == self.state:
                    row_str += "A "  # Agent
                elif (row, col) == self.goal:
                    row_str += "G "  # Goal
                elif (row, col) == self.start:
                    row_str += "S "  # Start
                elif self.maze[row, col] == 1:
                    row_str += "█ "  # Wall
                else:
                    row_str += "· "  # Path
            row_str += "|"
            print(row_str)
        print("=" * (self.width * 2 + 1))
    
    def get_valid_actions(self, state=None):
        """
        Return only the actions that won't hit a wall from the given state.
        
        Args:
            state: Position to check from. If None, uses current state.
            
        Returns:
            List of valid actions
        """
        if state is None:
            state = self.state
        
        row, col = state
        valid = []
        
        for action in self.actions:
            d_row, d_col = self.action_effects[action]
            next_row, next_col = row + d_row, col + d_col
            
            if (0 <= next_row < self.height and 
                0 <= next_col < self.width and 
                self.maze[next_row, next_col] == 0):
                valid.append(action)
        
        return valid if valid else self.actions  # Return all if no valid moves


def create_simple_maze():
    """Create a simple 5x5 maze for quick testing."""
    maze = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    return Maze(maze, start=(0, 0), goal=(4, 4))


def create_complex_maze():
    """Create a more challenging 10x10 maze."""
    maze = [
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]
    return Maze(maze, start=(0, 0), goal=(9, 9))
