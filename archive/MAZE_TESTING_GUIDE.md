# Maze Environment Testing Guide

This guide explains how to use the testing scripts for training and evaluating RL agents on the Maze environment.

## Available Testing Scripts

### 1. `test_maze.py` - Main Testing Script

The comprehensive testing script that trains all learning agents (Q-Learning, SARSA, and Expected SARSA) on the maze environment and generates detailed analysis graphs.

#### Features:
- Trains all three RL agents (Q-Learning, SARSA, Expected SARSA)
- Supports three maze difficulties (Simple 5x5, Default 7x7, Complex 10x10)
- Generates comprehensive comparison plots
- Provides detailed performance metrics
- Saves plots as high-resolution PNG files

#### Usage:

**Basic usage (with defaults):**
```bash
python test_maze.py
```

**With command-line arguments:**
```bash
python test_maze.py [maze_choice] [episodes] [max_steps]
```

**Parameters:**
- `maze_choice`: Maze difficulty (1=Simple, 2=Default, 3=Complex) [default: 2]
- `episodes`: Number of training episodes [default: 1000]
- `max_steps`: Maximum steps per episode [default: 200]

**Examples:**
```bash
# Test on simple maze with 200 episodes
python test_maze.py 1 200 100

# Test on default maze with 500 episodes
python test_maze.py 2 500 200

# Test on complex maze with 1000 episodes
python test_maze.py 3 1000 300
```

### 2. `quick_test_maze.py` - Quick Test Script

A convenience script with preset configurations for quick testing.

#### Usage:

**Interactive mode:**
```bash
python quick_test_maze.py
```

**Command-line mode:**
```bash
python quick_test_maze.py [mode]
```

**Available modes:**
- `1` - Quick Test: Simple Maze (5x5), 200 episodes
- `2` - Standard Test: Default Maze (7x7), 500 episodes [default]
- `3` - Comprehensive Test: Complex Maze (10x10), 1000 episodes

**Examples:**
```bash
# Quick test
python quick_test_maze.py 1

# Standard test
python quick_test_maze.py 2

# Comprehensive test
python quick_test_maze.py 3
```

## Output Files

### Generated Plots

The scripts generate a comprehensive comparison plot saved as `maze_agents_comparison.png` containing:

1. **Learning Curve (Rewards)**: Shows how total rewards evolve over training episodes
2. **Learning Efficiency (Steps)**: Displays the number of steps agents take to reach the goal
3. **Success Rate**: Tracks the percentage of successful episodes over time
4. **Final Performance Metrics**: Bar chart comparing final performance across all agents

### Console Output

The scripts provide real-time progress updates every 100 episodes, showing:
- Average reward (last 100 episodes)
- Average steps to goal (last 100 episodes)
- Success rate percentage

### Final Summary

At the end of training, a detailed summary is printed including:
- Average reward and standard deviation
- Average steps and standard deviation
- Success rate percentage
- Minimum and maximum rewards
- All metrics calculated over the last 100 episodes

## Maze Environments

### Simple Maze (5x5)
- Grid size: 5x5
- Start: (0, 0)
- Goal: (4, 4)
- Difficulty: Easy
- Recommended episodes: 200-300

### Default Maze (7x7)
- Grid size: 7x7
- Start: (0, 0)
- Goal: (6, 6)
- Difficulty: Medium
- Recommended episodes: 500-1000

### Complex Maze (10x10)
- Grid size: 10x10
- Start: (0, 0)
- Goal: (9, 9)
- Difficulty: Hard
- Recommended episodes: 1000-2000

## Agents Tested

### 1. Q-Learning
- **Type**: Off-policy TD control
- **Update Rule**: Uses max Q-value of next state
- **Characteristics**: More aggressive, faster convergence

### 2. SARSA
- **Type**: On-policy TD control
- **Update Rule**: Uses Q-value of actual next action taken
- **Characteristics**: More conservative, considers exploration policy

### 3. Expected SARSA
- **Type**: Hybrid approach
- **Update Rule**: Uses expected Q-value over all next actions
- **Characteristics**: More stable than SARSA, less aggressive than Q-Learning

## Default Hyperparameters

The scripts use the following default hyperparameters:
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9
- **Exploration Rate (ε)**: 0.1

To modify these, edit the `main()` function in `test_maze.py`.

## Interpreting Results

### Good Performance Indicators:
- High success rate (>95%)
- Decreasing steps per episode over time
- Increasing rewards over time
- Low variance in final performance

### What to Look For:
- **Q-Learning**: Usually achieves highest rewards but may be more variable
- **SARSA**: More conservative, stable performance
- **Expected SARSA**: Often balances between Q-Learning and SARSA

## Troubleshooting

### Training takes too long:
- Reduce number of episodes
- Use a simpler maze
- Reduce max_steps parameter

### Agents not reaching goal:
- Increase max_steps
- Increase number of episodes
- Adjust learning rate or exploration rate

### Poor performance:
- Try different hyperparameters
- Ensure maze is solvable
- Increase training episodes

## Requirements

Required Python packages:
- numpy
- matplotlib

Install with:
```bash
pip install numpy matplotlib
```

## Examples of Running Tests

### Quick validation test:
```bash
python test_maze.py 1 100 100
```

### Standard comparison:
```bash
python quick_test_maze.py 2
```

### Full benchmark:
```bash
python test_maze.py 3 2000 300
```

## Advanced Usage

### Custom Maze Testing

To test on a custom maze, modify `test_maze.py`:

```python
# Define custom maze layout
custom_maze = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
]

# Create environment
env = Maze(custom_maze, start=(0, 0), goal=(4, 4))
```

### Modifying Hyperparameters

Edit the hyperparameters in the `main()` function:

```python
alpha = 0.15     # Faster learning
gamma = 0.95     # More forward-looking
epsilon = 0.05   # Less exploration
```

## Performance Tips

1. **For faster training**: Use smaller mazes and fewer episodes
2. **For better plots**: Increase number of episodes
3. **For smoother curves**: The moving average window is set to 50 episodes
4. **For detailed analysis**: Check the saved PNG file for high-resolution plots

## Citation

If you use these testing scripts in your research or project, please cite:
```
RL Framework - Maze Environment Testing Suite
Foundations of AI Project
```
