# Maze Environment Testing Scripts - Summary

This directory contains comprehensive testing scripts for evaluating reinforcement learning agents on the Maze environment.

## 📁 Available Scripts

### 1. **test_maze.py** - Main Comparison Script
**Purpose**: Train and compare all three RL agents (Q-Learning, SARSA, Expected SARSA) on the same maze.

**Features**:
- ✅ Trains all agents simultaneously
- ✅ Generates 4-panel comparison plots
- ✅ Provides detailed performance metrics
- ✅ Supports 3 maze difficulties
- ✅ Customizable training parameters

**Usage**:
```bash
# Default run (Default maze, 1000 episodes)
python test_maze.py

# Custom parameters
python test_maze.py [maze_choice] [episodes] [max_steps]

# Examples:
python test_maze.py 1 200 100    # Simple maze, 200 episodes
python test_maze.py 2 500 200    # Default maze, 500 episodes
python test_maze.py 3 1000 300   # Complex maze, 1000 episodes
```

**Output**:
- Console: Progress updates, final summary statistics
- File: `maze_agents_comparison.png` (4-panel comparison plot)

---

### 2. **quick_test_maze.py** - Preset Configurations
**Purpose**: Quick testing with optimized preset configurations.

**Features**:
- ✅ Three preset testing modes
- ✅ Optimized parameters for each difficulty
- ✅ No manual parameter tuning needed
- ✅ Interactive or command-line usage

**Usage**:
```bash
# Interactive mode
python quick_test_maze.py

# Command-line mode
python quick_test_maze.py [mode]

# Modes:
python quick_test_maze.py 1    # Quick: Simple maze, 200 episodes
python quick_test_maze.py 2    # Standard: Default maze, 500 episodes
python quick_test_maze.py 3    # Comprehensive: Complex maze, 1000 episodes
```

**Output**: Same as test_maze.py

---

### 3. **test_single_agent.py** - Individual Agent Analysis
**Purpose**: Deep dive into a single agent's learning process with detailed visualizations.

**Features**:
- ✅ Focus on one agent at a time
- ✅ Detailed learning curves
- ✅ Policy visualization (arrows showing best actions)
- ✅ Value function heatmap
- ✅ Reward distribution histogram

**Usage**:
```bash
# Default (Q-Learning on Default maze)
python test_single_agent.py

# Custom parameters
python test_single_agent.py [agent] [maze] [episodes]

# Examples:
python test_single_agent.py qlearning 2 500       # Q-Learning
python test_single_agent.py sarsa 1 300           # SARSA
python test_single_agent.py expected_sarsa 3 800  # Expected SARSA
```

**Output**:
- Files:
  - `{agent}_learning_curves.png` - 4-panel learning analysis
  - `{agent}_policy.png` - Policy visualization with arrows
  - `{agent}_value_function.png` - Value function heatmap

---

## 📊 Output Visualizations

### test_maze.py & quick_test_maze.py Output:
**File**: `maze_agents_comparison.png`

**Contains**:
1. **Learning Curve (Rewards)**: Total rewards over episodes
2. **Learning Efficiency (Steps)**: Steps to reach goal over episodes
3. **Success Rate**: Percentage of successful episodes over time
4. **Final Performance**: Bar chart comparing agents on 3 metrics

### test_single_agent.py Output:
**Files**: `{agent}_learning_curves.png`, `{agent}_policy.png`, `{agent}_value_function.png`

**Learning Curves Contains**:
1. **Rewards over Time**: Raw + moving average
2. **Steps per Episode**: Raw + moving average
3. **Success Rate**: Rolling window success rate
4. **Reward Distribution**: Histogram of last 100 episodes

**Policy Visualization**:
- Maze grid with arrows showing optimal action at each state
- Color-coded: Green=Start, Red=Goal, Gray=Path, Black=Wall

**Value Function**:
- Heatmap showing learned state values (max Q-value)
- Warmer colors = higher value states

---

## 🎯 Quick Start Guide

### For Quick Testing:
```bash
python quick_test_maze.py 2
```
Runs standard test: all 3 agents, default maze, 500 episodes (~2-3 minutes)

### For Detailed Analysis of One Agent:
```bash
python test_single_agent.py qlearning 2 500
```
Analyzes Q-Learning with detailed visualizations

### For Full Comparison:
```bash
python test_maze.py 2 1000 200
```
Comprehensive comparison of all agents

---

## 🏃 Recommended Testing Workflow

1. **Quick Validation** (2 min):
   ```bash
   python quick_test_maze.py 1
   ```
   Verifies everything works on simple maze

2. **Standard Comparison** (5 min):
   ```bash
   python quick_test_maze.py 2
   ```
   Compare agents on medium difficulty

3. **Deep Dive** (3 min each):
   ```bash
   python test_single_agent.py qlearning 2 500
   python test_single_agent.py sarsa 2 500
   python test_single_agent.py expected_sarsa 2 500
   ```
   Analyze each agent individually

4. **Final Benchmark** (15 min):
   ```bash
   python test_maze.py 3 2000 300
   ```
   Comprehensive test on complex maze

---

## 🔧 Customization

### Modify Hyperparameters
Edit in `test_maze.py` or `test_single_agent.py`:
```python
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
```

### Create Custom Maze
```python
from envs.maze import Maze

custom_maze = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]
env = Maze(custom_maze, start=(0, 0), goal=(2, 4))
```

---

## 📈 Understanding Results

### Good Performance Indicators:
- ✅ Success rate > 95%
- ✅ Steps decreasing over time
- ✅ Rewards increasing/stabilizing
- ✅ Low variance in final 100 episodes

### Agent Characteristics:
- **Q-Learning**: Highest rewards, most aggressive
- **SARSA**: Most conservative, stable
- **Expected SARSA**: Balanced, often best overall

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Training too slow | Reduce episodes or use simpler maze |
| Low success rate | Increase episodes or max_steps |
| Poor convergence | Adjust learning rate (alpha) |
| High variance | Increase training episodes |
| Import errors | Ensure you're in the rl_framework directory |

---

## 📚 Documentation

For detailed information, see: **MAZE_TESTING_GUIDE.md**

---

## 🎓 Educational Use

These scripts are ideal for:
- Understanding RL algorithm differences
- Visualizing learning processes
- Comparing on-policy vs off-policy methods
- Studying convergence behavior
- Creating teaching demonstrations

---

## 💡 Tips

1. **Start Simple**: Begin with simple maze and few episodes
2. **Compare Gradually**: Run quick tests before comprehensive ones
3. **Save Plots**: All plots are automatically saved as PNG files
4. **Check Console**: Real-time progress helps identify issues early
5. **Use Single Agent**: For debugging, test one agent at a time

---

## 📦 Requirements

```bash
pip install numpy matplotlib
```

---

## 🎯 Example Session

```bash
# Terminal session example
cd rl_framework

# Quick validation
python quick_test_maze.py 1
# Output: maze_agents_comparison.png

# Detailed Q-Learning analysis
python test_single_agent.py qlearning 2 500
# Output: qlearning_learning_curves.png, qlearning_policy.png, qlearning_value_function.png

# Full comparison
python test_maze.py 2 1000 200
# Output: maze_agents_comparison.png (overwritten with more episodes)
```

---

## 📧 Questions?

Refer to **MAZE_TESTING_GUIDE.md** for comprehensive documentation.
