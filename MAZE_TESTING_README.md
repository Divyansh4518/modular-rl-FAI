# Maze Environment Testing Suite

Complete testing framework for evaluating Reinforcement Learning agents on Maze environments.

## 🎯 Overview

This testing suite provides comprehensive tools for training, evaluating, and visualizing the performance of three RL algorithms (Q-Learning, SARSA, and Expected SARSA) on maze navigation tasks.

## 📦 What's Included

### Testing Scripts (4 total)

1. **test_maze.py** - Compare all agents on one maze
2. **quick_test_maze.py** - Preset configurations for quick testing
3. **test_single_agent.py** - Deep dive analysis of individual agents
4. **batch_test.py** - Automated testing across all combinations

### Documentation (2 files)

1. **TESTING_SCRIPTS_README.md** - Quick reference guide
2. **MAZE_TESTING_GUIDE.md** - Comprehensive documentation

## 🚀 Quick Start

### 1. Basic Comparison Test (Recommended First Step)
```bash
python test_maze.py 2 500 200
```
**Time**: ~3-5 minutes  
**Output**: `maze_agents_comparison.png`

### 2. Individual Agent Analysis
```bash
python test_single_agent.py qlearning 2 500
```
**Time**: ~2-3 minutes  
**Output**: 3 detailed visualization files

### 3. Quick Batch Test (Validation)
```bash
python batch_test.py 2
```
**Time**: ~5-10 minutes  
**Output**: Directory with all comparison plots

## 📊 Script Comparison

| Script | Purpose | Time | Output Files | Best For |
|--------|---------|------|--------------|----------|
| **test_maze.py** | Compare 3 agents | 3-5 min | 1 comparison plot | Algorithm comparison |
| **quick_test_maze.py** | Preset configs | 2-10 min | 1 comparison plot | Quick validation |
| **test_single_agent.py** | Single agent deep dive | 2-3 min | 3 visualization files | Understanding one agent |
| **batch_test.py** | Full automation | 5-45 min | 30+ files + report | Comprehensive analysis |

## 🎓 Usage Examples

### Example 1: Compare Agents on Default Maze
```bash
python test_maze.py 2 500 200
```
**What it does**: Trains Q-Learning, SARSA, and Expected SARSA for 500 episodes on the 7x7 default maze.

**Output**:
- Console: Real-time progress, final statistics
- File: `maze_agents_comparison.png` with 4 comparison panels

### Example 2: Analyze Q-Learning in Detail
```bash
python test_single_agent.py qlearning 2 500
```
**What it does**: Trains Q-Learning for 500 episodes and generates detailed visualizations.

**Output**:
- `qlearning_learning_curves.png` - Training progress analysis
- `qlearning_policy.png` - Learned policy with arrows
- `qlearning_value_function.png` - State value heatmap

### Example 3: Quick Validation
```bash
python quick_test_maze.py 1
```
**What it does**: Runs all agents on simple 5x5 maze for quick validation.

### Example 4: Full Benchmark Suite
```bash
python batch_test.py 1
```
**What it does**: Tests all agents on all mazes with comprehensive analysis.

**Output**: Timestamped directory containing:
- 3 comparison plots (one per maze)
- 27 individual analysis files (3 agents × 3 mazes × 3 visualizations)
- Summary report

## 📈 Understanding the Outputs

### Comparison Plot (`maze_agents_comparison.png`)

4 panels showing:
1. **Learning Curve**: Rewards over episodes (higher is better)
2. **Efficiency**: Steps to goal (lower is better)
3. **Success Rate**: % of successful episodes (higher is better)
4. **Final Performance**: Bar chart of last 100 episodes

### Individual Agent Visualizations

#### Learning Curves
- Raw and smoothed reward/step curves
- Success rate over time
- Reward distribution histogram

#### Policy Visualization
- Grid showing optimal action at each state
- Arrows indicate learned policy
- Colors: Green=Start, Red=Goal, Gray=Path, Black=Wall

#### Value Function
- Heatmap of state values
- Warmer colors = higher value
- Shows which states agent considers valuable

## 🎯 Recommended Workflow

### For Beginners:
```bash
# 1. Quick validation (2 min)
python quick_test_maze.py 1

# 2. Standard comparison (5 min)
python test_maze.py 2 500 200

# 3. Pick best agent and analyze (3 min)
python test_single_agent.py qlearning 2 500
```

### For Comprehensive Analysis:
```bash
# 1. Quick batch to verify setup (10 min)
python batch_test.py 2

# 2. Full batch test (45 min)
python batch_test.py 1

# 3. Examine generated report and plots
```

### For Research/Presentation:
```bash
# Test each agent individually for high-quality plots
python test_single_agent.py qlearning 2 1000
python test_single_agent.py sarsa 2 1000
python test_single_agent.py expected_sarsa 2 1000

# Generate comparison
python test_maze.py 2 1000 200
```

## 🔧 Customization

### Modify Training Parameters

Edit the hyperparameters in any script:
```python
alpha = 0.1      # Learning rate (0.01 - 0.5)
gamma = 0.9      # Discount factor (0.8 - 0.99)
epsilon = 0.1    # Exploration rate (0.05 - 0.2)
```

### Create Custom Maze

```python
from envs.maze import Maze

# Define custom layout (0=path, 1=wall)
my_maze = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
]

# Create environment
env = Maze(my_maze, start=(0, 0), goal=(4, 4))
```

### Adjust Plot Appearance

Modify plotting functions to change:
- Colors: Edit `colors` dictionary
- Figure size: Change `figsize=(width, height)`
- Moving average window: Adjust `window_size` parameter
- DPI for higher resolution: Change `dpi=300` in `plt.savefig()`

## 📁 Directory Structure After Testing

```
rl_framework/
├── test_maze.py
├── quick_test_maze.py
├── test_single_agent.py
├── batch_test.py
├── maze_agents_comparison.png          # Latest comparison
├── qlearning_learning_curves.png       # If test_single_agent.py run
├── qlearning_policy.png
├── qlearning_value_function.png
├── test_results_YYYYMMDD_HHMMSS/       # If batch_test.py run
│   ├── comparison_simple.png
│   ├── comparison_default.png
│   ├── comparison_complex.png
│   ├── qlearning_simple_learning_curves.png
│   ├── ... (30+ files)
│   └── summary_report.txt
└── quick_test_results_YYYYMMDD_HHMMSS/ # If quick batch run
    ├── quick_simple.png
    └── quick_default.png
```

## 🎨 Sample Outputs

### What to Expect

**Good Performance**:
- Success rate reaching 95-100%
- Steps decreasing over episodes
- Rewards increasing and stabilizing
- Low variance in final 100 episodes

**Learning Progress**:
- Initial random exploration (high variance)
- Gradual improvement (middle episodes)
- Convergence (final episodes, low variance)

### Typical Results on Default Maze (7x7):

| Agent | Avg Reward | Avg Steps | Success Rate |
|-------|------------|-----------|--------------|
| Q-Learning | 88 | 13 | 100% |
| SARSA | 87 | 13.5 | 100% |
| Expected SARSA | 88 | 13 | 100% |

*Based on 500 episodes, last 100 episodes average*

## 🐛 Troubleshooting

### Common Issues:

**"ModuleNotFoundError: No module named 'matplotlib'"**
```bash
pip install matplotlib numpy
```

**"Training takes too long"**
- Reduce number of episodes
- Use simpler maze (choice 1)
- Reduce max_steps parameter

**"Agents not learning"**
- Increase number of episodes
- Check if maze is solvable
- Adjust learning rate (alpha)

**"Import errors"**
```bash
# Make sure you're in the correct directory
cd rl_framework
python test_maze.py
```

**"Low success rate"**
- Increase max_steps
- Increase number of training episodes
- Simplify maze or check for walls blocking path

## 📚 Documentation

- **TESTING_SCRIPTS_README.md** - Detailed script documentation
- **MAZE_TESTING_GUIDE.md** - Comprehensive usage guide

## 🔬 Technical Details

### Algorithms Implemented:

1. **Q-Learning** (Off-policy TD control)
   - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
   - Learns optimal policy regardless of behavior policy

2. **SARSA** (On-policy TD control)
   - Update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
   - Learns policy being followed (includes exploration)

3. **Expected SARSA** (Hybrid approach)
   - Update: Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',a')] - Q(s,a)]
   - Uses expected value over next actions

### Maze Environments:

| Maze | Size | Walls | Optimal Path | Difficulty |
|------|------|-------|--------------|------------|
| Simple | 5×5 | Few | ~8 steps | Easy |
| Default | 7×7 | Moderate | ~12 steps | Medium |
| Complex | 10×10 | Many | ~18 steps | Hard |

## 💡 Tips for Best Results

1. **Start Simple**: Test on simple maze first
2. **Increase Gradually**: Move to complex mazes after validation
3. **Multiple Runs**: Run tests multiple times for reliability
4. **Save Everything**: All plots are auto-saved as PNG files
5. **Check Console**: Progress updates help identify issues early
6. **Use Batch Tests**: For comprehensive comparison across all conditions

## 🎯 Performance Benchmarks

### Expected Training Times (on typical laptop):

| Test Type | Episodes | Time |
|-----------|----------|------|
| Quick test (Simple) | 200 | 1-2 min |
| Standard (Default) | 500 | 3-5 min |
| Comprehensive (Complex) | 1000 | 8-12 min |
| Full batch test | Various | 30-45 min |

*Times may vary based on hardware*

## 📝 Notes

- All plots use 300 DPI for publication quality
- Results are deterministic for same random seed
- Moving averages use 50-episode windows
- Success is defined as reaching goal within max_steps
- Rewards: +100 for goal, -1 per step, -1 for wall collision

## 🎓 Educational Value

This suite is perfect for:
- Learning RL algorithm differences
- Understanding exploration vs exploitation
- Visualizing convergence behavior
- Comparing on-policy vs off-policy methods
- Creating teaching demonstrations
- Research experiments

## ⚡ Quick Reference

```bash
# Basic comparison
python test_maze.py 2 500 200

# Quick preset test
python quick_test_maze.py 2

# Individual analysis
python test_single_agent.py qlearning 2 500

# Quick validation batch
python batch_test.py 2

# Full benchmark
python batch_test.py 1
```

## 📞 Support

For detailed information on each script, see:
- **TESTING_SCRIPTS_README.md** - Individual script documentation
- **MAZE_TESTING_GUIDE.md** - Complete usage guide

---

**Happy Testing! 🚀**
