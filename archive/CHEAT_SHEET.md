# 🚀 Testing Suite Cheat Sheet

## ONE-LINE COMMANDS

```bash
# Quick validation (2 min)
python quick_test_maze.py 1

# Standard test (5 min)
python test_maze.py 2 500 200

# Analyze Q-Learning (3 min)
python test_single_agent.py qlearning 2 500

# Quick batch (10 min)
python batch_test.py 2
```

---

## SCRIPT SYNTAX

```bash
# test_maze.py
python test_maze.py [maze] [episodes] [max_steps]
python test_maze.py 2 500 200  # Example

# quick_test_maze.py
python quick_test_maze.py [mode]
python quick_test_maze.py 2  # Example

# test_single_agent.py
python test_single_agent.py [agent] [maze] [episodes]
python test_single_agent.py qlearning 2 500  # Example

# batch_test.py
python batch_test.py [1|2]
python batch_test.py 2  # Example
```

---

## PARAMETERS

### Maze (1st param):
- `1` = Simple 5×5
- `2` = Default 7×7 ⭐
- `3` = Complex 10×10

### Episodes (2nd param):
- `200-300` = Quick
- `500-800` = Standard ⭐
- `1000+` = Comprehensive

### Agent (for single):
- `qlearning` ⭐
- `sarsa`
- `expected_sarsa`

---

## QUICK EXAMPLES

```bash
# Compare all agents, default maze
python test_maze.py 2 500 200

# Compare all agents, simple maze, quick
python test_maze.py 1 200 100

# Compare all agents, complex maze, thorough
python test_maze.py 3 1000 300

# Analyze Q-Learning on default maze
python test_single_agent.py qlearning 2 500

# Analyze SARSA on simple maze
python test_single_agent.py sarsa 1 300

# Quick batch test
python batch_test.py 2

# Full batch test
python batch_test.py 1
```

---

## OUTPUT FILES

```bash
# test_maze.py → 1 file
maze_agents_comparison.png

# test_single_agent.py → 3 files
{agent}_learning_curves.png
{agent}_policy.png
{agent}_value_function.png

# batch_test.py → Directory with 30+ files
test_results_[timestamp]/
```

---

## TIME ESTIMATES

| Command | Time |
|---------|------|
| quick_test_maze.py 1 | 2 min |
| test_maze.py 2 500 200 | 5 min |
| test_single_agent.py | 3 min |
| batch_test.py 2 | 10 min |
| batch_test.py 1 | 45 min |

---

## RECOMMENDED WORKFLOW

```bash
# Step 1: Validate (2 min)
python quick_test_maze.py 1

# Step 2: Compare (5 min)
python test_maze.py 2 500 200

# Step 3: Deep dive (3 min)
python test_single_agent.py qlearning 2 500
```

---

## TROUBLESHOOTING

```bash
# If import error:
pip install numpy matplotlib

# If command not found:
cd rl_framework

# If too slow:
python test_maze.py 1 200 100  # Use simple maze

# If poor results:
python test_maze.py 2 1000 200  # More episodes
```

---

## CUSTOMIZATION

### In script, modify:
```python
# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount
epsilon = 0.1    # Exploration

# Colors
colors = {'Q-Learning': '#1f77b4'}

# Plot size
figsize=(15, 12)
```

---

## EXPECTED RESULTS (Default Maze)

```
Success Rate: 95-100%
Avg Steps: 12-14
Avg Reward: 86-89
```

---

## DOCUMENTATION FILES

1. **MAZE_TESTING_README.md** - Overview
2. **TESTING_SCRIPTS_README.md** - Details
3. **MAZE_TESTING_GUIDE.md** - Comprehensive
4. **WORKFLOW_GUIDE.md** - Workflow
5. **CREATION_SUMMARY.md** - Summary
6. **This file** - Cheat sheet

---

## KEYBOARD SHORTCUTS

```bash
# Windows PowerShell
↑ - Previous command
Ctrl+C - Stop script
Tab - Auto-complete

# Copy-paste friendly
Right-click to paste in PowerShell
```

---

## MOST USEFUL COMMANDS

```bash
# 1. For quick demo (5 min):
python test_maze.py 2 500 200

# 2. For understanding agent (3 min):
python test_single_agent.py qlearning 2 500

# 3. For validation (2 min):
python quick_test_maze.py 1

# 4. For everything (10 min):
python batch_test.py 2
```

---

## FILE LOCATIONS

```
rl_framework/
├── test_maze.py              ← Main script
├── quick_test_maze.py        ← Preset script
├── test_single_agent.py      ← Single agent script
├── batch_test.py             ← Batch script
└── maze_agents_comparison.png ← Generated output
```

---

## AGENT CHARACTERISTICS

| Agent | Type | Behavior |
|-------|------|----------|
| Q-Learning | Off-policy | Aggressive |
| SARSA | On-policy | Conservative |
| Expected SARSA | Hybrid | Balanced |

---

## COPY-PASTE READY

### Beginner:
```bash
cd rl_framework
python quick_test_maze.py 1
```

### Standard:
```bash
cd rl_framework
python test_maze.py 2 500 200
```

### Complete:
```bash
cd rl_framework
python test_single_agent.py qlearning 2 500
python test_single_agent.py sarsa 2 500
python test_single_agent.py expected_sarsa 2 500
python test_maze.py 2 1000 200
```

---

## SUCCESS INDICATORS

✅ Console shows progress
✅ Success rate > 95%
✅ Steps decreasing
✅ Rewards increasing
✅ PNG files generated

---

## QUICK DEBUG

```bash
# Test if working:
python quick_test_maze.py 1

# If error:
pip install numpy matplotlib

# If still error:
python --version  # Check Python 3.x
```

---

## REMEMBER

- **Start in rl_framework directory**
- **Check console for progress**
- **Plots auto-save as PNG**
- **Higher episodes = better convergence**
- **Simple maze = faster testing**

---

## MOST COMMON USE

```bash
python test_maze.py 2 500 200
```

This single command:
- Trains all 3 agents
- Tests on default maze
- Generates comparison plot
- Takes ~5 minutes
- Saves as PNG

**Perfect for demos and comparisons!** ⭐
