# Testing Workflow Quick Reference

## 🎯 Choose Your Testing Path

```
┌─────────────────────────────────────────────────────────────┐
│              START: What do you want to do?                  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
           ┌────────▼────────┐  ┌──────▼──────┐
           │  Quick Check    │  │   Detailed  │
           │  (2-5 min)      │  │   Analysis  │
           └────────┬────────┘  │  (10-45 min)│
                    │           └──────┬──────┘
           ┌────────▼────────┐         │
           │  Compare All    │  ┌──────▼──────────┐
           │    Agents?      │  │  Single Agent   │
           └────────┬────────┘  │   or Batch?     │
                    │           └──────┬──────────┘
         ┌──────────┴───────┐          │
         │                  │   ┌──────┴──────┐
    ┌────▼────┐      ┌─────▼───▼───┐  ┌──────▼──────┐
    │  Preset │      │   Custom    │  │   Single    │
    │  Config │      │  Parameters │  │    Agent    │
    └────┬────┘      └─────┬───────┘  └──────┬──────┘
         │                 │                  │
         │                 │           ┌──────▼──────┐
         │                 │           │   Batch     │
         │                 │           │   Testing   │
         │                 │           └──────┬──────┘
         │                 │                  │
    ┌────▼─────────────────▼──────────────────▼────┐
    │                                               │
    │              RUN TESTING SCRIPT               │
    │                                               │
    └────┬──────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │         OUTPUTS GENERATED                   │
    ├─────────────────────────────────────────────┤
    │  • Console: Progress & Statistics           │
    │  • PNG Files: High-res visualizations       │
    │  • Directory: Batch results (if applicable) │
    └─────────────────────────────────────────────┘
```

---

## 📋 Decision Tree

### 1️⃣ "I want to quickly validate everything works"
```bash
python quick_test_maze.py 1
```
⏱️ 1-2 minutes | 📊 1 comparison plot

---

### 2️⃣ "I want to compare all three algorithms"
```bash
python test_maze.py 2 500 200
```
⏱️ 3-5 minutes | 📊 1 comparison plot (4 panels)

---

### 3️⃣ "I want to understand how Q-Learning learns"
```bash
python test_single_agent.py qlearning 2 500
```
⏱️ 2-3 minutes | 📊 3 visualization files

---

### 4️⃣ "I want to test all agents on all mazes"
```bash
python batch_test.py 2  # Quick
# OR
python batch_test.py 1  # Full
```
⏱️ 5-45 minutes | 📊 Multiple files + report

---

## 🎯 Script Selection Matrix

| Your Goal | Script | Time | Output |
|-----------|--------|------|--------|
| **Quick validation** | quick_test_maze.py 1 | 2 min | 1 plot |
| **Standard comparison** | test_maze.py 2 500 200 | 5 min | 1 plot |
| **Preset configs** | quick_test_maze.py 2 | 5 min | 1 plot |
| **Understand one agent** | test_single_agent.py | 3 min | 3 plots |
| **Complete benchmark** | batch_test.py 1 | 45 min | 30+ files |
| **Quick batch** | batch_test.py 2 | 10 min | 6+ files |
| **Custom parameters** | test_maze.py [params] | varies | 1 plot |
| **All agents detailed** | batch_test.py 1 | 45 min | full suite |

---

## 🔄 Typical Workflow

### For First-Time Users:
```
Step 1: Validate  →  python quick_test_maze.py 1
   ↓ (2 min)
Step 2: Compare   →  python test_maze.py 2 500 200
   ↓ (5 min)
Step 3: Deep Dive →  python test_single_agent.py qlearning 2 500
   ↓ (3 min)
Done! ✅
```

### For Research/Analysis:
```
Step 1: Quick Check    →  python quick_test_maze.py 1
   ↓ (2 min)
Step 2: Each Agent     →  python test_single_agent.py [agent] 2 1000
   ↓ (3 min × 3 agents)
Step 3: Full Batch     →  python batch_test.py 1
   ↓ (45 min)
Step 4: Analyze Results → Check test_results_[timestamp]/
Done! ✅
```

### For Quick Demo:
```
Single Command  →  python test_maze.py 2 500 200
   ↓ (5 min)
Show Plot ✅
```

---

## 📊 Output Files Guide

### From test_maze.py:
```
maze_agents_comparison.png
└── 4 panels:
    ├── Learning Curve (rewards)
    ├── Efficiency (steps)
    ├── Success Rate
    └── Final Performance Bars
```

### From test_single_agent.py:
```
{agent}_learning_curves.png    (4 panels: rewards, steps, success, distribution)
{agent}_policy.png             (Maze with directional arrows)
{agent}_value_function.png     (Heatmap of state values)
```

### From batch_test.py:
```
test_results_[timestamp]/
├── comparison_simple.png       (All agents, simple maze)
├── comparison_default.png      (All agents, default maze)
├── comparison_complex.png      (All agents, complex maze)
├── qlearning_simple_learning_curves.png
├── qlearning_simple_policy.png
├── qlearning_simple_value_function.png
├── ... (27 more files for other agents/mazes)
└── summary_report.txt          (Text summary)
```

---

## ⚡ Quick Commands Reference

```bash
# FASTEST: Quick validation (2 min)
python quick_test_maze.py 1

# STANDARD: Compare algorithms (5 min)
python test_maze.py 2 500 200

# DETAILED: Single agent analysis (3 min)
python test_single_agent.py qlearning 2 500

# COMPREHENSIVE: Full suite (45 min)
python batch_test.py 1
```

---

## 🎨 Visualization Types

### 1. Comparison Plots
- **Learning Curves**: Shows improvement over time
- **Steps to Goal**: Efficiency metric
- **Success Rate**: Reliability metric
- **Performance Bars**: Final comparison

### 2. Learning Analysis
- **Raw Data**: Episode-by-episode performance
- **Moving Average**: Smoothed trends
- **Distribution**: Performance consistency
- **Success Timeline**: Learning progression

### 3. Policy Visualization
- **Arrows**: Show best action at each state
- **Color Coding**: Start (green), Goal (red)
- **Grid Layout**: Clear spatial representation

### 4. Value Function
- **Heatmap**: State value visualization
- **Color Gradient**: Higher value = warmer color
- **Masked Walls**: Only show valid states

---

## 🎯 Parameters Guide

### Maze Choice (1st parameter):
- `1` = Simple (5×5) - Easy, quick testing
- `2` = Default (7×7) - Medium difficulty
- `3` = Complex (10×10) - Challenging, longer paths

### Episodes (2nd parameter):
- `200-300` = Quick test, may not fully converge
- `500-800` = Standard, good convergence
- `1000+` = Comprehensive, full convergence

### Max Steps (3rd parameter):
- `100` = Simple maze
- `200` = Default maze
- `300` = Complex maze

### Agent Type (for test_single_agent.py):
- `qlearning` = Off-policy, aggressive
- `sarsa` = On-policy, conservative
- `expected_sarsa` = Hybrid approach

---

## 💡 Pro Tips

### ✅ Best Practices:
1. Start with quick_test_maze.py mode 1
2. Check console output for errors
3. Examine plots for learning patterns
4. Run multiple times for consistency
5. Save plots with descriptive names

### ⚠️ Common Mistakes:
1. Not being in rl_framework directory
2. Using too few episodes (agents don't converge)
3. max_steps too low (agents can't reach goal)
4. Not checking console for progress
5. Expecting instant results (learning takes time)

### 🔧 Optimization:
1. Use mode 1 for debugging
2. Use mode 2 for standard testing
3. Use mode 3 for final benchmarks
4. Adjust hyperparameters in script
5. Custom mazes for specific tests

---

## 📈 Expected Performance

### Simple Maze (5×5, 200 episodes):
- Success Rate: 95-100%
- Avg Steps: 8-10
- Avg Reward: 90-93

### Default Maze (7×7, 500 episodes):
- Success Rate: 98-100%
- Avg Steps: 12-14
- Avg Reward: 86-89

### Complex Maze (10×10, 1000 episodes):
- Success Rate: 90-98%
- Avg Steps: 18-22
- Avg Reward: 78-83

*Based on last 100 episodes*

---

## 🚀 Getting Started NOW

### Absolute Beginner:
```bash
# Copy and paste this:
cd rl_framework
python quick_test_maze.py 1
```
Wait 2 minutes, check the plot! ✅

### I Want Results Fast:
```bash
python test_maze.py 2 500 200
```
Wait 5 minutes, get comparison plot! ✅

### I Want Everything:
```bash
python batch_test.py 2
```
Wait 10 minutes, get complete analysis! ✅

---

## 📞 Help Guide

| Problem | Solution |
|---------|----------|
| "Command not found" | `cd rl_framework` first |
| "Module not found" | `pip install numpy matplotlib` |
| "Takes too long" | Use mode 1 or reduce episodes |
| "Agents not learning" | Increase episodes |
| "Can't see plots" | Check for PNG files |
| "Poor performance" | Increase episodes or adjust α |

---

## 🎊 Summary

**4 Scripts** → Different use cases
**3 Docs** → Complete guidance
**Multiple Outputs** → Comprehensive analysis

**Start Here**: `python quick_test_maze.py 1` ✅
