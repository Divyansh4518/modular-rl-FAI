# 🚀 RL Framework Cheat Sheet

## 🎮 Quick Start

### Play Tic-Tac-Toe (Human vs AI)
```bash
python play_tictactoe.py
```
*Loads a pre-trained agent or trains one on the spot.*

### Interactive Agent Inspector
```bash
python tests/test_single_agent.py
```
*Visualize learning curves and policies for any agent/environment.*

---

## 🧪 Running Experiments (From Root)

### 1. Train & Battle-Harden Tic-Tac-Toe Agent
```bash
python tests/test_tictactoe.py
```
*Select Option 4 for the "Robust Multi-Stage Training" (Random -> Optimal -> Self-Play).*

### 2. Compare Agents on Maze
```bash
python tests/test_maze.py
```
*Compare Q-Learning vs SARSA on Simple (5x5), Default (7x7), or Complex (10x10) mazes.*

### 3. Cliff Walking (GridWorld) Demo
```bash
python tests/test_gridworld.py
```
*See the difference between Q-Learning (risky) and SARSA (safe).*

### 4. Run EVERYTHING
```bash
python tests/batch_test.py
```
*Runs all test suites sequentially.*

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `play_tictactoe.py` | Interactive game script. |
| `tests/test_tictactoe.py` | Training script for Tic-Tac-Toe agents. |
| `tests/test_maze.py` | Benchmarking script for Maze navigation. |
| `agents/qlearning_agent.py` | Implementation of Q-Learning logic. |
| `saved_models/` | Where `.pkl` agent files are stored. |
| `results/` | Where all plots (`.png`) are saved. |

## 🔧 Common Hyperparameters
*Found in `tests/` scripts or `agents/base_agent.py`*

- **Alpha ($\alpha$)**: Learning Rate (Default: `0.1`)
- **Gamma ($\gamma$)**: Discount Factor (Default: `0.9` or `0.99`)
- **Epsilon ($\epsilon$)**: Exploration Rate (Default: `0.1`, decays in training)

---

## 📦 Setup
```bash
pip install numpy matplotlib
```
