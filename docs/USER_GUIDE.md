# 📘 Modular RL Framework: The Complete Textbook

## 1. Project Overview
This repository is a modular, educational framework designed to explore and compare fundamental Reinforcement Learning (RL) algorithms. It separates **Agents** (the learning logic) from **Environments** (the world), allowing for "plug-and-play" experimentation.

The goal is to provide a clear, code-first understanding of how tabular RL methods work, how they differ, and how they perform across various challenges—from simple navigation to adversarial games.

---

## 2. Agents & Algorithms
We implement three core tabular RL algorithms. All agents share a common interface defined in `agents/base_agent.py`.

### 🤖 Q-Learning (`agents/qlearning_agent.py`)
- **Type**: Off-Policy TD Control.
- **Concept**: The "optimistic" learner. It updates its Q-values based on the *best possible* future action, regardless of what action it actually takes next.
- **Equation**: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a'} Q(S', a') - Q(S, A)]$
- **Behavior**: Tends to find the shortest path but may risk penalties (e.g., walking right next to a cliff) because it assumes optimal play.

### 🐢 SARSA (`agents/sarsa_agent.py`)
- **Type**: On-Policy TD Control.
- **Concept**: The "conservative" learner. It updates its Q-values based on the action it *actually* takes next (which might be random due to exploration).
- **Equation**: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$
- **Behavior**: Learns a safer path. In the Cliff World, it will walk further away from the edge to avoid falling due to random exploration.

### ⚖️ Expected SARSA (`agents/expected_sarsa_agent.py`)
- **Type**: On-Policy / Off-Policy Hybrid.
- **Concept**: Updates based on the *expected* value of the next state, considering the probability of taking each action.
- **Behavior**: Generally more stable than SARSA and less variance than Q-Learning.

---

## 3. Environments
The environments are located in `envs/` and follow a standard Gym-like interface (`reset()`, `step()`, `render()`).

### 🧩 Maze (`envs/maze.py`)
- **Goal**: Navigate from Start (S) to Goal (G) avoiding Walls (#).
- **State**: (row, col) coordinates.
- **Reward**: -1 per step (encourages shortest path), +100 at Goal.
- **Variants**: Simple (5x5), Default (7x7), Complex (10x10).

### 🧗 GridWorld (`envs/gridworld.py`)
- **Goal**: "Cliff Walking". Get from Start to Goal without falling into the "The Cliff".
- **Challenge**: The optimal path is right next to the cliff. A safe path is far away.
- **Reward**: -1 per step, -100 for falling off the cliff.

### ❌⭕ Tic-Tac-Toe (`envs/tic_tac_toe.py`)
- **Goal**: Win the game (3 in a row).
- **State**: Tuple representation of the 3x3 board.
- **Opponents**:
    - **Random**: Plays randomly.
    - **Optimal**: Uses Minimax to play perfectly (unbeatable).
    - **Self-Play**: Agent vs Agent training.

---

## 4. How to Run: Interactive Demos

### 🎮 Play Tic-Tac-Toe (Human vs AI)
Challenge a pre-trained agent!
```bash
python play_tictactoe.py
```
- The script will check for a saved model in `saved_models/`.
- If none exists, it will offer to train a new one on the spot.
- You play via the console (entering 0-8).

### 🔬 Single Agent Analysis
Deep dive into a specific agent/environment pair.
```bash
python tests/test_single_agent.py
```
- Follow the interactive prompts to select an Environment (Maze/GridWorld/TicTacToe) and an Agent.
- **Output**: It will generate detailed plots in `results/` showing the policy map and learning curves.

---

## 5. The Testing Suite (`tests/`)
We provide specialized scripts to benchmark performance. All plots are saved to `results/`.

### `tests/test_maze.py`
Compares all 3 agents on the Maze environment.
- **Usage**: `python tests/test_maze.py`
- **Metrics**: Steps to Goal, Total Reward, Success Rate.
- **Insight**: Watch how Q-Learning converges faster but SARSA has a smoother learning curve.

### `tests/test_gridworld.py`
Demonstrates the "Cliff Walking" difference.
- **Usage**: `python tests/test_gridworld.py`
- **Insight**: Q-Learning will learn the path hugging the cliff. SARSA will learn the safe path through the middle.

### `tests/test_tictactoe.py`
Trains agents to master the game.
- **Usage**: `python tests/test_tictactoe.py`
- **Modes**:
    1. **Vs Random**: Easy training.
    2. **Vs Optimal**: Hard training (learning to draw).
    3. **Robust Training**: A 3-phase regimen (Random -> Optimal -> Self-Play) to create a "Battle-Hardened" agent. **(Recommended)**

### `tests/batch_test.py`
Runs ALL tests sequentially. Good for verifying the entire codebase.
- **Usage**: `python tests/batch_test.py`

---

## 6. Interpreting the Results

### 📈 Learning Curves
- **X-Axis**: Episodes (Time).
- **Y-Axis**: Total Reward or Steps.
- **Good Sign**: Reward goes UP, Steps go DOWN.
- **Noise**: RL is stochastic. High variance (jagged lines) is normal, especially for Q-Learning.

### 🗺️ Policy Maps (GridWorld/Maze)
- **Arrows**: Indicate the "best action" the agent has learned for that square.
- **Green Square**: Start.
- **Red Square**: Goal.
- **Black Square**: Wall/Obstacle.

### 🔥 Value Heatmaps
- **Color Intensity**: Represents the $V(s)$ (Value of the state).
- **Hot (Red/Yellow)**: High value (close to goal).
- **Cold (Blue)**: Low value (far from goal or near danger).

---

## 7. Directory Structure
```
.
├── agents/          # The brains (Q-Learning, SARSA)
├── envs/            # The worlds (Maze, TicTacToe)
├── tests/           # The experiments
├── results/         # The output plots
├── saved_models/    # Pre-trained agent files (.pkl)
├── notebooks/       # Jupyter notebooks for education
├── docs/            # You are here
├── play_tictactoe.py # Interactive Game
└── README.md        # Landing Page
```
