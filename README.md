# 🧠 Modular RL Framework

A clean, educational, and modular Reinforcement Learning framework designed to explore tabular RL algorithms (Q-Learning, SARSA) across various environments (Maze, GridWorld, Tic-Tac-Toe).

## 🚀 Quick Start

**Play Tic-Tac-Toe against the AI:**
```bash
python play_tictactoe.py
```
*This will load a pre-trained agent (or train one on the spot) and let you play via the console.*

**Run the Interactive Test Suite:**
```bash
python tests/test_single_agent.py
```
*Select an environment and agent to visualize learning curves and policies.*

## 📂 Repository Structure

*   **`agents/`**: The "Brains". Contains implementations of Q-Learning, SARSA, and Expected SARSA.
*   **`envs/`**: The "Worlds". Contains Maze, GridWorld (Cliff Walking), and Tic-Tac-Toe.
*   **`tests/`**: The "Experiments". Scripts to train agents and generate performance plots.
*   **`results/`**: The "Output". All plots and graphs are saved here.
*   **`saved_models/`**: Stores trained agent files (`.pkl`).
*   **`docs/`**: Detailed documentation.

## 📚 Documentation

For a deep dive into the algorithms, testing methodologies, and how to interpret the results, please read the **[Comprehensive User Guide](docs/USER_GUIDE.md)**.

## 📦 Requirements

*   Python 3.x
*   `numpy`
*   `matplotlib`

Install dependencies:
```bash
pip install numpy matplotlib
```
