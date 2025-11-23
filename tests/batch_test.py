"""
Master testing script to run all environment tests in sequence.
Useful for generating a full report of all environments.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_maze import run_test as run_maze
from test_gridworld import run_test as run_gridworld
from test_tictactoe import run_comparison as run_tictactoe

def main():
    start_time = time.time()
    print("="*80)
    print("STARTING BATCH TEST EXECUTION")
    print("="*80)

    # 1. Maze Tests
    print("\n" + "!"*40)
    print("RUNNING MAZE TESTS")
    print("!"*40 + "\n")
    try:
        # Run standard maze test (Default Maze, 500 episodes)
        run_maze(maze_choice='2', episodes=500, max_steps=200)
    except Exception as e:
        print(f"ERROR in Maze Test: {e}")

    # 2. GridWorld Tests
    print("\n" + "!"*40)
    print("RUNNING GRIDWORLD TESTS")
    print("!"*40 + "\n")
    try:
        # Run standard gridworld test
        run_gridworld(episodes=500)
    except Exception as e:
        print(f"ERROR in GridWorld Test: {e}")

    # 3. Tic-Tac-Toe Tests
    print("\n" + "!"*40)
    print("RUNNING TIC-TAC-TOE TESTS")
    print("!"*40 + "\n")
    try:
        # Run TTT test against Random opponent (faster than optimal for batch)
        # Using fewer episodes for batch speed, but enough for convergence
        run_tictactoe(opponent_type='random', episodes=10000)
    except Exception as e:
        print(f"ERROR in Tic-Tac-Toe Test: {e}")

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print(f"BATCH TEST COMPLETE")
    print(f"Total Duration: {duration:.2f} seconds")
    print("="*80)
    print("Check the 'results' directory for generated plots and images.")

if __name__ == "__main__":
    main()