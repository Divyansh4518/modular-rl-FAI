import os
import sys
import pickle
import time
import numpy as np

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.tic_tac_toe import TicTacToe
from agents.qlearning_agent import QLearningAgent

MODEL_PATH = os.path.join("saved_models", "tictactoe_qlearning.pkl")

def get_action_from_index(index):
    """Maps 0-8 index to (row, col)."""
    return (index // 3, index % 3)

def get_index_from_action(action):
    """Maps (row, col) to 0-8 index."""
    return action[0] * 3 + action[1]

def print_board(state):
    """
    Prints the board in a clean 3x3 grid.
    State is a tuple of tuples (rows).
    """
    # Flatten the state for easier printing
    flat_board = [cell for row in state for cell in row]
    
    print("\n Current Board:")
    print(f" {flat_board[0]} | {flat_board[1]} | {flat_board[2]} ")
    print("---+---+---")
    print(f" {flat_board[3]} | {flat_board[4]} | {flat_board[5]} ")
    print("---+---+---")
    print(f" {flat_board[6]} | {flat_board[7]} | {flat_board[8]} ")
    print()

def print_guide():
    """Prints the input guide."""
    print("\nInput Guide (0-8):")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    print()

def train_agent(episodes):
    """Trains a new agent on the spot."""
    print(f"\nTraining new agent for {episodes} episodes against Optimal Opponent...")
    env = TicTacToe(opponent='optimal')
    agent = QLearningAgent(env.get_actions(), alpha=0.1, gamma=0.99, epsilon=1.0)
    
    # Training loop
    epsilon_decay = 0.9999
    epsilon_min = 0.05
    
    start_time = time.time()
    for ep in range(episodes):
        state = env.reset()
        done = False
        
        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
    print(f"Training finished in {time.time() - start_time:.2f}s.")
    return agent

def load_model():
    """Loads model or trains a new one."""
    if os.path.exists(MODEL_PATH):
        choice = input(f"Found a pre-trained Pro Agent at '{MODEL_PATH}'. Load it? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                with open(MODEL_PATH, 'rb') as f:
                    agent = pickle.load(f)
                print("Agent loaded successfully.")
                return agent
            except Exception as e:
                print(f"Error loading agent: {e}")
    
    print("\n--- Train a New Agent ---")
    print("1. Easy (500 episodes)")
    print("2. Medium (2000 episodes)")
    print("3. Hard (10000 episodes)")
    
    choice = input("Choose difficulty (1-3) [default=2]: ").strip()
    if choice == '1':
        episodes = 500
    elif choice == '3':
        episodes = 10000
    else:
        episodes = 2000
        
    return train_agent(episodes)

import random

def get_agent_state(env, agent_symbol):
    """
    Returns the state representation for the agent.
    If agent is 'O', swaps 'X' and 'O' so the agent thinks it's 'X'.
    """
    if agent_symbol == 'X':
        return env._get_state()
    
    # Swap symbols for 'O' agent
    swapped_board = [[' ' for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            cell = env.board[r][c]
            if cell == agent_symbol:
                swapped_board[r][c] = 'X'  # Agent sees itself as X
            elif cell != ' ':
                swapped_board[r][c] = 'O'  # Agent sees opponent as O
            else:
                swapped_board[r][c] = ' '
    
    return tuple(tuple(row) for row in swapped_board)

def play_game():
    """Main game loop."""
    agent = load_model()
    
    # Set agent to exploitation mode
    agent.epsilon = 0.0
    
    # We use opponent='none' so we can manually handle the turns
    env = TicTacToe(opponent='none')
    
    while True:
        print("\n" + "="*30)
        print(" NEW GAME")
        print("="*30)
        
        state = env.reset()
        done = False
        print_guide()
        
        # Coin toss
        if random.random() < 0.5:
            current_turn = 'Human'
            human_symbol = 'X'
            agent_symbol = 'O'
            print("Coin toss result: YOU go first! (You are X)")
        else:
            current_turn = 'Agent'
            human_symbol = 'O'
            agent_symbol = 'X'
            print("Coin toss result: AGENT goes first! (Agent is X)")
            
        while not done:
            if current_turn == 'Agent':
                # --- Agent Turn ---
                print(f">> Agent ({agent_symbol}) is thinking...")
                time.sleep(0.5)
                
                # Get state from agent's perspective (always sees itself as X)
                agent_view_state = get_agent_state(env, agent_symbol)
                action = agent.choose_action(agent_view_state)
                row, col = action
                
                # Safety Net: Check validity
                if env.board[row][col] != ' ':
                    print(">> Agent tried invalid move. Correcting...")
                    available = env.get_actions()
                    if available:
                        action = random.choice(available)
                        row, col = action
                    else:
                        # Should not happen if done is False
                        break
                
                # Apply move
                env.board[row][col] = agent_symbol
                state = env._get_state()
                print_board(state)
                
                # Check outcome
                if env._check_win(agent_symbol):
                    print(f"\n*** AGENT ({agent_symbol}) WINS! ***")
                    done = True
                elif env._is_full():
                    print("\n--- DRAW ---")
                    done = True
                else:
                    current_turn = 'Human'

            else:
                # --- Human Turn ---
                valid_move = False
                while not valid_move:
                    try:
                        user_input = input(f"Your turn ({human_symbol}) [0-8]: ").strip()
                        if not user_input.isdigit():
                            print("Please enter a number between 0 and 8.")
                            continue
                            
                        idx = int(user_input)
                        if idx < 0 or idx > 8:
                            print("Number must be between 0 and 8.")
                            continue
                            
                        row, col = get_action_from_index(idx)
                        
                        # Check if empty
                        if env.board[row][col] != ' ':
                            print("That spot is already taken!")
                            continue
                            
                        # Apply move
                        env.board[row][col] = human_symbol
                        state = env._get_state()
                        print_board(state)
                        valid_move = True
                        
                        # Check outcome
                        if env._check_win(human_symbol):
                            print(f"\n*** YOU ({human_symbol}) WIN! ***")
                            done = True
                        elif env._is_full():
                            print("\n--- DRAW ---")
                            done = True
                        else:
                            # Switch turn
                            current_turn = 'Agent'
                            
                    except Exception as e:
                        print(f"Invalid input: {e}")
        
        # End of game
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\nGame quit.")
