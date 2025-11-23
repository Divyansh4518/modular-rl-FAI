"""
Comprehensive testing script for Tic-Tac-Toe environment.

This script trains Q-Learning and SARSA agents on Tic-Tac-Toe against
Random and Optimal opponents, generating detailed analysis graphs.
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tic_tac_toe import TicTacToe, TicTacToeSelfPlay
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent

# --- Hyperparameters ---
TOTAL_EPISODES = 20000  # Reduced slightly for faster testing, can be increased
ALPHA = 0.1
GAMMA = 0.99 
EPSILON_DECAY = 0.99995 
EPSILON_MIN = 0.05

def get_agent_state(env, player):
    """
    Returns the state representation for the agent.
    If player is 'O', swaps 'X' and 'O' so the agent thinks it's 'X'.
    """
    if player == 'X':
        return env._get_state()
    
    # Swap symbols for 'O' agent
    swapped_board = [[' ' for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            cell = env.board[r][c]
            if cell == player:
                swapped_board[r][c] = 'X'  # Agent sees itself as X
            elif cell != ' ':
                swapped_board[r][c] = 'O'  # Agent sees opponent as O
            else:
                swapped_board[r][c] = ' '
    
    return tuple(tuple(row) for row in swapped_board)

def train_agent_tictactoe(env, agent, episodes=TOTAL_EPISODES, verbose=True, is_self_play=False):
    """Train a tabular RL agent on Tic-Tac-Toe."""
    results_list = []
    initial_epsilon = agent.epsilon 
    
    for ep in range(episodes):
        state = env.reset()
        
        # For self-play, we need to handle state perspective
        current_player = 'X'
        if is_self_play:
            state = get_agent_state(env, current_player)
            
        action = agent.choose_action(state)
        done = False

        # Decay epsilon
        agent.epsilon = max(EPSILON_MIN, initial_epsilon * (EPSILON_DECAY ** ep))

        while not done:
            next_state_raw, reward, done, info = env.step(action)
            
            if is_self_play:
                # In self-play, env.step switches the player internally
                # So next_state_raw is the board state for the NEXT player
                # We need to process the reward and update for the CURRENT player
                
                # Note: In this simple self-play implementation, we might need to be careful
                # about whose turn it is.
                # TicTacToeSelfPlay switches current_player AFTER the move.
                # So if X moved, now it is O's turn.
                
                next_player = 'O' if current_player == 'X' else 'X'
                next_state = get_agent_state(env, next_player)
                
                # If game is done, the reward belongs to the player who just moved (current_player)
                # If not done, reward is 0.
                
                # Choose action for the next player (if not done)
                next_action = None
                if not done:
                    next_action = agent.choose_action(next_state)
                
                # Update Q-value for the player who just moved
                if isinstance(agent, SARSAAgent):
                    agent.learn(state, action, reward, next_state, next_action)
                else:
                    agent.learn(state, action, reward, next_state)
                
                if not done:
                    action = next_action
                    state = next_state
                    current_player = next_player
            else:
                # Standard training against environment opponent
                if isinstance(agent, SARSAAgent):
                    next_action = agent.choose_action(next_state_raw)
                    agent.learn(state, action, reward, next_state_raw, next_action)
                    action = next_action
                else:
                    agent.learn(state, action, reward, next_state_raw)
                    action = agent.choose_action(next_state_raw)

                state = next_state_raw
        
        # Record outcome (always from X's perspective for consistency in plots)
        if info.get('winner') == 'X':
            results_list.append(1)
        elif info.get('winner') == 'O':
            results_list.append(-1)
        elif info.get('winner') == 'draw':
            results_list.append(0)
        else:
             results_list.append(-2)
        
        if verbose and (ep + 1) % 1000 == 0:
            win_rate = (results_list[-1000:].count(1) / 1000) * 100
            loss_rate = (results_list[-1000:].count(-1) / 1000) * 100
            draw_rate = (results_list[-1000:].count(0) / 1000) * 100
            print(f"Episode {ep + 1}/{episodes} | Win: {win_rate:.1f}%, Loss: {loss_rate:.1f}%, Draw: {draw_rate:.1f}%")
    
    return results_list

def calculate_metric(results, metric='win', window_size=1000):
    """Calculate rolling metric."""
    target = 1 if metric == 'win' else (-1 if metric == 'loss' else 0)
    rates = []
    for i in range(len(results)):
        start_idx = max(0, i - window_size + 1)
        window = results[start_idx:i+1]
        rate = (window.count(target) / len(window)) * 100
        rates.append(rate)
    return rates

def run_comparison(opponent_type='random', episodes=TOTAL_EPISODES):
    """Run comparison between agents."""
    print(f"\n{'='*60}")
    print(f"TIC-TAC-TOE: {opponent_type.upper()} OPPONENT")
    print(f"{'='*60}")
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    env = TicTacToe(opponent=opponent_type)
    
    # Initialize agents
    qlearn = QLearningAgent(env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    sarsa = SARSAAgent(env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    
    print("\nTraining Q-Learning Agent...")
    q_results = train_agent_tictactoe(env, qlearn, episodes)
    
    print("\nTraining SARSA Agent...")
    s_results = train_agent_tictactoe(env, sarsa, episodes)
    
    # Plotting
    metrics = ['win', 'loss', 'draw']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Agents vs {opponent_type.title()} Opponent', fontsize=16)
    
    for i, metric in enumerate(metrics):
        q_rate = calculate_metric(q_results, metric)
        s_rate = calculate_metric(s_results, metric)
        
        axes[i].plot(q_rate, label='Q-Learning', alpha=0.8)
        axes[i].plot(s_rate, label='SARSA', alpha=0.8)
        axes[i].set_title(f'{metric.title()} Rate')
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Rate (%)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    filename = f'tictactoe_vs_{opponent_type}.png'
    save_path = os.path.join('results', filename)
    plt.savefig(save_path)
    print(f"\nPlot saved as '{save_path}'")
    
    return qlearn

def run_robust_comparison(episodes_per_phase=5000):
    """
    Run a multi-stage robust training regimen for both agents.
    Phase 1: Vs Random
    Phase 2: Vs Optimal
    Phase 3: Self-Play
    """
    print(f"\n{'='*60}")
    print(f"ROBUST MULTI-STAGE TRAINING")
    print(f"{'='*60}")
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Initialize agents
    # We use a dummy env just to get actions
    dummy_env = TicTacToe()
    qlearn = QLearningAgent(dummy_env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    sarsa = SARSAAgent(dummy_env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    
    combined_q_results = []
    combined_s_results = []
    
    # --- Phase 1: Vs Random ---
    print("\n--- Phase 1: The Wild West (Vs Random) ---")
    env_random = TicTacToe(opponent='random')
    
    print("Training Q-Learning (Phase 1)...")
    qlearn.epsilon = 1.0 # Reset epsilon for Phase 1
    combined_q_results.extend(train_agent_tictactoe(env_random, qlearn, episodes_per_phase))
    
    print("Training SARSA (Phase 1)...")
    sarsa.epsilon = 1.0
    combined_s_results.extend(train_agent_tictactoe(env_random, sarsa, episodes_per_phase))
    
    # --- Phase 2: Vs Optimal ---
    print("\n--- Phase 2: The Master Class (Vs Optimal) ---")
    env_optimal = TicTacToe(opponent='optimal')
    
    print("Training Q-Learning (Phase 2)...")
    # Keep epsilon from end of Phase 1, or reset slightly? 
    # Let's boost it slightly to ensure adaptation, but not full reset
    qlearn.epsilon = max(0.3, qlearn.epsilon) 
    combined_q_results.extend(train_agent_tictactoe(env_optimal, qlearn, episodes_per_phase))
    
    print("Training SARSA (Phase 2)...")
    sarsa.epsilon = max(0.3, sarsa.epsilon)
    combined_s_results.extend(train_agent_tictactoe(env_optimal, sarsa, episodes_per_phase))
    
    # --- Phase 3: Self-Play ---
    print("\n--- Phase 3: Self-Play (Refinement) ---")
    env_self = TicTacToeSelfPlay()
    
    print("Training Q-Learning (Phase 3)...")
    qlearn.epsilon = max(0.2, qlearn.epsilon)
    combined_q_results.extend(train_agent_tictactoe(env_self, qlearn, episodes_per_phase, is_self_play=True))
    
    print("Training SARSA (Phase 3)...")
    sarsa.epsilon = max(0.2, sarsa.epsilon)
    combined_s_results.extend(train_agent_tictactoe(env_self, sarsa, episodes_per_phase, is_self_play=True))
    
    # Plotting
    metrics = ['win', 'loss', 'draw']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Robust Training Progress (3 Phases)', fontsize=16)
    
    for i, metric in enumerate(metrics):
        q_rate = calculate_metric(combined_q_results, metric)
        s_rate = calculate_metric(combined_s_results, metric)
        
        axes[i].plot(q_rate, label='Q-Learning', alpha=0.8)
        axes[i].plot(s_rate, label='SARSA', alpha=0.8)
        
        # Add phase separators
        axes[i].axvline(x=episodes_per_phase, color='k', linestyle='--', alpha=0.3, label='Phase Change')
        axes[i].axvline(x=episodes_per_phase*2, color='k', linestyle='--', alpha=0.3)
        
        axes[i].set_title(f'{metric.title()} Rate')
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Rate (%)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    filename = 'tictactoe_robust_training.png'
    save_path = os.path.join('results', filename)
    plt.savefig(save_path)
    print(f"\nPlot saved as '{save_path}'")
    
    return qlearn

if __name__ == "__main__":
    print("\nSelect Training Mode:")
    print("1. Standard Comparison (Vs Random)")
    print("2. Standard Comparison (Vs Optimal)")
    print("3. Run Both Standard Comparisons")
    print("4. Train Robust Agent (Multi-Stage: Random -> Optimal -> Self-Play)")
    
    choice = input("\nEnter choice (1-4) [default=4]: ").strip() or '4'
    
    trained_agent = None
    
    if choice == '1':
        run_comparison('random')
    elif choice == '2':
        run_comparison('optimal')
    elif choice == '3':
        run_comparison('random')
        run_comparison('optimal')
    else:
        trained_agent = run_robust_comparison(episodes_per_phase=5000)

    # Save the trained agent (only if we ran the robust training)
    if trained_agent:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(root_dir, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'tictactoe_qlearning.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(trained_agent, f)
        print(f"\nBattle-Hardened Agent saved to {model_path}")
