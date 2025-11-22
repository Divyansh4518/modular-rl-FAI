import numpy as np
import matplotlib.pyplot as plt
from envs.tic_tac_toe import TicTacToe
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
# Assuming you have a plot_rewards function in utils.py
from utils import plot_rewards, moving_average 

# --- Hyperparameters for Tic-Tac-Toe ---
# TTT has a massive state space (approx 5,478 non-terminal states), requiring more episodes.
TOTAL_EPISODES = 50000 
ALPHA = 0.1
GAMMA = 0.99 
# Epsilon decay is CRUCIAL for TTT to shift from random play to learned policy
EPSILON_DECAY = 0.99995 
EPSILON_MIN = 0.05

def train_agent_tictactoe(env, agent, episodes=TOTAL_EPISODES, verbose=True):
    """
    Train a tabular RL agent on the Tic-Tac-Toe environment, 
    including epsilon decay.
    
    Returns: A list of win/loss/draw results (1 for win, 0 for draw, -1 for loss)
    """
    results_list = []
    
    # Store initial epsilon for decay
    initial_epsilon = agent.epsilon 
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False

        # Decay epsilon slightly each episode
        agent.epsilon = max(EPSILON_MIN, initial_epsilon * (EPSILON_DECAY ** ep))

        while not done:
            next_state, reward, done, info = env.step(action)
            
            # Learn step
            if isinstance(agent, SARSAAgent):
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            else:  # Q-Learning
                agent.learn(state, action, reward, next_state)
                action = agent.choose_action(next_state)

            state = next_state
        
        # Record game outcome based on reward (1 for win, -1 for loss, 0.5 for draw, -10 for invalid move)
        if info.get('winner') == 'X':
            results_list.append(1)
        elif info.get('winner') == 'O':
            results_list.append(-1)
        elif info.get('winner') == 'draw':
            results_list.append(0) # Treat draw as 0 for win rate calculation
        else: # Should only happen for invalid move (terminal state, highly penalized)
             results_list.append(-2)
        
        # Print progress every 5000 episodes
        if verbose and (ep + 1) % 5000 == 0:
            avg_result = np.mean([r for r in results_list[-5000:] if r in [-1, 0, 1]])
            win_rate = (results_list[-5000:].count(1) / 5000) * 100
            loss_rate = (results_list[-5000:].count(-1) / 5000) * 100
            draw_rate = (results_list[-5000:].count(0) / 5000) * 100
            
            print(f"Episode {ep + 1}/{episodes} | Win: {win_rate:.1f}%, Loss: {loss_rate:.1f}%, Draw: {draw_rate:.1f}%")
    
    return results_list

def calculate_metric(results, metric='win', window_size=1000):
    """Calculate rolling metric (win/loss/draw rate)."""
    if metric == 'win':
        target = 1
    elif metric == 'loss':
        target = -1
    elif metric == 'draw':
        target = 0
    else:
        raise ValueError("Metric must be 'win', 'loss', or 'draw'")
        
    rates = []
    for i in range(len(results)):
        start_idx = max(0, i - window_size + 1)
        window = results[start_idx:i+1]
        rate = (window.count(target) / len(window)) * 100
        rates.append(rate)
    return rates

def run_comparison(opponent_type='random'):
    """Run and plot the comparison between agents."""
    print(f"\n{'='*70}")
    print(f"STARTING COMPARISON AGAINST '{opponent_type.upper()}' OPPONENT")
    print(f"{'='*70}")
    
    # 1. Initialize environment with desired opponent
    env = TicTacToe(opponent=opponent_type)
    
    # 2. Initialize agents
    qlearn = QLearningAgent(env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    sarsa = SARSAAgent(env.get_actions(), alpha=ALPHA, gamma=GAMMA, epsilon=1.0)
    
    # 3. Train agents
    print("\nTraining Q-Learning Agent...")
    qlearn_results = train_agent_tictactoe(env, qlearn)
    
    print("\nTraining SARSA Agent...")
    sarsa_results = train_agent_tictactoe(env, sarsa)
    
    # 4. Process metrics
    qlearn_win_rate = calculate_metric(qlearn_results, 'win')
    sarsa_win_rate = calculate_metric(sarsa_results, 'win')
    qlearn_loss_rate = calculate_metric(qlearn_results, 'loss')
    sarsa_loss_rate = calculate_metric(sarsa_results, 'loss')
    qlearn_draw_rate = calculate_metric(qlearn_results, 'draw')
    sarsa_draw_rate = calculate_metric(sarsa_results, 'draw')

    
    # 5. Plot Comparison (Win Rate)
    plt.figure(figsize=(12, 6))
    plt.plot(qlearn_win_rate, label='Q-Learning Win Rate', alpha=0.8)
    plt.plot(sarsa_win_rate, label='SARSA Win Rate', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%) (Rolling Average)')
    plt.title(f"Q-Learning vs SARSA: Learning Win Rate Against {opponent_type.title()} Opponent")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tictactoe_win_rate_vs_{opponent_type}.png')
    # plt.show() # Uncomment if running locally and want to see plot immediately
    
    # 6. Plot Comparison (Loss Rate - Crucial for Optimal Opponent)
    plt.figure(figsize=(12, 6))
    plt.plot(qlearn_loss_rate, label='Q-Learning Loss Rate', alpha=0.8)
    plt.plot(sarsa_loss_rate, label='SARSA Loss Rate', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Loss Rate (%) (Rolling Average)')
    plt.title(f"Q-Learning vs SARSA: Learning Loss Rate Against {opponent_type.title()} Opponent")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tictactoe_loss_rate_vs_{opponent_type}.png')
    # plt.show()

    # 7. Print Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (Last 5000 Episodes)")
    print("="*70)
    print(f"Opponent: {opponent_type.title()}")
    
    q_final_win = qlearn_win_rate[-1]
    s_final_win = sarsa_win_rate[-1]
    q_final_loss = qlearn_loss_rate[-1]
    s_final_loss = sarsa_loss_rate[-1]
    q_final_draw = qlearn_draw_rate[-1]
    s_final_draw = sarsa_draw_rate[-1]

    print(f"\n{'Agent':<20} | {'Win Rate (%)':<15} | {'Loss Rate (%)':<15} | {'Draw Rate (%)':<15}")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    print(f"{'Q-Learning':<20} | {q_final_win:<15.2f} | {q_final_loss:<15.2f} | {q_final_draw:<15.2f}")
    print(f"{'SARSA':<20} | {s_final_win:<15.2f} | {s_final_loss:<15.2f} | {s_final_draw:<15.2f}")
    print(f"{'Winner (Q/S)':<20} | {('Q' if q_final_win > s_final_win else 'S'):<15} | {('Q' if q_final_loss < s_final_loss else 'S'):<15} | {('Q' if q_final_draw > s_final_draw else 'S'):<15}")
    print("\n" + "="*70)


if __name__ == "__main__":
    # Scenario 1: Comparison against a Random Opponent
    run_comparison(opponent_type='random')
    
    # Scenario 2: Comparison against an Optimal Opponent (Crucial for report)
    run_comparison(opponent_type='optimal')
    
    print("\nTic-Tac-Toe Analysis Complete! Check your files for plots.")


# --- END OF tictactoe_analysis.py ---