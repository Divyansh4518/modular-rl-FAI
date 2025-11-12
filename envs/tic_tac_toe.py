import numpy as np
import random
from envs.base_env import GameEnv

class TicTacToe(GameEnv):
    """
    Tic-Tac-Toe environment for reinforcement learning.
    
    The agent plays as 'X' and the opponent plays as 'O'.
    State is represented as a tuple of tuples (immutable for dictionary keys).
    Actions are (row, col) tuples indicating where to place the mark.
    """
    
    def __init__(self, opponent='random'):
        """
        Initialize the Tic-Tac-Toe environment.
        
        Args:
            opponent: Type of opponent ('random', 'optimal', or 'none' for self-play)
        """
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.opponent_type = opponent
        self.current_player = 'X'  # Agent always plays as X
        
    def reset(self):
        """Reset the board to empty state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        return self._get_state()
    
    def _get_state(self):
        """Convert board to immutable state representation."""
        return tuple(tuple(row) for row in self.board)
    
    def get_actions(self, state=None):
        """
        Return available actions (empty positions).
        
        Args:
            state: Board state to check. If None, uses current board.
            
        Returns:
            List of (row, col) tuples for empty positions
        """
        if state is not None:
            # Convert state back to list for checking
            board = [list(row) for row in state]
        else:
            board = self.board
        
        actions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    actions.append((i, j))
        return actions
    
    def step(self, action):
        """
        Execute an action (place X on the board).
        
        Args:
            action: Tuple (row, col) where agent wants to place 'X'
            
        Returns:
            next_state: New board state
            reward: Reward for the action
            done: Whether game has ended
            info: Additional information
        """
        row, col = action
        
        # Check if move is valid
        if self.board[row][col] != ' ':
            # Invalid move - penalize heavily
            return self._get_state(), -10, True, {'invalid_move': True}
        
        # Place agent's mark
        self.board[row][col] = 'X'
        
        # Check if agent won
        if self._check_win('X'):
            return self._get_state(), 1, True, {'winner': 'X'}
        
        # Check if board is full (draw)
        if self._is_full():
            return self._get_state(), 0.5, True, {'winner': 'draw'}
        
        # Opponent's turn
        if self.opponent_type != 'none':
            self._opponent_move()
            
            # Check if opponent won
            if self._check_win('O'):
                return self._get_state(), -1, True, {'winner': 'O'}
            
            # Check if board is full after opponent move
            if self._is_full():
                return self._get_state(), 0.5, True, {'winner': 'draw'}
        
        # Game continues
        return self._get_state(), 0, False, {}
    
    def _check_win(self, player):
        """Check if the specified player has won."""
        # Check rows
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        
        # Check columns
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def _is_full(self):
        """Check if the board is full."""
        return all(cell != ' ' for row in self.board for cell in row)
    
    def _opponent_move(self):
        """Execute opponent's move based on opponent type."""
        if self.opponent_type == 'random':
            self._random_opponent_move()
        elif self.opponent_type == 'optimal':
            self._optimal_opponent_move()
    
    def _random_opponent_move(self):
        """Random opponent - picks a random empty cell."""
        available = self.get_actions()
        if available:
            row, col = random.choice(available)
            self.board[row][col] = 'O'
    
    def _optimal_opponent_move(self):
        """
        Optimal opponent using minimax strategy.
        Tries to win, blocks agent's wins, and plays strategically.
        """
        # Try to win
        move = self._find_winning_move('O')
        if move:
            self.board[move[0]][move[1]] = 'O'
            return
        
        # Block agent's winning move
        move = self._find_winning_move('X')
        if move:
            self.board[move[0]][move[1]] = 'O'
            return
        
        # Take center if available
        if self.board[1][1] == ' ':
            self.board[1][1] = 'O'
            return
        
        # Take a corner
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        available_corners = [c for c in corners if self.board[c[0]][c[1]] == ' ']
        if available_corners:
            row, col = random.choice(available_corners)
            self.board[row][col] = 'O'
            return
        
        # Take any available space
        available = self.get_actions()
        if available:
            row, col = random.choice(available)
            self.board[row][col] = 'O'
    
    def _find_winning_move(self, player):
        """
        Find a move that would make the player win immediately.
        
        Args:
            player: 'X' or 'O'
            
        Returns:
            (row, col) tuple if winning move exists, None otherwise
        """
        for row, col in self.get_actions():
            # Try the move
            self.board[row][col] = player
            
            # Check if it wins
            if self._check_win(player):
                self.board[row][col] = ' '  # Undo the move
                return (row, col)
            
            # Undo the move
            self.board[row][col] = ' '
        
        return None
    
    def render(self):
        """Print a visual representation of the board."""
        print("\n  0   1   2")
        for i, row in enumerate(self.board):
            print(f"{i} {row[0]} | {row[1]} | {row[2]}")
            if i < 2:
                print(" ---+---+---")
        print()
    
    def get_winner(self):
        """
        Return the winner of the game.
        
        Returns:
            'X' if agent won, 'O' if opponent won, 'draw' if draw, None if game not over
        """
        if self._check_win('X'):
            return 'X'
        if self._check_win('O'):
            return 'O'
        if self._is_full():
            return 'draw'
        return None


class TicTacToeSelfPlay(TicTacToe):
    """
    Tic-Tac-Toe variant for self-play training.
    Both players are controlled by RL agents.
    """
    
    def __init__(self):
        super().__init__(opponent='none')
        self.current_player = 'X'
    
    def step(self, action):
        """
        Execute an action for the current player.
        
        Returns state from perspective of the current player.
        """
        row, col = action
        
        # Check if move is valid
        if self.board[row][col] != ' ':
            return self._get_state(), -10, True, {'invalid_move': True}
        
        # Place current player's mark
        self.board[row][col] = self.current_player
        
        # Check if current player won
        if self._check_win(self.current_player):
            reward = 1 if self.current_player == 'X' else -1
            return self._get_state(), reward, True, {'winner': self.current_player}
        
        # Check for draw
        if self._is_full():
            return self._get_state(), 0, True, {'winner': 'draw'}
        
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return self._get_state(), 0, False, {}


def play_human_vs_agent(agent):
    """
    Allow a human to play against a trained agent.
    
    Args:
        agent: Trained RL agent
    """
    env = TicTacToe(opponent='none')
    state = env.reset()
    done = False
    
    print("You are 'O', Agent is 'X'")
    print("Enter your move as: row col (e.g., '0 1' for top middle)")
    
    while not done:
        env.render()
        
        if env.current_player == 'X':
            # Agent's turn
            action = agent.choose_action(state)
            print(f"Agent plays: {action}")
            state, reward, done, info = env.step(action)
        else:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    move = input("Your move (row col): ")
                    row, col = map(int, move.split())
                    if (row, col) in env.get_actions():
                        env.board[row][col] = 'O'
                        valid_move = True
                        state = env._get_state()
                        
                        # Check if human won
                        if env._check_win('O'):
                            env.render()
                            print("You win!")
                            return
                        if env._is_full():
                            env.render()
                            print("It's a draw!")
                            return
                    else:
                        print("Invalid move! Try again.")
                except (ValueError, IndexError):
                    print("Invalid input! Enter as: row col (e.g., '0 1')")
    
    env.render()
    winner = info.get('winner', 'unknown')
    if winner == 'X':
        print("Agent wins!")
    elif winner == 'draw':
        print("It's a draw!")
