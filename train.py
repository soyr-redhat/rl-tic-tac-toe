import numpy as np
import pickle
from typing import Tuple, List

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0=empty, 1=player, -1=opponent
        self.current_player = 1
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        """Return list of valid move positions (0-8)"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute a move and return (new_state, reward, done)
        """
        if self.board[position] != 0:
            raise ValueError(f"Invalid move: position {position} already occupied")
        
        self.board[position] = self.current_player
        done = False
        reward = 0
        
        # Check if current player won
        if self._check_winner(self.current_player):
            reward = 1
            done = True
        # Check if it's a draw
        elif len(self.get_valid_moves()) == 0:
            reward = 0.5
            done = True
        
        self.current_player *= -1
        return self.board.copy(), reward, done
    
    def _check_winner(self, player: int) -> bool:
        board_2d = self.board.reshape(3, 3)
        
        # Check rows and columns
        if np.any(np.all(board_2d == player, axis=0)) or np.any(np.all(board_2d == player, axis=1)):
            return True
        
        # Check diagonals
        if (board_2d[0, 0] == board_2d[1, 1] == board_2d[2, 2] == player) or \
           (board_2d[0, 2] == board_2d[1, 1] == board_2d[2, 0] == player):
            return True
        
        return False


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.q_table = {}  # Maps (state, action) -> Q-value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
    def state_to_key(self, board: np.ndarray) -> str:
        """Convert board state to hashable key"""
        return board.tobytes()
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair"""
        key = (self.state_to_key(state), action)
        return self.q_table.get(key, 0.0)
    
    def get_best_action(self, state: np.ndarray, valid_moves: List[int]) -> int:
        """Get action with highest Q-value"""
        if not valid_moves:
            return None
        q_values = [self.get_q_value(state, move) for move in valid_moves]
        return valid_moves[np.argmax(q_values)]
    
    def choose_action(self, state: np.ndarray, valid_moves: List[int]) -> int:
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Explore: random move
            return np.random.choice(valid_moves)
        else:
            # Exploit: best move
            return self.get_best_action(state, valid_moves)
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_valid_moves: List[int]):
        """Update Q-value using Q-learning formula"""
        key = (self.state_to_key(state), action)
        current_q = self.q_table.get(key, 0.0)
        
        # Get max Q-value for next state
        if next_valid_moves:
            next_q = self.get_q_value(next_state, self.get_best_action(next_state, next_valid_moves))
        else:
            next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[key] = new_q
    
    def save(self, filename: str):
        """Save agent to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Agent saved to {filename}")
    
    def load(self, filename: str):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Agent loaded from {filename}")


class RandomAgent:
    def choose_action(self, valid_moves: List[int]) -> int:
        return np.random.choice(valid_moves)