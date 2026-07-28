import numpy as np
import pickle
from typing import Tuple, List
import os
import sys

def resource_path(filename):
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(__file__), filename)

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
        '''
        [0,0,0]
        [0,0,0]
        [0,0,0]
        '''
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
        path = resource_path(filename)
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Agent loaded from {path}")


class RandomAgent:
    """Opponent that plays random valid moves"""
    def choose_action(self, valid_moves: List[int]) -> int:
        return np.random.choice(valid_moves)


def train_agent(opp=RandomAgent(), episodes=10000):
    """Train the Q-learning agent"""
    game = TicTacToe()
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    opponent = opp
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        
        while not done:
            # Agent's turn (player 1)
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            next_state, reward, done = game.make_move(action)
            agent.update_q_value(state, action, reward, next_state, game.get_valid_moves())
            state = next_state
            
            if done:
                break
            
            # Opponent's turn (player -1)
            valid_moves = game.get_valid_moves()
            if isinstance(opponent, QLearningAgent):
                opponent_action = opponent.choose_action(state, valid_moves)
            else:
                opponent_action = opponent.choose_action(valid_moves)
            next_state, opponent_reward, done = game.make_move(opponent_action)
            state = next_state
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")
    
    return agent

def evaluate_agent(agent, opp=RandomAgent(), episodes=100, verbose=False):
    """Evaluate trained agent against random opponent"""
    game = TicTacToe()
    opponent = opp
    
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        
        while not done:
            # Agent's turn (player 1) - use greedy policy (no exploration)
            valid_moves = game.get_valid_moves()
            action = agent.get_best_action(state, valid_moves)
            next_state, reward, done = game.make_move(action)
            state = next_state
            
            if done:
                if reward == 1:
                    wins += 1
                elif reward == 0.5:
                    draws += 1
                break
            
            # Opponent's turn (player -1)
            valid_moves = game.get_valid_moves()
            if isinstance(opponent, QLearningAgent):
                opponent_action = opponent.get_best_action(state, valid_moves)
            else:
                opponent_action = opponent.choose_action(valid_moves)
            next_state, opponent_reward, done = game.make_move(opponent_action)
            state = next_state
            
            if done:
                losses += 1
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Eval Episode {episode + 1}/{episodes}")
    
    win_rate = wins / episodes * 100
    draw_rate = draws / episodes * 100
    loss_rate = losses / episodes * 100
    
    print(f"\n=== Evaluation Results ({episodes} games) ===")
    print(f"Wins:  {wins} ({win_rate:.1f}%)")
    print(f"Draws: {draws} ({draw_rate:.1f}%)")
    print(f"Losses: {losses} ({loss_rate:.1f}%)")
    
    return wins, draws, losses


def train_agents_together(episodes=10000):
    """Train two agents against each other, they both learn from each game"""
    game = TicTacToe()
    agent1 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    agent2 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        moves_history = []  # Track (agent, state, action) for learning
        
        # Randomly decide who goes first
        agent1_first = np.random.choice([True, False])
        
        while not done:
            if agent1_first:
                # Agent 1's turn (player 1)
                valid_moves = game.get_valid_moves()
                action = agent1.choose_action(state, valid_moves)
                moves_history.append(('agent1', state.copy(), action))
                next_state, reward, done = game.make_move(action)
                state = next_state
                
                if done:
                    # Agent 1 learns from the result
                    for agent_name, s, a in moves_history:
                        if agent_name == 'agent1':
                            agent1.update_q_value(s, a, reward, state, game.get_valid_moves())
                    # Agent 2 learns from the loss
                    if moves_history[-1][0] == 'agent1':
                        agent2.update_q_value(moves_history[-2][1], moves_history[-2][2], -reward if reward == 1 else reward, state, game.get_valid_moves())
                    
                    if reward == 1:
                        agent1_wins += 1
                    elif reward == 0.5:
                        draws += 1
                    break
                
                # Agent 2's turn (player -1)
                valid_moves = game.get_valid_moves()
                action = agent2.choose_action(state, valid_moves)
                moves_history.append(('agent2', state.copy(), action))
                next_state, opponent_reward, done = game.make_move(action)
                state = next_state
                
                if done:
                    # Agent 2 learns from the result
                    for agent_name, s, a in moves_history:
                        if agent_name == 'agent2':
                            agent2.update_q_value(s, a, opponent_reward, state, game.get_valid_moves())
                    # Agent 1 learns from the loss
                    if moves_history[-1][0] == 'agent2':
                        agent1.update_q_value(moves_history[-2][1], moves_history[-2][2], -opponent_reward if opponent_reward == 1 else opponent_reward, state, game.get_valid_moves())
                    
                    if opponent_reward == 1:
                        agent2_wins += 1
                    elif opponent_reward == 0.5:
                        draws += 1
            else:
                # Agent 2's turn first (player 1)
                valid_moves = game.get_valid_moves()
                action = agent2.choose_action(state, valid_moves)
                moves_history.append(('agent2', state.copy(), action))
                next_state, reward, done = game.make_move(action)
                state = next_state
                
                if done:
                    # Agent 2 learns from the result
                    for agent_name, s, a in moves_history:
                        if agent_name == 'agent2':
                            agent2.update_q_value(s, a, reward, state, game.get_valid_moves())
                    # Agent 1 learns from the loss
                    if moves_history[-1][0] == 'agent2':
                        agent1.update_q_value(moves_history[-2][1], moves_history[-2][2], -reward if reward == 1 else reward, state, game.get_valid_moves())
                    
                    if reward == 1:
                        agent2_wins += 1
                    elif reward == 0.5:
                        draws += 1
                    break
                
                # Agent 1's turn (player -1)
                valid_moves = game.get_valid_moves()
                action = agent1.choose_action(state, valid_moves)
                moves_history.append(('agent1', state.copy(), action))
                next_state, opponent_reward, done = game.make_move(action)
                state = next_state
                
                if done:
                    # Agent 1 learns from the result
                    for agent_name, s, a in moves_history:
                        if agent_name == 'agent1':
                            agent1.update_q_value(s, a, opponent_reward, state, game.get_valid_moves())
                    # Agent 2 learns from the loss
                    if moves_history[-1][0] == 'agent1':
                        agent2.update_q_value(moves_history[-2][1], moves_history[-2][2], -opponent_reward if opponent_reward == 1 else opponent_reward, state, game.get_valid_moves())
                    
                    if opponent_reward == 1:
                        agent1_wins += 1
                    elif opponent_reward == 0.5:
                        draws += 1
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} - Agent1: {agent1_wins}, Agent2: {agent2_wins}, Draws: {draws}")
    
    # Return the better agent
    if agent1_wins >= agent2_wins:
        print(f"\nAgent 1 is better! ({agent1_wins} wins vs {agent2_wins})")
        return agent1
    else:
        print(f"\nAgent 2 is better! ({agent2_wins} wins vs {agent1_wins})")
        return agent2

if __name__ == "__main__":
    print("Training two agents against each other...\n")
    EPISODES = 2500000
    
    # Load best agent if it exists, otherwise create fresh agents
    if os.path.exists("best_agent.pkl"):
        print("Loading best agent and creating two instances to battle...\n")
        agent1 = QLearningAgent()
        agent1.load("best_agent.pkl")
        
        agent2 = QLearningAgent()
        agent2.load("best_agent.pkl")
    else:
        print("No existing best agent found. Creating two fresh agents...\n")
        agent1 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
        agent2 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    
    # Train both agents together
    best_agent = train_agents_together(episodes=EPISODES)
    print("Training complete!\n")
    
    print("Evaluating best agent vs random...")
    evaluate_agent(best_agent, episodes=100)
    
    # Save the best trained agent
    best_agent.save("best_agent.pkl")