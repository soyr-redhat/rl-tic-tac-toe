import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from train import TicTacToe, QLearningAgent


def play_gui(agent):
    """Play tic tac toe against the trained agent using GUI"""
    window = tk.Tk()
    window.title("Tic Tac Toe vs AI")
    window.geometry("400x500")
    
    game = TicTacToe()
    state = game.reset()
    game_over = False
    player_first = np.random.choice([True, False])  # Randomize who goes first
    
    buttons = [[None for _ in range(3)] for _ in range(3)]
    
    def get_button_index(row, col):
        return row * 3 + col
    
    def update_buttons():
        """Update button display based on current board state"""
        board_2d = state.reshape(3, 3)
        for row in range(3):
            for col in range(3):
                val = board_2d[row, col]
                if val == 1:
                    buttons[row][col].config(text="X", font=("Arial", 20, "bold"), fg="red")
                elif val == -1:
                    buttons[row][col].config(text="O", font=("Arial", 20, "bold"), fg="blue")
                else:
                    buttons[row][col].config(text="", font=("Arial", 20, "bold"), state="normal")
    
    def make_player_move(row, col):
        nonlocal state, game_over
        
        if game_over:
            return
        
        pos = get_button_index(row, col)
        valid_moves = game.get_valid_moves()
        
        if pos not in valid_moves:
            messagebox.showerror("Invalid", "That position is already taken!")
            return
        
        # Player move
        next_state, reward, done = game.make_move(pos)
        state = next_state
        update_buttons()
        
        if done:
            if reward == 0.5:
                messagebox.showinfo("Game Over", "Draw!")
            else:
                messagebox.showinfo("Game Over", "You won!")
            game_over = True
            return
        
        # AI move
        window.after(500, make_ai_move)
    
    def make_ai_move():
        nonlocal state, game_over
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            messagebox.showinfo("Game Over", "Draw!")
            game_over = True
            return
        
        ai_move = agent.get_best_action(state, valid_moves)
        next_state, reward, done = game.make_move(ai_move)
        state = next_state
        update_buttons()
        
        if done:
            if reward == 0.5:
                messagebox.showinfo("Game Over", "Draw!")
            else:
                messagebox.showinfo("Game Over", "AI won!")
            game_over = True
    
    def start_ai_first():
        """Let AI go first"""
        nonlocal state, game_over
        if game_over:
            return
        
        valid_moves = game.get_valid_moves()
        if valid_moves:
            ai_move = agent.get_best_action(state, valid_moves)
            next_state, reward, done = game.make_move(ai_move)
            state = next_state
            update_buttons()
            
            if done:
                if reward == 0.5:
                    messagebox.showinfo("Game Over", "Draw!")
                else:
                    messagebox.showinfo("Game Over", "AI won!")
                game_over = True
    
    def reset_game():
        nonlocal state, game_over, player_first
        state = game.reset()
        game_over = False
        player_first = np.random.choice([True, False])
        update_buttons()
        
        if not player_first:
            title_label.config(text="AI is X (going first), You are O")
            window.after(500, start_ai_first)
        else:
            title_label.config(text="You are X (going first), AI is O")
    
    # Create title label
    title_label = tk.Label(window, text="", font=("Arial", 14))
    title_label.pack(pady=10)
    
    # Create game board
    board_frame = tk.Frame(window)
    board_frame.pack(pady=10)
    
    for row in range(3):
        for col in range(3):
            btn = tk.Button(
                board_frame,
                width=8,
                height=4,
                font=("Arial", 20, "bold"),
                command=lambda r=row, c=col: make_player_move(r, c)
            )
            btn.grid(row=row, column=col, padx=2, pady=2)
            buttons[row][col] = btn
    
    # Create reset button
    reset_btn = tk.Button(window, text="New Game", command=reset_game, font=("Arial", 12))
    reset_btn.pack(pady=10)
    
    update_buttons()
    reset_game()  # Initialize the first game
    window.mainloop()


if __name__ == "__main__":
    agent = QLearningAgent()
    agent.load("best_agent.pkl")
    play_gui(agent)