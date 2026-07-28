import numpy as np
import pickle
from train import TicTacToe, QLearningAgent

def train_selfplay(episodes=10_000_000):
    game = TicTacToe()
    agent1 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.15)
    agent2 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.15)

    for episode in range(episodes):
        state = game.reset()
        done = False
        history = []
        first_is_1 = episode % 2 == 0

        agents = (agent1, agent2) if first_is_1 else (agent2, agent1)

        turn = 0
        while not done:
            agent = agents[turn % 2]
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            history.append((turn % 2, state.copy(), action))
            next_state, reward, done = game.make_move(action)
            state = next_state
            turn += 1

        for idx, (side, s, a) in enumerate(history):
            agent = agents[side]
            if done and reward == 1:
                winner_side = (turn - 1) % 2
                if side == winner_side:
                    agent.update_q_value(s, a, 1.0, state, [])
                else:
                    agent.update_q_value(s, a, -1.0, state, [])
            elif done and reward == 0.5:
                agent.update_q_value(s, a, 0.3, state, [])

        if (episode + 1) % 1_000_000 == 0:
            print(f"Episode {episode + 1:,}/{episodes:,}")

    merged = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.0)
    merged.q_table = {**agent1.q_table}
    for k, v in agent2.q_table.items():
        if k not in merged.q_table or abs(v) > abs(merged.q_table[k]):
            merged.q_table[k] = v

    return merged

if __name__ == "__main__":
    print("Training with 10M episodes of pure self-play...\n")
    agent = train_selfplay(10_000_000)
    print(f"\nQ-table size: {len(agent.q_table):,} entries")

    from train import evaluate_agent, RandomAgent
    print("\nEvaluating vs random (agent goes first):")
    evaluate_agent(agent, episodes=1000)

    agent.save("best_agent.pkl")
    print("\nSaved to best_agent.pkl")
