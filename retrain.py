import numpy as np
from train import TicTacToe, QLearningAgent, RandomAgent


def train_selfplay(episodes=10_000_000):
    game = TicTacToe()
    agent1 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.15)
    agent2 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.15)

    for episode in range(episodes):
        state = game.reset()
        done = False
        history = []
        agents = (agent1, agent2) if episode % 2 == 0 else (agent2, agent1)

        turn = 0
        while not done:
            agent = agents[turn % 2]
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            history.append((turn % 2, state.copy(), action))
            next_state, reward, done = game.make_move(action)
            state = next_state
            turn += 1

        winner_side = (turn - 1) % 2
        for side, s, a in history:
            agent = agents[side]
            if reward == 1:
                r = 1.0 if side == winner_side else -1.0
            else:
                r = 0.3
            agent.update_q_value(s, a, r, state, [])

        if (episode + 1) % 1_000_000 == 0:
            print(f"Episode {episode + 1:,}/{episodes:,}")

    merged = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.0)
    merged.q_table = {**agent1.q_table}
    for k, v in agent2.q_table.items():
        if k not in merged.q_table or abs(v) > abs(merged.q_table[k]):
            merged.q_table[k] = v

    return merged


def evaluate(agent, episodes=1000):
    game = TicTacToe()
    opponent = RandomAgent()
    wins = losses = draws = 0

    for _ in range(episodes):
        state = game.reset()
        done = False
        while not done:
            action = agent.get_best_action(state, game.get_valid_moves())
            state, reward, done = game.make_move(action)
            if done:
                if reward == 1: wins += 1
                else: draws += 1
                break
            opp_action = opponent.choose_action(game.get_valid_moves())
            state, _, done = game.make_move(opp_action)
            if done: losses += 1

    print(f"\nEval vs random ({episodes} games):")
    print(f"  Wins: {wins} ({wins/episodes*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/episodes*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/episodes*100:.1f}%)")


if __name__ == "__main__":
    print("Training with 10M episodes of pure self-play...\n")
    agent = train_selfplay(10_000_000)
    print(f"\nQ-table size: {len(agent.q_table):,} entries")
    evaluate(agent)
    agent.save("best_agent.pkl")
