from copy import deepcopy
from submission.baseline_agent import random_agent, greedy_agent
from submission.DeepQLearner.DeepQLearner import dqn_agent, HexDQNAgent, load_dqn_agent
import torch
from hex_engine import hexPosition
import submission.config as config

def evaluate_agent(env, dqn_agent_function, num_games=100):
    results = {
        "vs_greedy": {"wins": 0, "losses": 0, "draws": 0},
        "vs_random": {"wins": 0, "losses": 0, "draws": 0},
    }

    def play_game(agent1, agent2):
        env.reset()
        current_player = 1
        while env.winner == 0:
            state = deepcopy(env.board)
            action_set = env.get_action_space()

            if current_player == 1:
                move = agent1(state, action_set)
            else:
                move = agent2(state, action_set)

            env.move(move)
            current_player *= -1

        return env.winner

    print("Evaluating against Random Agent...")
    for _ in range(num_games):
        winner = play_game(dqn_agent_function, random_agent)
        if winner == 1:
            results["vs_random"]["wins"] += 1
        elif winner == -1:
            results["vs_random"]["losses"] += 1
        else:
            results["vs_random"]["draws"] += 1

    print("Evaluating against Greedy Agent...")
    for _ in range(num_games):
        winner = play_game(dqn_agent_function, greedy_agent)
        if winner == 1:
            results["vs_greedy"]["wins"] += 1
        elif winner == -1:
            results["vs_greedy"]["losses"] += 1
        else:
            results["vs_greedy"]["draws"] += 1

    # Report
    for opponent in results:
        stats = results[opponent]
        total = sum(stats.values())
        print(f"Results vs {opponent}:")
        print(f"  Wins: {stats['wins']} ({stats['wins']/total:.2%})")
        print(f"  Losses: {stats['losses']} ({stats['losses']/total:.2%})")
        print(f"  Draws: {stats['draws']} ({stats['draws']/total:.2%})")

    return results

if __name__ == "__main__":
    environment = hexPosition(config.BOARD_SIZE)
    load_dqn_agent("a2cagent.pt")
    evaluate_agent(environment, dqn_agent, num_games=400)