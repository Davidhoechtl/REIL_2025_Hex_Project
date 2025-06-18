import numpy as np
import torch
from copy import deepcopy
import submission.config as config
from submission.DeepQLearner.DeepQLearner import HexDQNAgent
from submission.LastTry.LastTryLearner import Agent
from hex_engine import hexPosition
from submission.baseline_agent import random_agent, greedy_agent
import random
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from submission.LastTry.gameDataGeneration import generate_play_data as generate_play_data_with_heuristics

def generate_play_data(model_fn, enemy_fn, num_steps):
    """
    Generate play data by simulating games between model_fn and enemy_fn.

    Each step generates a transition tuple and stores it into a list, which can
    later be used for batch training or analysis.

    Args:
        model_fn: The agent to collect training data for (must have `select_action`)
        enemy_fn: The opponent agent
        num_steps: Number of moves to generate

    Returns:
        List of transition tuples: (state, action, reward, next_state, done)
    """
    database = []
    steps_collected = 0
    wins = 0
    total_games = 0
    while steps_collected < num_steps:
        game = hexPosition(size=config.BOARD_SIZE)
        game.reset()
        trajectory = []

        player_turn = 1  # the one we collect data for

        while game.winner == 0:
            state = deepcopy(game.board)
            action_space = game.get_action_space()

            if game.player == player_turn:
                action = model_fn.select_action(torch.tensor(state, dtype=torch.float32), action_space)
            else:
                # action = enemy_fn.select_action(torch.tensor(state, dtype=torch.float32), action_space)
                 action = enemy_fn(torch.tensor(state, dtype=torch.float32), action_space)

            scalar_action = game.coordinate_to_scalar(action)
            game.move(action)
            next_state = deepcopy(game.board)
            done = game.winner != 0

            # Only collect data for model_fn's moves
            if game.player != player_turn:  # record previous move
                trajectory.append({
                    "state": state,
                    "action": scalar_action,
                    "next_state": next_state,
                    "done": done,
                    "player": player_turn
                })

        total_games+=1
        if game.winner == player_turn:
            wins+=1

        # Assign rewards based on outcome
        for step in trajectory:
            if step["player"] == game.winner:
                step["reward"] = 1
            elif game.winner == 0:
                step["reward"] = 0
            else:
                step["reward"] = -1

            transition = (
                torch.tensor(step["state"], dtype=torch.float32),
                step["action"],
                step["reward"],
                torch.tensor(step["next_state"], dtype=torch.float32),
                step["done"]
            )
            database.append(transition)
            steps_collected += 1
            if steps_collected >= num_steps:
                break

    return database, wins/total_games

def generate_self_play_data(model_fn, num_games=10):
    for game_idx in range(num_games):
        game = hexPosition(size=config.BOARD_SIZE)
        game.reset()
        trajectory = []

        while game.winner == 0:
            state = deepcopy(game.board)
            action_space = game.get_action_space()

            # model makes a move
            action = model_fn.select_action(torch.tensor(state, dtype=torch.float32), action_space)
            scalar_action = game.coordinate_to_scalar(action)

            game.move(action)

            next_state = deepcopy(game.board)
            done = game.winner != 0

            # store state before move
            trajectory.append({
                "state": state,
                "action": scalar_action,
                "next_state": next_state,
                "done": done,
                "player": game.player
            })

        # assign rewards after game ends
        for step in trajectory:
            if step["player"] == game.winner:
                step["reward"] = 1
            elif game.winner == 0:
                step["reward"] = 0
            else:
                step["reward"] = -1

            model_fn.store_transition(
                torch.tensor(step["state"], dtype=torch.float32),
                step["action"],
                step["reward"],
                torch.tensor(step["next_state"], dtype=torch.float32),
                step["done"]
            )

def validate(model, noob, num_games):
    wins = 0
    for i in range(num_games):
        game = hexPosition(size=config.BOARD_SIZE)
        game.reset()
        while game.winner == 0:
            state = deepcopy(game.board)
            action_space = game.get_action_space()

            # model makes a move
            if game.player == 1:
                action = model.select_action(torch.tensor(state, dtype=torch.float32), action_space)
            else:
                action = noob.select_action(torch.tensor(state, dtype=torch.float32), action_space)

            game.move(action)

        if game.winner == 1:
            wins+=1

    return wins/num_games

def print_training_stats(stats, current_win_rate, step=None):
    """
    Print training statistics in a readable format.

    Args:
        stats (dict): Dictionary containing loss values and metrics.
        step (int, optional): Current training step or episode number.
    """
    prefix = f"[Step {step}] " if step is not None else ""
    print(f"{prefix}"
          f"Loss: {stats['loss']:.4f} | "
          f"Actor: {stats['actor']:.4f} | "
          f"Critic: {stats['critic']:.4f} | "
          f"Adv Mean: {stats['advantage_mean']:.4f} | "
          f"Adv Std: {stats['advantage_std']:.4f} | " 
          f"Win Rate: {current_win_rate:.4f}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    epochs = 100
    steps_per_epoch= 2048
    batch_size = 1024
    learning_steps = (steps_per_epoch / batch_size) * epochs # number how many times we will train the model

    model = Agent(learning_steps).to(device)

    for epoch in range(epochs):
        # 1. Generate play data from self-play
        transitions, win_rate = generate_play_data_with_heuristics(model, random_agent, steps_per_epoch)

        # 2. Unpack the data
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 3. Train in batches
        total_batches = len(states) // batch_size
        logs = []
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            log = model.train_step(
                states[start:end],
                actions[start:end],
                rewards[start:end],
                next_states[start:end],
                dones[start:end],
                batch_size=batch_size
            )
            logs.append(log)

        # 4. Print average stats per epoch
        avg_loss = sum(l["loss"] for l in logs) / len(logs)
        avg_actor = sum(l["actor"] for l in logs) / len(logs)
        avg_critic = sum(l["critic"] for l in logs) / len(logs)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Actor: {avg_actor:.4f} | Critic: {avg_critic:.4f} | WinRate: {win_rate:.4f}")

    print("Training complete.")

