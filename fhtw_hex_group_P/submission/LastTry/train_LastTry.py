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

def generate_self_play_data_static(model_fn, static, num_games=10):
    for game_idx in range(num_games):
        game = hexPosition(size=config.BOARD_SIZE)
        game.reset()
        trajectory = []

        player_turn = 1

        while game.winner == 0:
            state = deepcopy(game.board)
            action_space = game.get_action_space()

            # model makes a move
            if game.player == player_turn:
                action = model_fn.select_action(torch.tensor(state, dtype=torch.float32), action_space)
            else:
                action = static.select_action(torch.tensor(state, dtype=torch.float32), action_space)

            scalar_action = game.coordinate_to_scalar(action)

            game.move(action)
            next_state = deepcopy(game.board)
            done = game.winner != 0

            # store state before move
            if game.player == player_turn:
                trajectory.append({
                    "state": state,
                    "action": scalar_action,
                    "next_state": next_state,
                    "done": done,
                    "player": player_turn
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

    # generate data through self playing
    model = Agent().to(device)
    noob = deepcopy(model)
    generate_self_play_data_static(model, noob, num_games=500)
    epoch_loss = []

    current_win_rate = 0
    for epoch in range(1001):
        # if epoch+1 % config.new_games_played_in_epoch == 0:
        #     generate_self_play_data_static(model, noob, num_games=50)
        if epoch%50==0:
            current_win_rate = validate(model,noob,20)

        loss = model.train_step(batch_size=64)
        epoch_loss.append(loss['loss'])
        print_training_stats(loss, current_win_rate, step=epoch)

    # After training, plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="win rate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

