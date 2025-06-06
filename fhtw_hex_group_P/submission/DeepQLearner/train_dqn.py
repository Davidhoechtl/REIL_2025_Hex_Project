import numpy as np
import torch
from copy import deepcopy
import submission.config as config
from submission.DeepQLearner.DeepQLearner import ReplayBuffer, HexDQNAgent, dqn_agent
from hex_engine import hexPosition
from submission.baseline_agent import random_agent, greedy_agent
import random
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_self_play_data(env, dqn_agent_model, opponents, device, num_games=100):
    """
    Play self-play Hex games and collect transitions into the replay buffer.

    Parameters:
        env (hexPosition): The game environment.
        dqn_agent_model: (model_with_replay_buffer)
        opponent_agents (function): Function(board, action_set) -> action
        num_games (int): Number of games to simulate.
    """
    player_games_won = 0;
    for game in range(num_games):
        env.reset()
        state = deepcopy(env.board)

        # Randomly decide who goes first and which agent is which
        dqn_player = 1 # IMPORTANT NEED TO BE 1 for the model. Feature engineering what is player move what enemy is bound to this. see select_action in DeepQLearner.py
        opponent_fn = random.choice(opponents)

        trajectory = []
        while env.winner == 0:
            action_set = env.get_action_space()
            if env.player == dqn_player:
                action = dqn_agent_model.select_action(torch.tensor(state, dtype=torch.float32), action_set)
            else:
                action = opponent_fn(torch.tensor(state, dtype=torch.float32), action_set)
            scalar_action = env.coordinate_to_scalar(action)

            env.move(action)

            next_state = deepcopy(env.board)
            done = env.winner != 0

            # Store transition (only when model plays)
            if env.player == dqn_player:
                trajectory.append({
                    "state": state,
                    "action": scalar_action,
                    "next_state": next_state,
                    "done": done
                })

            state = next_state

        # Assign final reward based on winner
        reward = 1 if env.winner == dqn_player else -1  # white wins: +1, black wins: -1
        if env.winner == dqn_player:
            player_games_won += 1

        for idx, step in enumerate(trajectory):
            penalty = config.step_penalty * idx
            step["reward"] = reward - penalty  # final outcome reward for all moves

            dqn_agent_model.store_transition(
                torch.tensor(step["state"], dtype=torch.float32),
                step["action"],
                step["reward"],
                torch.tensor(step["next_state"], dtype=torch.float32),
                step["done"]
            )
    print(f"Model won {player_games_won}/{num_games} games ({player_games_won/num_games:.2%})")
    return player_games_won/num_games



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # generate data through self playing
    env = hexPosition(size=config.BOARD_SIZE)

    best_loss = float('inf')
    no_improve_epochs = 0
    epoch_loss = []
    win_rates = []

    # initialize the model and provide it with the data
    model = HexDQNAgent(config.BOARD_SIZE, 5000).to(device)
    generate_self_play_data(env, model, [model.select_action], device, num_games=100)  # model plays next 100 games to refill the replay buffer
    local = deepcopy(model)  # keep a local copy of the model for evaluation
    for epoch in range(config.EPOCHS):
        if epoch % config.new_games_played_in_epoch == 0:
            win_rate = generate_self_play_data(env, model, [local.select_action], device, num_games=100)
            win_rate_for_game_cycle = [win_rate] * config.new_games_played_in_epoch
            win_rates.extend(win_rate_for_game_cycle)
            local = deepcopy(model)  # keep a local copy of the model for evaluation

        loss = model.train_step(batch_size=128)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {loss:.4f}")

        epoch_loss.append(loss.detach().cpu().item())
        if loss is not None:
            if loss < best_loss:
                best_loss = loss
                no_improve_epochs = 0

                # Save best model checkpoint
                torch.save(model.state_dict(), "dqnagent.pt")
                print(f"New best model saved with loss {best_loss:.4f} at epoch {epoch + 1}")
            else:
                no_improve_epochs += 1
                print(f"No improvement for {no_improve_epochs} epochs")

            if no_improve_epochs >= config.patience:
                print(f"Early stopping triggered after {config.patience} epochs without improvement.")
                break

    # After training, plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="Training Loss")
    plt.plot(range(1, len(win_rates) + 1), win_rates, label="win rate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    input("Training complete. Press Enter to exit...")