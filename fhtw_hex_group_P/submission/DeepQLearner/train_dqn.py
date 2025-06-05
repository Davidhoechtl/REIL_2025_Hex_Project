import numpy as np
import torch
from copy import deepcopy
import submission.config as config
from submission.DeepQLearner.DeepQLearner import ReplayBuffer, HexDQNAgent, dqn_agent
from hex_engine import hexPosition
from submission.baseline_agent import random_agent, greedy_agent
import random

def generate_self_play_data(env, dqn_agent_model, opponents, device, num_games=100):
    """
    Play self-play Hex games and collect transitions into the replay buffer.

    Parameters:
        env (hexPosition): The game environment.
        dqn_agent_model: (model_with_replay_buffer)
        opponent_agents (function): Function(board, action_set) -> action
        num_games (int): Number of games to simulate.
    """
    for game in range(num_games):
        env.reset()
        state = deepcopy(env.board)

        # Randomly decide who goes first and which agent is which
        dqn_player = random.choice([1, -1])
        opponent_fn = random.choice(opponents)

        trajectory = []
        while env.winner == 0:
            action_set = env.get_action_space()
            if env.player == dqn_player:
                #state_tensor = torch.tensor(state, dtype=torch.float32).to(device) # send the current state to the gpu
                action = dqn_agent_model.select_action(np.array(state), action_set)
            else:
                action = opponent_fn(np.array(state), action_set)
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
        reward = 1 if env.winner == 1 else -1  # white wins: +1, black wins: -1

        # Post-process rewards (if needed)
        for step in trajectory:
            step["reward"] = reward  # final outcome reward for all moves

            dqn_agent_model.store_transition(
                torch.tensor(step["state"], dtype=torch.float32),
                step["action"],
                step["reward"],
                torch.tensor(step["next_state"], dtype=torch.float32),
                step["done"]
            )


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

    # initialize the model and provide it with the data
    model = HexDQNAgent(config.BOARD_SIZE, 50_000).to(device)
    generate_self_play_data(env, model, [greedy_agent, random_agent], device, num_games=5000)  # model plays next 100 games to refill the replay buffer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        loss = model.train_step(batch_size=64)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "dqnagent.pt")