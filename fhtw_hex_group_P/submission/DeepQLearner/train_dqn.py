import torch
from copy import deepcopy
import submission.config as config
from submission.DeepQLearner.DeepQLearner import ReplayBuffer, HexDQNAgent, dqn_agent
from hex_engine import hexPosition

def generate_self_play_data(env, agent_fn, buffer, num_games=100):
    """
    Play self-play Hex games and collect transitions into the replay buffer.

    Parameters:
        env (hexPosition): The game environment.
        agent_fn (function): Function(board, action_set) -> action
        buffer (ReplayBuffer): Replay buffer to store transitions.
        num_games (int): Number of games to simulate.
    """
    for game in range(num_games):
        env.reset()
        state = deepcopy(env.board)

        trajectory = []
        while env.winner == 0:
            action_set = env.get_action_space()
            action = agent_fn(state, action_set)

            scalar_action = env.coordinate_to_scalar(action)

            env.move(action)

            next_state = deepcopy(env.board)
            done = env.winner != 0

            # Store transition
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

            buffer.push(
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
    replay_buffer = ReplayBuffer(capacity=10_000, board_size=config.BOARD_SIZE)
    generate_self_play_data(env, dqn_agent, replay_buffer, num_games=1000)

    # initialize the model and provide it with the data
    model = HexDQNAgent(config.BOARD_SIZE,replay_buffer=replay_buffer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        loss = model.train_step(batch_size=64)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "dqnagent.pt")