import torch
import torch.optim as optim
import numpy as np
import submission.config as config
from hex_engine import hexPosition
from submission.terminator import HexNet, net_agent
from submission.baseline_agent import random_agent, greedy_agent

def board_to_tensor(board):
    return torch.tensor(board, dtype=torch.float32)

def play_selfplay_game(agent, board_size):
    game = hexPosition(size=board_size)
    states, actions, players = [], [], []
    while game.winner == 0:
        state = np.array(game.board)
        action_set = game.get_action_space()
        action = agent(game.board, action_set)
        states.append(state)
        actions.append(game.coordinate_to_scalar(action))
        players.append(game.player)
        game.move(action)
    # Assign winner to each state
    rewards = [game.winner * p for p in players]
    return states, actions, rewards

import random

def generate_data(n_games, model, board_size):
    all_states, all_actions, all_rewards = [], [], []
    net_play_agent = net_agent(model, board_size)
    for _ in range(n_games):
        # Randomly pick an agent for this game
        agent_choice = random.choice([random_agent, greedy_agent, net_play_agent])
        s, a, r = play_selfplay_game(agent_choice, board_size)
        all_states.extend(s)
        all_actions.extend(a)
        all_rewards.extend(r)
    return np.array(all_states), np.array(all_actions), np.array(all_rewards)

def train_hexnet(model, optimizer, states, actions, rewards, epochs=1, batch_size=32):
    model.train()
    n = len(states)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_states = torch.tensor(states[batch_idx], dtype=torch.float32, device=device)
            batch_actions = torch.tensor(actions[batch_idx], dtype=torch.long, device=device)
            batch_rewards = torch.tensor(rewards[batch_idx], dtype=torch.float32, device=device)
            logits, values = model(batch_states)
            policy_loss = torch.nn.functional.nll_loss(logits, batch_actions)
            value_loss = torch.nn.functional.mse_loss(values.view(-1), batch_rewards.view(-1))
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.detach().item()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    
    model = HexNet(config.BOARD_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        states, actions, rewards = generate_data(config.GAMES_PER_EPOCH, model, config.BOARD_SIZE)
        loss = train_hexnet(model, optimizer, states, actions, rewards, epochs=1, batch_size=config.BATCH_SIZE)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "hexnet.pt")