import torch
from copy import deepcopy
import submission.config as config
from submission.LastTry.LastTryLearner import Agent
from hex_engine import hexPosition
import os

from submission.baseline_agent import random_agent, greedy_agent, opponent_adjacent_agent, edge_seeking_agent, center_seeking_agent, corner_seeking_agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from submission.LastTry.gameDataGeneration import generate_play_data as generate_play_data_with_heuristics
import uuid
from submission.LastTry.enemyBlender import OpponentBlender

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

def create_checkpoint_model(run_id, epoch, avg_reward, model):
    color = "white"
    player_token = model.get_player_token()
    if player_token == -1:
        color = "black"

    checkpoint_dir = './checkpoints'
    checkpoint_filename = f'{run_id}_model_{color}_epoch_{epoch}_{avg_reward:.3f}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # check if the checkpoint sub dir is existing
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # picks a random agent for each train game
    # using more than one agent increases the variety
    enemy_blender = OpponentBlender()
    enemy_blender.register_agent(random_agent)
    enemy_blender.register_agent(greedy_agent)
    enemy_blender.register_agent(opponent_adjacent_agent)
    enemy_blender.register_agent(corner_seeking_agent)
    enemy_blender.register_agent(edge_seeking_agent)
    enemy_blender.register_agent(center_seeking_agent)

    # id for the trainings run will be used to file name of the checkpoint models
    run_id = uuid.uuid4().hex[:6]

    player_token = 1
    best_mean_reward = -float("inf")
    epochs = 200
    steps_per_epoch= 2048
    batch_size = 1024
    learning_steps = (steps_per_epoch / batch_size) * epochs # number how many times we will train the model

    model = Agent(config.BOARD_SIZE, player_token, learning_steps).to(device)
    for epoch in range(epochs):
        # 1. Generate play data from self-play
        transitions, win_rate = generate_play_data_with_heuristics(model, enemy_blender, steps_per_epoch)

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
        avg_mean_reward = sum(l["mean_reward"] for l in logs) / len(logs)

        new_checkpoint_found = False
        if best_mean_reward < avg_mean_reward:
            # new checkpoint found
            create_checkpoint_model(run_id, epoch, avg_mean_reward, model)
            best_mean_reward = avg_mean_reward
            new_checkpoint_found = True

        if new_checkpoint_found:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Actor: {avg_actor:.4f} | Critic: {avg_critic:.4f} | WinRate: {win_rate:.4f} | MeanReward: {avg_mean_reward:.4} --> new checkpoint")
        else:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Actor: {avg_actor:.4f} | Critic: {avg_critic:.4f} | WinRate: {win_rate:.4f} | MeanReward: {avg_mean_reward:.4}")

    create_checkpoint_model(run_id, 999, 999, model)
    print("Training complete.")

