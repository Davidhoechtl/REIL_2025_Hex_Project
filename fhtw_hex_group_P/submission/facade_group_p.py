from submission.LastTry.LastTryLearner import Agent
import torch
import os

_agent_white = None
def agent_white(board, action_set):
    player_token = 1
    # load model that is trained on player_token == 1 trajectories
    global _agent_white
    if _agent_white is None:
        board_size = len(board)
        _agent_white = load_the_best_model(player_token, board_size)

    return _agent_white.select_action(torch.tensor(board, dtype=torch.float32), action_set)

_agent_black = None
def agent_black(board, action_set):
    player_token = -1
    # load model that is trained on player_token == -1 trajectories
    global _agent_black
    if _agent_black is None:
        board_size = len(board)
        _agent_black = load_the_best_model(player_token, board_size)

    return _agent_black.select_action(torch.tensor(board, dtype=torch.float32), action_set)

def load_the_best_model(player_token, board_size):
    model = Agent(board_size)
    # Get the directory where the script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, 'LastTry', 'checkpoints')
    if player_token == 1:
        model_file = os.path.join(checkpoint_dir, '66541b_model_white_epoch_31_0.993.pt')
        model.load_state_dict(torch.load(model_file)) # player_token == 1 -> white stones
    else:
        model.load_state_dict(torch.load("black_model.pt")) # player_token == -1 -> black stones

    if model is not None:
        print("Successfully loaded the model from .pt file.")

    return model