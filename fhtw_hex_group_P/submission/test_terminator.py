import torch
import numpy as np
from hex_engine import hexPosition
from submission.terminator import HexNet
from submission.baseline_agent import random_agent, greedy_agent

def net_agent(model, board_size):
    def agent(board, action_set):
        model.eval()
        state = torch.tensor(np.array(board), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state)
        probs = torch.exp(logits).squeeze().numpy()
        # Mask illegal moves
        mask = np.zeros(board_size * board_size)
        for a in action_set:
            mask[board_size * a[0] + a[1]] = 1
        probs = probs * mask
        if probs.sum() == 0:
            idx = np.random.choice(np.where(mask == 1)[0])
        else:
            idx = np.argmax(probs)
        move = (idx // board_size, idx % board_size)
        return move
    return agent

if __name__ == "__main__":
    board_size = 5
    model = HexNet(board_size)
    model.load_state_dict(torch.load("hexnet.pt"))
    agent = net_agent(model, board_size)
    # Play 20 games vs random
    wins = 0
    for _ in range(20):
        game = hexPosition(size=board_size)
        while game.winner == 0:
            if game.player == 1:
                move = agent(game.board, game.get_action_space())
            else:
                move = greedy_agent(game.board, game.get_action_space())
            game.move(move)
        if game.winner == 1:
            wins += 1
    print(f"Net agent won {wins}/20 vs greedy agent.")