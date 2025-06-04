import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HexNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.policy_head = nn.Conv2d(64, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Conv2d(64, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, board_size, board_size]
        x = x.unsqueeze(1)  # [batch, 1, board_size, board_size]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_head(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_head(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
    
def net_agent(model, board_size):
    def agent(board, action_set):
        model.eval()
        # Ensure the state is on the same device as the model
        device = next(model.parameters()).device
        state = torch.tensor(np.array(board), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state)
        probs = torch.exp(logits).squeeze().cpu().numpy()
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