import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import submission.config as config

class ReplayBuffer:
    def __init__(self, capacity, board_size):
        """
        Initialize the replay buffer.
        Args:
            capacity (int): Maximum number of transitions to store.
            board_size (int): Used to flatten state shape for consistency.
        """
        self.capacity = capacity
        self.board_size = board_size
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        Args:
            state (np.array): Current board state (2D).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next board state (2D).
            done (bool): If the episode has ended.
        """
        self.buffer.append((state.clone(), action, reward, next_state.clone(), done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).float()
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv_block(x))

class DQN(nn.Module):
    def __init__(self, board_size, action_dim):
        super().__init__()
        self.board_size = board_size

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.AvgPool2d(kernel_size=2, stride=1),  # preserves output shape at 5x5
            ResidualBlock(64),
        )

        self.head = nn.Sequential(
            nn.Flatten(),  # from (B, 64, 5, 5) → (B, 1600)
            nn.Linear(64 * 6 * 6, 512), #pooling reduces wxh
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, action_dim)  # final layer: [B, 25]
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x

class HexDQNAgent(nn.Module):
    def __init__(self, board_size, replay_buffer_size=10_000):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size
        self.q_net = DQN(board_size, self.action_dim)
        self.target_net = DQN(board_size, self.action_dim)
        self.update_target()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device
        self.memory = ReplayBuffer(capacity=replay_buffer_size, board_size=board_size)

    def update_target(self, tau=1e-2):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, board, action_set):
        if random.random() < self.epsilon:
            return random.choice(action_set)

        state = self.convert_state_to_input(board)  # shape: (3, board_size, board_size)
        with torch.no_grad():
            q_values = self.q_net(state).squeeze()

        # Mask illegal actions
        masked_q = torch.full_like(q_values, -float('inf'))
        for a in action_set:
            a_index = a[0] * self.board_size + a[1] # convert [x,y] into [n]. 2d index into flattened 1d index
            masked_q[a_index] = q_values[a_index]

        best_action_index = masked_q.argmax().item()
        return divmod(best_action_index, self.board_size)  # return (x, y) tuple

    def store_transition(self, s, a, r, s_next, done):
        self.memory.push(s, a, r, s_next, done)

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Move batch to the model’s device
        states = self.convert_state_to_input(states)
        next_states = self.convert_state_to_input(next_states)

        device = next(self.parameters()).device
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1) # use double DQN target
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.update_target(tau=1e-3)

        return loss

    def convert_state_to_input(self, boards):
        # boards: shape (B, board_size, board_size)
        if len(boards.shape) == 3:
            # batch of boards
            player_stones = (boards == 1).float()
            opponent_stones = (boards == -1).float()
            legal_moves = (boards == 0).float()
            return torch.stack([player_stones, opponent_stones, legal_moves], dim=1)  # (B, 3, H, W)
        elif len(boards.shape) == 2:
            # single board
            player_stones = (boards == 1).float()
            opponent_stones = (boards == -1).float()
            legal_moves = (boards == 0).float()
            return torch.stack([player_stones, opponent_stones, legal_moves], dim=0).unsqueeze(0)  # (1, 3, H, W)
        else:
            raise ValueError(f"Unexpected board shape: {boards.shape}")
_agent_instance = None
def dqn_agent(board, action_set):
    global _agent_instance
    board_size = len(board)
    if _agent_instance is None:
       _agent_instance = HexDQNAgent(board_size)
    return _agent_instance.select_action(np.array(board), action_set)

def load_dqn_agent(path):
    model = HexDQNAgent(config.BOARD_SIZE)
    model.load_state_dict(torch.load(path))
    global _agent_instance
    _agent_instance = model
    if _agent_instance is not None:
        print("Successfully loaded the model from .pt file.")
