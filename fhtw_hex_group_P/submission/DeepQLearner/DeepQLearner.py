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

        states = torch.stack(states).view(len(states), -1).float()
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states).view(len(next_states), -1).float()
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, board_size, action_dim):
        super().__init__()
        self.board_size = board_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,1,5,5) -> (B,32,5,5)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B,64,5,5)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (B,64,5,5)
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, 25)
        x = x.view(-1, 1, self.board_size, self.board_size)  # (B,1,5,5)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class HexDQNAgent(nn.Module):
    def __init__(self, board_size, replay_buffer_size=10_000):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size
        self.q_net = DQN(board_size, self.action_dim)
        self.target_net = DQN(board_size, self.action_dim)
        self.update_target()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
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

        state = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
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
        device = next(self.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
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
