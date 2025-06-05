import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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
        self.model = nn.Sequential(
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class HexDQNAgent(nn.Module):
    def __init__(self, board_size, replay_buffer = None):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size
        self.q_net = DQN(board_size, self.action_dim)
        self.target_net = DQN(board_size, self.action_dim)
        self.update_target()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = replay_buffer
        if replay_buffer is None:
            self.memory = ReplayBuffer(capacity=10000, board_size=board_size)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, board, action_set):
        if random.random() < self.epsilon:
            return random.choice(action_set)

        state = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state).squeeze()

        # Mask illegal actions
        masked_q = torch.full_like(q_values, -float('inf'))
        for a in action_set:
            masked_q[a] = q_values[a]
        return masked_q.argmax().item()

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
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

_agent_instance = None
def dqn_agent(board, action_set):
    global _agent_instance
    board_size = len(board)
    if _agent_instance is None:
       _agent_instance = HexDQNAgent(board_size)
    return _agent_instance.select_action(np.array(board), action_set)
