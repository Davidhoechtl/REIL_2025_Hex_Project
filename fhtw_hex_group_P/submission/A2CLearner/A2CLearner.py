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

class ActorCriticNet(nn.Module):
    def __init__(self, board_size, action_dim):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.AvgPool2d(kernel_size=2, stride=1),
            ResidualBlock(64),
        )

        self.flatten = nn.Flatten()
        feature_dim = 64 * (config.BOARD_SIZE-1) * (config.BOARD_SIZE-1)  # Adjust according to pooling and board size

        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)  # Output logits for all actions
        )

        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # New hidden layer with 256 units
            nn.ReLU(),
            nn.Linear(256, 1)  # Output scalar value V(s)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.flatten(x)
        policy_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)  # Shape: (B,)
        return policy_logits, value

class A2CAgent(nn.Module):
    def __init__(self, board_size, replay_buffer_size=10_000, tau=0.005):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size
        self.net = ActorCriticNet(board_size, self.action_dim)
        self.target_net = ActorCriticNet(board_size, self.action_dim)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)
        self.gamma = 0.99
        self.tau = tau
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)
        self.target_net.to(self.device)

        self.memory = ReplayBuffer(replay_buffer_size, board_size)

    def select_action(self, board, action_set):
        assert len(action_set) > 0, "action_set is empty!"
        state = self.convert_state_to_input(board).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(state)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        mask = torch.zeros_like(probs, dtype=torch.bool)
        indices = [a[0] * self.board_size + a[1] for a in action_set]
        mask[indices] = True

        masked_probs = probs * mask.float()
        sum_probs = masked_probs.sum().item()

        EPS = 1e-8
        if sum_probs < EPS:
            # fallback to uniform over legal moves
            legal_indices = torch.nonzero(mask).squeeze()
            if legal_indices.dim() == 0:
                # only one legal move
                chosen_index = legal_indices.item()
            else:
                chosen_index = legal_indices[torch.randint(len(legal_indices), (1,))].item()
            return divmod(chosen_index, self.board_size)

        masked_probs /= sum_probs
        assert torch.isfinite(masked_probs).all(), "masked_probs contains NaN or Inf after normalization"

        action_index = torch.multinomial(masked_probs, 1).item()
        return divmod(action_index, self.board_size)

    def store_transition(self, s, a, r, s_next, done):
        self.memory.push(s, a, r, s_next, done)

    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = self.convert_state_to_input(states).to(self.device)
        next_states = self.convert_state_to_input(next_states).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        logits, state_values = self.net(states)
        with torch.no_grad():
            _, next_state_values = self.target_net(next_states)
            next_state_values = next_state_values.detach()
            targets = rewards + self.gamma * next_state_values * (1 - dones)

        advantage_raw  = targets - state_values
        advantage = (advantage_raw  - advantage_raw .mean()) / (advantage_raw .std() + 1e-8)

        # Log prob of chosen action
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Entropy loss (to encourage exploration)
        entropy_loss = -(probs * log_probs).sum(dim=1).mean() * 0.01
        actor_loss = -selected_log_probs * advantage.detach()
        critic_loss = advantage_raw.pow(2)
        loss = actor_loss.mean() + 0.5 * critic_loss.mean() - entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.update_target_net()
        return {
            "loss": loss.item(),
            "actor": actor_loss.mean().item(),
            "critic": critic_loss.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item()
        }

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