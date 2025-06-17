import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

from torch.xpu import device

import submission.config as config
from copy import deepcopy

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
    def __init__(self, channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class GameNet(nn.Module):
    """
    input_channels:
    -player stones
    -opponent stones
    -legal moves
    """
    def __init__(self, input_channels=3):
        super(GameNet, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(256)

        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(2 * config.BOARD_SIZE * config.BOARD_SIZE, 25)  # 5x5 board has 25 positions

        # Value head
        self.value_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_relu = nn.ReLU()
        self.value_fc = nn.Linear(2 * config.BOARD_SIZE * config.BOARD_SIZE, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        v = self.value_tanh(v)

        return p, v

class Agent(nn.Module):
    def __init__(self, replay_buffer_size=10_000, tau=0.005, lr=1e-4):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.board_size = config.BOARD_SIZE
        self.tau = tau

        self.policy_net = GameNet().to(self.device)
        self.target_net = deepcopy(self.policy_net).to(self.device)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, config.BOARD_SIZE)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, board, action_set):
        """
        Select an action given a board and set of legal actions.
        Uses the policy network to sample from the probability distribution.
        """
        self.policy_net.eval()
        with torch.no_grad():
            input_tensor = self.convert_state_to_input(board).unsqueeze(0).to(self.device)
            policy_logits, _ = self.policy_net(input_tensor)
            probs = torch.softmax(policy_logits, dim=1).squeeze(0)

            legal_indices = [a[0] * self.board_size + a[1] for a in action_set]
            legal_probs = probs[legal_indices]

            EPS = 1e-8
            if legal_probs.sum() < EPS:
                # fallback to uniform over legal moves
                if len(legal_indices) == 1:
                    # only one legal move
                    chosen_index = legal_indices[0]
                else:
                    chosen_index = legal_indices[torch.randint(len(legal_indices), (1,))]
                return divmod(chosen_index, self.board_size)

            legal_probs /= legal_probs.sum()
            chosen_index = torch.multinomial(legal_probs, 1).item()
            chosen_action = action_set[chosen_index]
        self.policy_net.train()
        return chosen_action

    def store_transition(self, s, a, r, s_next, done):
        """
        Stores a transition in the replay buffer.
        a: (i, j) tuple -> convert to flat index
        """
        self.replay_buffer.push(s, a, r, s_next, done)

    def update_target_net(self):
        """
        Soft update of the target network.
        """
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.stack([
            self.convert_state_to_input(board) for board in states
        ])
        next_states = torch.stack([
            self.convert_state_to_input(board) for board in next_states
        ])

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Current estimates
        policy_logits, values = self.policy_net(states)
        values = values.squeeze(1)

        # Target values
        with torch.no_grad():
            _, next_values = self.target_net(next_states)
            next_values = next_values.squeeze(1)
            target_values = rewards + (1 - dones) * next_values

        # Critic loss
        critic_loss = F.mse_loss(values, target_values)

        # Actor loss
        log_probs = F.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs[range(batch_size), actions]
        advantage = (target_values - values).detach()
        actor_loss = -(selected_log_probs * advantage).mean()

        # Total loss
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_net()

        return {
            "loss": total_loss.item(),
            "actor": actor_loss.mean().item(),
            "critic": critic_loss.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item()
        }

    def convert_state_to_input(self, board):
        """
        Converts a 2D board state to a 4-channel input tensor:
          - Channel 0: current player's stones (1s)
          - Channel 1: opponent's stones (1s)
          - Channel 2: legal moves (1s where empty)

        Assumes:
          - board is a 2D tensor of shape (board_size, board_size)
          - values: 1 for current player, -1 for opponent, 0 for empty
        """
        if len(board.shape) == 2:
            board_size = board.shape[0]
            player_plane = (board == 1).float()
            opponent_plane = (board == -1).float()
            legal_plane = (board == 0).float()
            player_turn = 1 if (board == 1).sum() <= (board == -1).sum() else 0

            input_tensor = torch.stack([player_plane, opponent_plane, legal_plane], dim=0)
            return input_tensor
        else:
            raise ValueError(f"Unexpected board shape: {board.shape}")