import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

from torch.xpu import device

import submission.config as config
from copy import deepcopy
from submission.LastTry.GameNetBig import GameNet as GameNetBig
from submission.LastTry.GameNetSmall import GameNet as GameNetSmall
from submission.LastTry.GameNetMedium import GameNet as GameNetMedium


class Agent(nn.Module):
    def __init__(self, board_size, total_steps = 100, lr=0.0005 ):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.board_size = board_size
        self.policy_net = GameNetMedium().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-5)
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.critic_coef = 0.8

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

    def train_step(self,  states, actions, rewards, next_states, dones, batch_size):

        batch_size = states.shape[0]
        states = torch.stack([ self.convert_state_to_input(board) for board in states ]).to(self.device)
        next_states = torch.stack([ self.convert_state_to_input(board) for board in next_states ]).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Current estimates
        policy_logits, values = self.policy_net(states)
        values = values.squeeze(1)

        # Target values
        with torch.no_grad():
            _, next_values = self.policy_net(next_states)
            next_values = next_values.squeeze(1)
            target_values = rewards + self.gamma * next_values * (1 - dones)

        # Critic loss
        critic_loss = F.mse_loss(values, target_values)

        # Actor loss
        log_probs = F.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs[range(batch_size), actions]
        advantage_raw = (target_values - values).detach()
        #advantage = (advantage_raw - advantage_raw.mean()) / (advantage_raw.std() + 1e-8)
        actor_loss = -(selected_log_probs * advantage_raw).mean()

        # entropy loss
        probs = torch.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()
        entropy_loss = entropy * self.entropy_coef

        # Total loss
        total_loss = actor_loss + self.critic_coef * critic_loss - entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        return {
            "loss": total_loss.item(),
            "actor": actor_loss.mean().item(),
            "critic": critic_loss.mean().item(),
            "advantage_mean": advantage_raw.mean().item(),
            "advantage_std": advantage_raw.std().item(),
            "mean_reward": rewards.mean().item()
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