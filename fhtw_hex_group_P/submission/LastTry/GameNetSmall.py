import torch.nn as nn
import torch.nn.functional as F
import submission.config as config

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class GameNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)

        self.res_block = ResidualBlock(64)  # just one block

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_fc = nn.Linear(2 * config.BOARD_SIZE * config.BOARD_SIZE, config.BOARD_SIZE * config.BOARD_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_fc = nn.Linear(config.BOARD_SIZE * config.BOARD_SIZE, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.res_block(x)

        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        v = self.value_tanh(v)

        return p, v
