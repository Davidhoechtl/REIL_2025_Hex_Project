import torch.nn as nn
import torch.nn.functional as F
import submission.config as config

class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
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
    def __init__(self, input_channels=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(128)

        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(3)])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(2 * config.BOARD_SIZE * config.BOARD_SIZE, config.BOARD_SIZE * config.BOARD_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
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
