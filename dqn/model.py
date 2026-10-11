"""Q-network from Section 4.1: one forward pass gives Q(s, a) for ALL actions."""
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4), nn.ReLU(),   # 84 -> 20
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),            # 20 -> 9
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, n_actions),                                        # linear output, 1 per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: float (B,4,84,84) in [0,1]
        return self.net(x)
