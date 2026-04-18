from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VacuumCNN(BaseFeaturesExtractor):
    """Small CNN for (3, H, W) vacuum environment observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(self.cnn(observations)))
