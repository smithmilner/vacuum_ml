from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VacuumMultiInputExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observation space.

    CNN branch:    (2, 84, 84) map  → 256-dim
    Sensor branch: (6,) sensors     → 64-dim
    Concat output: 320-dim
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 320):
        super().__init__(observation_space, features_dim)

        map_space = observation_space["map"]      # Box(2, 84, 84)
        sensor_space = observation_space["sensors"]  # Box(6,)
        n_channels = map_space.shape[0]           # 2

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1),  # 84→42
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # 42→21
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 21→11
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(map_space.sample()[None]).float()
            cnn_out_dim = self.cnn(sample).shape[1]  # 64 * 11 * 11 = 7744

        self.cnn_linear = nn.Linear(cnn_out_dim, 256)

        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        map_obs = observations["map"].float()
        sensor_obs = observations["sensors"].float()
        cnn_out = torch.relu(self.cnn_linear(self.cnn(map_obs)))
        mlp_out = self.sensor_mlp(sensor_obs)
        return torch.cat([cnn_out, mlp_out], dim=1)
