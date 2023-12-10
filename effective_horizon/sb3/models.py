from typing import List, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from effective_horizon.impala_cnn import ImpalaConvSequence


class ImpalaCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        conv_channels: List[int] = [16, 32, 32],
        use_bn: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, spaces.Box)
        obs_shape: Tuple[int, ...] = observation_space.shape
        assert len(obs_shape) == 3

        in_channels, width, height = obs_shape
        assert in_channels <= 4
        self.conv_sequences = nn.ModuleList()
        for out_channels in conv_channels:
            self.conv_sequences.append(
                ImpalaConvSequence(in_channels, out_channels, use_bn=use_bn)
            )
            in_channels = out_channels
            width, height = (width + 1) // 2, (height + 1) // 2

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * width * height, features_dim)

    def forward(self, observations: torch.Tensor):
        out = observations

        for conv_seq in self.conv_sequences:
            out = conv_seq(out)

        out = self.flatten(out)
        out = F.relu(out)
        out = self.fc(out)
        return out


class MiniGridCNN(BaseFeaturesExtractor):
    """
    Similar to NatureCNN but works on smaller MiniGrid observations.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear(self.cnn(observations)))
