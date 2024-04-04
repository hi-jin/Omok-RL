import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import gymnasium as gym


class OmokFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(OmokFeatureExtractor, self).__init__()
        board_size = observation_space.shape[0]
        self.board_shape = (board_size, board_size)
        self.features_dim = features_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)  # (32, board_size - 4, board_size - 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # (64, board_size - 6, board_size - 6)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # (128, board_size - 8, board_size - 8)
        self.linear = nn.Linear(128 * (board_size - 8) * (board_size - 8), features_dim)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = rearrange(x, "h w -> () () h w")
        elif len(x.shape) == 3:
            x = rearrange(x, "b h w -> b () h w")

        assert (
            len(x.shape) == 4 and x.shape[-2] == x.shape[-1]
        ), f"board shape should be (batch_size, channel_size, board_size, board_size), but got {x.shape}"

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = rearrange(x, "b c h w -> b (c h w)")
        x = F.relu(self.linear(x))

        return x
