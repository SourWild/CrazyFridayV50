from typing import Tuple

import torch
import torch.nn as nn


def mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...],
        lstm_hidden_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.net = mlp(lstm_hidden_size, hidden_dims, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        mean = self.net(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample(self, obs_seq: torch.Tensor):
        mean, log_std = self.forward(obs_seq)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        noise = normal.rsample()
        action = torch.tanh(noise)
        log_prob = (
            normal.log_prob(noise) - torch.log(1 - action.pow(2) + 1e-7)
        ).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate(self, obs_seq: torch.Tensor, actions: torch.Tensor):
        mean, log_std = self.forward(obs_seq)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        unsquashed = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
        log_prob = (
            normal.log_prob(unsquashed) - torch.log(1 - actions.pow(2) + 1e-7)
        ).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...],
        lstm_hidden_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.net = mlp(lstm_hidden_size, hidden_dims, 1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        return self.net(features)
