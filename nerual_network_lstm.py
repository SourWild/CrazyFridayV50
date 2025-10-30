from typing import Tuple

import torch
import torch.nn as nn


def mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    # Feedforward backbone / 前馈骨干网络
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
        # Recurrent encoder / 循环编码器
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        # Policy head / 策略头
        self.net = mlp(lstm_hidden_size, hidden_dims, act_dim)
        # Log-std parameter / 对数标准差参数
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode sequence / 编码观测序列
        _, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        # Mean action / 均值动作
        mean = self.net(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample(self, obs_seq: torch.Tensor):
        # Sample with Tanh-squash / 采样并压缩到[-1,1]
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
        # Evaluate log-prob for PPO / 评估 PPO 所需的对数概率
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
        # Shared encoder / 价值网络序列编码器
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        # Scalar value head / 标量价值头
        self.net = mlp(lstm_hidden_size, hidden_dims, 1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # Encode then regress value / 编码后回归状态价值
        _, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        return self.net(features)
