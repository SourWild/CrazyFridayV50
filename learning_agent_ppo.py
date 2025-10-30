import math
import time
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    total_timesteps: int = 600_000
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: Tuple[int, ...] = (256,)
    sequence_length: int = 10
    lstm_hidden_size: int = 256
    transformer_embed_dim: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    seed: int = 42

class PPOTrainer:
    def __init__(
        self,
        env,
        config: PPOConfig,
        PolicyNetwork,
        ValueNetwork,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        value_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.cfg = config
        policy_kwargs = policy_kwargs or {}
        value_kwargs = value_kwargs or {}

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, act_dim, config.hidden_dims, **policy_kwargs).to(self.device)
        self.value = ValueNetwork(obs_dim, config.hidden_dims, **value_kwargs).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_optim = optim.Adam(self.value.parameters(), lr=config.learning_rate)

        self.obs_buf = np.zeros((config.rollout_steps, config.sequence_length, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((config.rollout_steps, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.dones_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.values_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.logprobs_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.running_ep_reward = 0.0
        self.completed_ep_rewards = []
        self.obs_window: Optional[Deque[np.ndarray]] = None

    def _init_obs_window(self, obs: np.ndarray) -> None:
        obs = obs.astype(np.float32)
        self.obs_window = deque([obs.copy() for _ in range(self.cfg.sequence_length)], maxlen=self.cfg.sequence_length)


    def collect_rollout(self, start_obs: np.ndarray) -> Tuple[np.ndarray, bool]:
        obs = start_obs.astype(np.float32)
        if self.obs_window is None:
            self._init_obs_window(obs)
        done = False
        for step in range(self.cfg.rollout_steps):
            obs_seq = np.stack(self.obs_window, axis=0)
            obs_tensor = torch.as_tensor(
                obs_seq, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _ = self.policy.sample(obs_tensor)
                value_tensor = self.value(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            log_prob = float(log_prob_tensor.cpu().item())
            value = float(value_tensor.cpu().item())

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.obs_buf[step] = obs_seq
            self.actions_buf[step] = action
            self.rewards_buf[step] = reward
            self.dones_buf[step] = float(done)
            self.values_buf[step] = value
            self.logprobs_buf[step] = log_prob
            self.running_ep_reward += reward

            next_obs = next_obs.astype(np.float32)
            self.obs_window.append(next_obs)
            obs = next_obs
            if done:
                self.completed_ep_rewards.append(self.running_ep_reward)
                self.running_ep_reward = 0.0
                obs, _ = self.env.reset()
                obs = obs.astype(np.float32)
                self._init_obs_window(obs)
        return obs, done

    def compute_gae(self, next_value: np.ndarray, last_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(self.rewards_buf)
        lastgaelam = 0.0
        for step in reversed(range(self.cfg.rollout_steps)):
            if step == self.cfg.rollout_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones_buf[step + 1]
                next_values = self.values_buf[step + 1]
            delta = (
                self.rewards_buf[step]
                + self.cfg.gamma * next_values * next_non_terminal
                - self.values_buf[step]
            )
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * lastgaelam
            advantages[step] = lastgaelam
        returns = advantages + self.values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, advantages: np.ndarray, returns: np.ndarray) -> None:
        obs_tensor = torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(self.actions_buf, dtype=torch.float32, device=self.device)
        old_logprobs_tensor = torch.as_tensor(self.logprobs_buf, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = self.cfg.rollout_steps
        inds = np.arange(batch_size)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = inds[start:end]

                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_old_logprobs = old_logprobs_tensor[mb_inds]
                mb_advantages = advantages_tensor[mb_inds]
                mb_returns = returns_tensor[mb_inds]

                new_logprobs, entropy = self.policy.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs.squeeze(-1) - mb_old_logprobs)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_estimates = self.value(mb_obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(value_estimates, mb_returns)

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
                self.policy_optim.step()
                self.value_optim.step()

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        obs = obs.astype(np.float32)
        self._init_obs_window(obs)
        total_updates = math.ceil(self.cfg.total_timesteps / self.cfg.rollout_steps)
        for update in range(1, total_updates + 1):
            start_time = time.time()
            obs, last_done = self.collect_rollout(obs)
            with torch.no_grad():
                if last_done:
                    next_value = 0.0
                else:
                    obs_seq = np.stack(self.obs_window, axis=0)
                    obs_tensor = torch.as_tensor(
                        obs_seq, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    next_value = self.value(obs_tensor).cpu().numpy().squeeze(0)
            advantages, returns = self.compute_gae(next_value, last_done)
            self.update(advantages, returns)

            elapsed = time.time() - start_time
            avg_reward = (
                np.mean(self.completed_ep_rewards[-10:]) if self.completed_ep_rewards else 0.0
            )
            print(
                f"Update {update}/{total_updates} | "
                f"Steps {update * self.cfg.rollout_steps} | "
                f"AvgReward {avg_reward:.2f} | "
                f"Time {elapsed:.2f}s"
            )
        self.env.close()


def main() -> None:
    config = PPOConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = ObstacleEnv(xml_path=xml_path, obstacle_mode="periodic", render_mode=None)
    trainer = PPOTrainer(env, config)
    trainer.train()


if __name__ == "__main__":
    main()
