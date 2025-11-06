import math
import time
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningMeanStd:
    def __init__(self, shape):
        # Track running mean/variance for normalization / 跟踪归一化所需的滑动均值方差
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot_count

    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


@dataclass
class PPOConfig:
    # total_timesteps: int = 600_000
    iterations: int = 500
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

        # Online normalizers keep observation/reward scales in check / 在线归一化器用于稳定观测与奖励尺度
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=(1,))

        # Rollout buffers capture sequence inputs for RNN/Transformer / 采样缓存保存序列输入供循环或注意力网络使用
        self.obs_buf = np.zeros((config.rollout_steps, config.sequence_length, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((config.rollout_steps, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.dones_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.values_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.logprobs_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.running_ep_reward = 0.0
        self.completed_ep_rewards = []
        self.running_ep_reward_separate = np.array([0, 0, 0, 0, 0, 0])
        self.completed_ep_rewards_separate = []   
        self.completed_success = []
        self.completed_failure = []
        self.ep_num = 0
        
        
        self.obs_window: Optional[Deque[np.ndarray]] = None

    def _init_obs_window(self, norm_obs: np.ndarray) -> None:
        norm_obs = norm_obs.astype(np.float32)
        self.obs_window = deque([norm_obs.copy() for _ in range(self.cfg.sequence_length)], maxlen=self.cfg.sequence_length)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        self.obs_rms.update(obs[None, :])
        norm = (obs - self.obs_rms.mean) / self.obs_rms.std()
        return np.clip(norm, -10.0, 10.0).astype(np.float32)

    def _normalize_reward(self, reward: float) -> float:
        reward = float(reward)
        self.reward_rms.update(np.array([[reward]], dtype=np.float32))
        scale = float(self.reward_rms.std().squeeze())
        if scale < 1e-6:
            return reward
        return reward / scale


    def collect_rollout(self, start_obs: np.ndarray) -> Tuple[np.ndarray, bool, bool, bool]:
        # Core sampling loop: build sequence, act, and push transitions / 采样主循环：构造序列、执行动作并存储转移
        obs = start_obs.astype(np.float32)
        if self.obs_window is None:
            self._init_obs_window(obs)
        last_done = False
        last_terminated = False
        last_truncated = False
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

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            last_done = last_terminated or last_truncated

            self.obs_buf[step] = obs_seq
            self.actions_buf[step] = action
            norm_reward = self._normalize_reward(reward)
            self.rewards_buf[step] = norm_reward
            self.dones_buf[step] = float(last_done)
            self.values_buf[step] = value
            self.logprobs_buf[step] = log_prob
            self.running_ep_reward += reward
            self.running_ep_reward_separate = self.running_ep_reward_separate + info["R_separate"]

            next_obs = next_obs.astype(np.float32)
            norm_next_obs = self._normalize_obs(next_obs)
            self.obs_window.append(norm_next_obs)
            obs = norm_next_obs
            last_terminated_step = 0
            if last_terminated:
                # 只有双方得分导致的episode结束时才更新reward
                self.completed_ep_rewards.append(self.running_ep_reward)
                self.running_ep_reward = 0.0
                if self.running_ep_reward_separate[1]:
                    self.completed_success.append(1)
                elif self.running_ep_reward_separate[2]:
                    self.completed_failure.append(1)
                self.completed_ep_rewards_separate.append(self.running_ep_reward_separate)
                self.running_ep_reward_separate = np.array([0, 0, 0, 0, 0, 0])
                self.ep_num += 1
                
            if last_done:   
                obs, _ = self.env.reset()
                obs = obs.astype(np.float32)
                norm_reset_obs = self._normalize_obs(obs)
                self._init_obs_window(norm_reset_obs)
                obs = norm_reset_obs
                                      
        return obs, last_done

    def compute_gae(self, next_value: np.ndarray, last_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        # Generalized Advantage Estimation / 广义优势估计
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
        # Training loop: rollout → advantage/return → PPO update / 训练循环：采样 → 计算优势回报 → 执行 PPO 更新
        obs, _ = self.env.reset(seed=self.cfg.seed)
        obs = obs.astype(np.float32)
        norm_obs = self._normalize_obs(obs)
        self._init_obs_window(norm_obs)
        obs = norm_obs
        # total_updates = math.ceil(self.cfg.total_timesteps / self.cfg.rollout_steps)
        total_updates = self.cfg.iterations
        start_time = time.time()
        
        for update in range(1, total_updates + 1):
            iteration_start_time = time.time()
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

            time_elapsed = time.time() - start_time
            iteration_elapsed = time.time() - iteration_start_time
            fps = self.cfg.rollout_steps / iteration_elapsed
                     
            
            ep_rew_mean = (
                np.mean(self.completed_ep_rewards[-10:]) if self.completed_ep_rewards else 0.0
            )
            success_rate = (
                np.mean(self.completed_success[-10:]) if self.completed_success else 0.0
            )
            failuer_rate = (
                np.mean(self.completed_failure[-10:]) if self.completed_failure else 0.0
            )
            
            arr = np.array(self.completed_ep_rewards_separate, dtype=np.float32)
            if arr.size > 0:
                ep_rew_mean_separate = np.mean(arr[-10:], axis=0)
            else:
                ep_rew_mean_separate = np.zeros(6, dtype=np.float32)                
            
            if self.ep_num == 0:
                ep_len_mean = 0
            else: 
                ep_len_mean = update * self.cfg.rollout_steps / self.ep_num
            
            print(
                f"----------------------------------------\n"
                f"rollout：\n"
                f"ep_num {self.ep_num} | \n"
                f"ep_len_mean {ep_len_mean} | \n"
                f"ep_rew_mean {ep_rew_mean:.2f} | \n"
                f"success_rate {success_rate*100:.2f}%  | \n"
                f"failuer_rate {failuer_rate*100:.2f}% | \n"
                f"ep_rew_mean_dist: {ep_rew_mean_separate[0]:.2f} |\n"
                f"ep_rew_mean_success: {ep_rew_mean_separate[1]:.2f} |\n"
                f"ep_rew_mean_failure: {ep_rew_mean_separate[2]:.2f} |\n"
                f"ep_rew_mean_collision: {ep_rew_mean_separate[3]:.2f} |\n"
                f"ep_rew_mean_action: {ep_rew_mean_separate[4]:.2f} |\n"
                f"ep_rew_mean_step: {ep_rew_mean_separate[5]:.2f} |\n"
                
                
                f"----------------------------------------\n"
                f"time：\n"
                f"fps {fps:.2f} | \n"
                f"iterations {update}/{total_updates} | \n"
                f"time_elapsed {time_elapsed:.2f}s | \n"
                f"total_timesteps {update * self.cfg.rollout_steps} | \n"
                
                
                f"----------------------------------------\n"
                f"train：\n"
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
