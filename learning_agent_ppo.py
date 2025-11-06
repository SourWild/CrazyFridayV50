import math
import os
import time
from dataclasses import dataclass, asdict
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obstacle_env import ObstacleEnv


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
    checkpoint_interval: int = 10
    checkpoint_root: str = "checkpoints"


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
        self.last_obs: Optional[np.ndarray] = None
        self.last_done: bool = False
        self.checkpoint_interval = max(0, config.checkpoint_interval)
        self.checkpoint_root = os.path.abspath(config.checkpoint_root)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = os.path.join(self.checkpoint_root, self.run_timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.last_checkpoint_path: Optional[str] = None
        self.start_update_idx: int = 0
        self.has_loaded_checkpoint: bool = False

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

    def _get_rms_state(self, rms: RunningMeanStd) -> Dict[str, Any]:
        return {
            "mean": rms.mean.copy(),
            "var": rms.var.copy(),
            "count": rms.count,
        }

    def _get_rng_state(self) -> Dict[str, Any]:
        rng_state: Dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        env_rng = getattr(self.env, "np_random", None)
        if env_rng is not None:
            if hasattr(env_rng, "bit_generator"):
                rng_state["env"] = env_rng.bit_generator.state
            elif hasattr(env_rng, "get_state"):
                rng_state["env"] = env_rng.get_state()
        return rng_state

    def _get_env_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        data = getattr(self.env, "data", None)
        if data is not None:
            for attr in ("qpos", "qvel", "ctrl"):
                if hasattr(data, attr):
                    try:
                        state[attr] = np.array(getattr(data, attr)).copy()
                    except Exception:
                        pass
            if hasattr(data, "time"):
                state["time"] = float(data.time)
        state["episode_step"] = getattr(self.env, "episode_step", None)
        state["max_episode_steps"] = getattr(self.env, "max_episode_steps", None)
        state["obstacle_mode"] = getattr(self.env, "obstacle_mode", None)
        return state

    def _apply_env_state(self, env_state: Optional[Dict[str, Any]]) -> None:
        if not env_state:
            return
        data = getattr(self.env, "data", None)
        if data is None:
            return
        for attr in ("qpos", "qvel", "ctrl"):
            if attr in env_state and hasattr(data, attr):
                try:
                    np.copyto(getattr(data, attr), np.array(env_state[attr], dtype=np.float64))
                except Exception:
                    pass
        if "time" in env_state:
            data.time = float(env_state["time"])
        if "episode_step" in env_state and hasattr(self.env, "episode_step"):
            try:
                self.env.episode_step = int(env_state["episode_step"])
            except Exception:
                pass
        if "max_episode_steps" in env_state and hasattr(self.env, "max_episode_steps"):
            try:
                self.env.max_episode_steps = int(env_state["max_episode_steps"])
            except Exception:
                pass
        if "obstacle_mode" in env_state and env_state["obstacle_mode"] is not None:
            try:
                self.env.obstacle_mode = env_state["obstacle_mode"]
            except Exception:
                pass
        try:
            mujoco.mj_forward(self.env.model, data)
        except Exception:
            pass

    def save_checkpoint(self, update_idx: int, total_timesteps: int) -> str:
        if self.obs_window is not None:
            obs_window = [np.array(item).copy() for item in self.obs_window]
        else:
            obs_window = None

        checkpoint: Dict[str, Any] = {
            "update_idx": update_idx,
            "total_timesteps": total_timesteps,
            "config": asdict(self.cfg),
            "run_timestamp": self.run_timestamp,
            "policy_state": self.policy.state_dict(),
            "value_state": self.value.state_dict(),
            "policy_optimizer_state": self.policy_optim.state_dict(),
            "value_optimizer_state": self.value_optim.state_dict(),
            "obs_rms": self._get_rms_state(self.obs_rms),
            "reward_rms": self._get_rms_state(self.reward_rms),
            "running_ep_reward": self.running_ep_reward,
            "running_ep_reward_separate": self.running_ep_reward_separate.copy(),
            "completed_ep_rewards": list(self.completed_ep_rewards),
            "completed_ep_rewards_separate": [
                np.array(arr).copy() for arr in self.completed_ep_rewards_separate
            ],
            "completed_success": list(self.completed_success),
            "completed_failure": list(self.completed_failure),
            "ep_num": self.ep_num,
            "obs_window": obs_window,
            "last_obs": None if self.last_obs is None else np.array(self.last_obs).copy(),
            "last_done": self.last_done,
            "rng_state": self._get_rng_state(),
            "env_state": self._get_env_state(),
        }

        filename = f"update_{update_idx:05d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + ".tmp"

        try:
            torch.save(checkpoint, tmp_path)
        except RuntimeError as exc:
            if "PytorchStreamWriter" in str(exc):
                torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
            else:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

        os.replace(tmp_path, path)
        self.last_checkpoint_path = path
        print(f"[checkpoint] Saved update {update_idx} to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint_path = os.path.abspath(path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        saved_cfg = checkpoint.get("config")
        if isinstance(saved_cfg, dict):
            for key, value in saved_cfg.items():
                if not hasattr(self.cfg, key):
                    continue
                if key == "iterations":
                    try:
                        self.cfg.iterations = max(int(value), int(self.cfg.iterations))
                    except Exception:
                        self.cfg.iterations = int(self.cfg.iterations)
                    continue
                setattr(self.cfg, key, value)

        self.policy.load_state_dict(checkpoint["policy_state"])
        self.value.load_state_dict(checkpoint["value_state"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state"])
        self.value_optim.load_state_dict(checkpoint["value_optimizer_state"])

        obs_rms_state = checkpoint.get("obs_rms")
        if obs_rms_state:
            np.copyto(self.obs_rms.mean, np.array(obs_rms_state.get("mean"), dtype=np.float64))
            np.copyto(self.obs_rms.var, np.array(obs_rms_state.get("var"), dtype=np.float64))
            self.obs_rms.count = float(obs_rms_state.get("count", self.obs_rms.count))

        reward_rms_state = checkpoint.get("reward_rms")
        if reward_rms_state:
            np.copyto(self.reward_rms.mean, np.array(reward_rms_state.get("mean"), dtype=np.float64))
            np.copyto(self.reward_rms.var, np.array(reward_rms_state.get("var"), dtype=np.float64))
            self.reward_rms.count = float(reward_rms_state.get("count", self.reward_rms.count))

        self.running_ep_reward = float(checkpoint.get("running_ep_reward", 0.0))
        self.running_ep_reward_separate = np.array(
            checkpoint.get("running_ep_reward_separate", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        self.completed_ep_rewards = list(checkpoint.get("completed_ep_rewards", []))
        self.completed_ep_rewards_separate = [
            np.array(arr, dtype=np.float32)
            for arr in checkpoint.get("completed_ep_rewards_separate", [])
        ]
        self.completed_success = list(checkpoint.get("completed_success", []))
        self.completed_failure = list(checkpoint.get("completed_failure", []))
        self.ep_num = int(checkpoint.get("ep_num", 0))

        obs_window_state = checkpoint.get("obs_window")
        if obs_window_state is not None:
            self.obs_window = deque(
                [np.array(item, dtype=np.float32) for item in obs_window_state],
                maxlen=self.cfg.sequence_length,
            )
        else:
            self.obs_window = None

        last_obs_state = checkpoint.get("last_obs")
        self.last_obs = None if last_obs_state is None else np.array(last_obs_state, dtype=np.float32)
        self.last_done = bool(checkpoint.get("last_done", False))
        if self.obs_window is None and self.last_obs is not None:
            self._init_obs_window(self.last_obs.astype(np.float32))

        rng_state = checkpoint.get("rng_state", {})
        torch_state = rng_state.get("torch")
        if torch_state is not None:
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state)
        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [
                    cs if isinstance(cs, torch.ByteTensor) else torch.tensor(cs, dtype=torch.uint8)
                    for cs in cuda_state
                ]
            elif not isinstance(cuda_state, torch.ByteTensor):
                cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
            torch.cuda.set_rng_state_all(cuda_state)
        env_state_rng = rng_state.get("env")
        env_rng = getattr(self.env, "np_random", None)
        if env_rng is not None and env_state_rng is not None:
            if hasattr(env_rng, "bit_generator"):
                env_rng.bit_generator.state = env_state_rng
            elif hasattr(env_rng, "set_state"):
                env_rng.set_state(env_state_rng)

        env_state = checkpoint.get("env_state")
        self._apply_env_state(env_state)

        self.start_update_idx = int(checkpoint.get("update_idx", 0))
        total_timesteps = checkpoint.get("total_timesteps")
        if total_timesteps is not None:
            try:
                self.start_update_idx = max(self.start_update_idx, int(total_timesteps // self.cfg.rollout_steps))
            except Exception:
                pass

        checkpoint_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_root = parent_dir if parent_dir else self.checkpoint_root
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.run_timestamp = checkpoint.get("run_timestamp", os.path.basename(self.checkpoint_dir))
        self.last_checkpoint_path = checkpoint_path
        self.has_loaded_checkpoint = True
        print(f"[checkpoint] Loaded state from {checkpoint_path}; resuming at update {self.start_update_idx}.")

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
        if self.has_loaded_checkpoint and self.last_obs is not None and self.obs_window is not None:
            obs = self.last_obs.astype(np.float32)
            if len(self.obs_window) != self.cfg.sequence_length:
                self._init_obs_window(obs)
        else:
            raw_obs, _ = self.env.reset(seed=self.cfg.seed)
            raw_obs = raw_obs.astype(np.float32)
            norm_obs = self._normalize_obs(raw_obs)
            self._init_obs_window(norm_obs)
            obs = norm_obs
            self.last_obs = obs.copy()
            self.last_done = False

        total_updates = self.cfg.iterations
        start_time = time.time()

        start_update = self.start_update_idx
        if start_update >= total_updates:
            print(f"[checkpoint] Requested iterations ({total_updates}) already completed; nothing to train.")
            return

        for update in range(start_update + 1, total_updates + 1):
            iteration_start_time = time.time()
            obs, last_done = self.collect_rollout(obs)
            self.last_obs = obs.copy()
            self.last_done = last_done

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
            total_timesteps = update * self.cfg.rollout_steps

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
                f"total_timesteps {total_timesteps} | \n"
                f"----------------------------------------\n"
                f"train：\n"
            )

            if self.checkpoint_interval > 0 and update % self.checkpoint_interval == 0:
                self.save_checkpoint(update, total_timesteps)

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
