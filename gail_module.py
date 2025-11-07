import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class ExpertDataset:
    """Load expert demonstrations formatted as obs(24)+action(8)+start+terminated."""

    def __init__(self, path: str, obs_dim: int, act_dim: int):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expert dataset not found: {path}")
        self.path = path
        structured = self._load_structured_if_available(path)
        if structured is not None and self._has_named_columns(structured):
            self._init_from_structured(structured, obs_dim, act_dim)
            return

        data = self._load_numeric_array(path)
        min_cols = obs_dim + act_dim
        if data.shape[1] < min_cols:
            raise ValueError(
                f"Expert data at {path} has {data.shape[1]} columns but at least {min_cols} are required."
            )
        self.obs = data[:, :obs_dim].astype(np.float32)
        self.actions = data[:, obs_dim : obs_dim + act_dim].astype(np.float32)
        extra = data[:, obs_dim + act_dim :]
        if extra.shape[1] >= 2:
            self.start_flags = extra[:, 0].astype(np.float32)
            self.terminated_flags = extra[:, 1].astype(np.float32)
        else:
            self.start_flags = None
            self.terminated_flags = None

    @staticmethod
    def _load_structured_if_available(path: str) -> Optional[np.ndarray]:
        try:
            data = np.genfromtxt(
                path,
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
        except Exception:
            return None
        if data is None:
            return None
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            return data
        return None

    @staticmethod
    def _load_numeric_array(path: str) -> np.ndarray:
        if path.endswith(".npz"):
            npz = np.load(path)
            if "data" in npz:
                data = npz["data"]
            elif "arr_0" in npz:
                data = npz["arr_0"]
            else:
                raise ValueError(f"NPZ file {path} does not contain 'data' or 'arr_0'.")
            return np.asarray(data, dtype=np.float32)

        if path.endswith(".npy"):
            return np.asarray(np.load(path), dtype=np.float32)

        loaders = (
            dict(delimiter=",", skiprows=1),
            dict(delimiter=",", skiprows=0),
            dict(delimiter="\t", skiprows=1),
            dict(delimiter="\t", skiprows=0),
        )
        last_exc: Optional[Exception] = None
        for kwargs in loaders:
            try:
                data = np.loadtxt(path, **kwargs)
            except Exception as exc:  # pragma: no cover - fallback path
                last_exc = exc
                continue
            else:
                data = np.asarray(data, dtype=np.float32)
                break
        else:
            raise ValueError(f"Unable to load expert data from {path}: {last_exc}")  # pragma: no cover

        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    @staticmethod
    def _has_named_columns(structured: np.ndarray) -> bool:
        names = structured.dtype.names
        if not names:
            return False
        required = {"qpos_0", "qvel_0", "qpos_12", "action_0"}
        return required.issubset(set(names))

    def _init_from_structured(self, structured: np.ndarray, obs_dim: int, act_dim: int) -> None:
        names = structured.dtype.names
        data = np.ma.getdata(structured)

        def _column(key: str) -> np.ndarray:
            if key not in names:
                raise ValueError(f"Missing column '{key}' in expert data {self.path}")
            col = np.asarray(data[key], dtype=np.float32)
            if col.ndim == 0:
                col = col.reshape(1)
            return col

        obs_columns = []
        # Expect qpos_0-7, qvel_0-7, qpos_12-19
        for idx in range(8):
            obs_columns.append(_column(f"qpos_{idx}"))
        for idx in range(8):
            obs_columns.append(_column(f"qvel_{idx}"))
        for idx in range(12, 20):
            obs_columns.append(_column(f"qpos_{idx}"))
        self.obs = np.stack(obs_columns, axis=1).astype(np.float32)

        action_columns = []
        for idx in range(act_dim):
            action_columns.append(_column(f"action_{idx}"))
        self.actions = np.stack(action_columns, axis=1).astype(np.float32)

        step = _column("step_num") if "step_num" in names else None
        terminated = _column("terminated") if "terminated" in names else None
        if step is not None:
            self.start_flags = (step == 0).astype(np.float32)
        else:
            self.start_flags = None
        if terminated is not None:
            self.terminated_flags = np.asarray(terminated, dtype=np.float32)
        else:
            self.terminated_flags = None

    def __len__(self) -> int:
        return self.obs.shape[0]

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Expert dataset is empty.")
        idx = np.random.randint(0, len(self), size=batch_size)
        return self.obs[idx], self.actions[idx]


class GAILDiscriminator(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        input_dim = obs_dim + act_dim
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.SiLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


class GAILModule:
    """Self-contained discriminator + training helper for minimal PPO integration."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        dataset_path: str,
        hidden_dims: Sequence[int],
        batch_size: int,
        iters_per_update: int,
        learning_rate: float,
        reward_scale: float,
        grad_penalty_coef: float,
    ):
        self.device = device
        self.dataset = ExpertDataset(dataset_path, obs_dim, act_dim)
        self.discriminator = GAILDiscriminator(obs_dim, act_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.batch_size = max(1, batch_size)
        self.iters_per_update = max(1, iters_per_update)
        self.reward_scale = reward_scale
        self.grad_penalty_coef = grad_penalty_coef
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.last_metrics: Dict[str, float] = {}

    def _prepare_policy_tensors(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return obs_tensor, act_tensor

    def _sample_expert_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, actions = self.dataset.sample(batch_size)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return obs_tensor, act_tensor

    def _gradient_penalty(
        self,
        expert_obs: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_obs: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(expert_obs.size(0), 1, device=self.device)
        interp_obs = (alpha * expert_obs + (1 - alpha) * policy_obs).requires_grad_(True)
        interp_actions = (alpha * expert_actions + (1 - alpha) * policy_actions).requires_grad_(True)
        logits = self.discriminator(interp_obs, interp_actions)
        grad_outputs = torch.ones_like(logits)
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=[interp_obs, interp_actions],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        penalty = 0.0
        for grad in gradients:
            penalty = penalty + (grad.view(grad.size(0), -1).norm(2, dim=1) - 1.0).pow(2)
        return penalty.mean()

    def update_and_reward(
        self, policy_obs: np.ndarray, policy_actions: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if policy_obs.size == 0:
            self.last_metrics = {"expert_loss": 0.0, "policy_loss": 0.0, "grad_penalty": 0.0}
            return np.zeros(0, dtype=np.float32), self.last_metrics.copy()

        pol_obs_tensor, pol_act_tensor = self._prepare_policy_tensors(policy_obs, policy_actions)
        policy_count = pol_obs_tensor.shape[0]
        effective_batch = min(self.batch_size, len(self.dataset), policy_count)
        if effective_batch <= 0:
            raise ValueError("Effective GAIL batch is zero. Check expert data and rollout length.")

        expert_loss_total = 0.0
        policy_loss_total = 0.0
        grad_penalty_total = 0.0

        for _ in range(self.iters_per_update):
            idx = np.random.randint(0, policy_count, size=effective_batch)
            pi_obs_batch = pol_obs_tensor[idx]
            pi_act_batch = pol_act_tensor[idx]
            exp_obs_batch, exp_act_batch = self._sample_expert_batch(effective_batch)

            logits_exp = self.discriminator(exp_obs_batch, exp_act_batch)
            logits_pi = self.discriminator(pi_obs_batch, pi_act_batch)

            expert_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_exp, torch.ones_like(logits_exp)
            )
            policy_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_pi, torch.zeros_like(logits_pi)
            )
            loss = expert_loss + policy_loss
            if self.grad_penalty_coef > 0.0:
                gp = self._gradient_penalty(exp_obs_batch, exp_act_batch, pi_obs_batch, pi_act_batch)
                loss = loss + self.grad_penalty_coef * gp
                grad_penalty_total += float(gp.detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            expert_loss_total += float(expert_loss.detach().cpu().item())
            policy_loss_total += float(policy_loss.detach().cpu().item())

        iterations = float(self.iters_per_update)
        metrics = {
            "expert_loss": expert_loss_total / iterations,
            "policy_loss": policy_loss_total / iterations,
            "grad_penalty": grad_penalty_total / iterations if self.grad_penalty_coef > 0.0 else 0.0,
        }
        self.last_metrics = metrics.copy()

        with torch.no_grad():
            logits = self.discriminator(pol_obs_tensor, pol_act_tensor)
            probs = torch.sigmoid(logits)
            rewards = -torch.log(torch.clamp(1.0 - probs, min=1e-8))
            rewards = rewards.squeeze(-1).cpu().numpy().astype(np.float32)
            rewards = rewards * self.reward_scale
        return rewards, metrics.copy()
