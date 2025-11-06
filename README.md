# Group4_fencing_NN_submit – MuJoCo Fencing PPO Project

This repository contains our custom MuJoCo fencing environment and a from-scratch PPO trainer capable of running LSTM- or Transformer-based policies. The instructions below assume no prior knowledge of the codebase and walk through environment setup, training, evaluation, and debugging utilities.

---

## 1. Prerequisites

- Linux with Anaconda/Miniconda installed (CUDA-capable GPU optional; project auto-falls back to CPU).
- MuJoCo 3 assets are bundled under `Fencing_agent&obstacle_description/` – no external download required.

---

## 2. Environment Setup

```bash
cd Group4_fencing_NN_submit
# If the drl_mujoco environment already exists from previous submission:
conda env update --name drl_mujoco --file environment.yml

# Otherwise create it for the first time:
conda env create -f environment.yml

conda activate drl_mujoco
```

The provided environment installs MuJoCo, Gymnasium, PyTorch 2.8 (CUDA 12.x build), and utility packages listed in `environment.yml`.

---

## 3. Project Layout

| Path | Description |
| --- | --- |
| `main.py` | CLI entry point for training and evaluation. |
| `learning_agent_ppo.py` | PPO trainer with observation/reward normalization, GAE, and manual rollout buffer. |
| `nerual_network_lstm.py` | LSTM-based policy/value heads with LayerNorm + SiLU feature blocks. |
| `nerual_network_trans.py` | Transformer alternative (encoder stack + feature MLP). |
| `obstacle_env.py` | Custom Gymnasium environment with MuJoCo control, termination, and reward shaping. |
| `debug_nn_io.py` | Standalone script to inspect NN forward/backward tensor statistics. |
| `Fencing_agent&obstacle_description/` | MuJoCo XML and mesh assets for the fencing agent and obstacle. |

---

## 4. Basic Usage – Train a Policy

```bash
python main.py \
  --model-type lstm \
  --obstacle-mode periodic
```

Key CLI options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model-type` | `lstm` | Choose `lstm` or `trans` policy/value architecture. |
| `--obstacle-mode` | `periodic` | Environment obstacle controller (`static`, `none`, `periodic`, `reactive`). |
| `--total-timesteps` | 600000 | Override PPO total interaction steps. |
| `--seed` | 42 | Reproducible RNG seed. |

Training automatically:
- Normalizes observations and rewards online.
- Maintains a 10-step sliding window for sequence models.
- Prints per-update reward summaries (`AvgReward-R_*`, termination vs truncation flags).
- Model and log persistence will be added in the Phase 3 submission (current version runs without saving).

> **Tip:** GPU warnings about CUDA initialization indicate PyTorch fell back to CPU; training still proceeds.

---

## 5. Inspect Network Forward/Backward Once (Optional)

To help graders verify NN design without running a full training loop:

```bash
python debug_nn_io.py --model-type lstm   # or --model-type trans
```

This script:
- Creates the requested network.
- Runs a single forward + backward pass on random data.
- Prints tensor shapes and statistics at the policy/value entry points.
- Leaves the project state untouched.

---

## 6. Environment Controls and Reward

`obstacle_env.py` implements:
- PD control for agent joints and obstacle behaviors (static/periodic/reactive).
- 100,000-step episode horizon enforced via `truncated=True`.
- Reward decomposition recorded in `info["R_separate"]` with components for distance, success, failure, collisions, action smoothness, and time penalty.

Terminology:
- `terminated=True` – scoring event (either agent or obstacle succeeds).
- `truncated=True` – episode hits configured horizon.

---

## 7. Switching Policy Architecture

`main.py` wires both network choices through the same PPO trainer:

- **LSTM variant** (`nerual_network_lstm.py`)
  - Single-layer LSTM encoder, LayerNorm, feature MLP.
  - State-dependent `log_std` per action dimension, clamped to `[-5, 2]`.

- **Transformer variant** (`nerual_network_trans.py`)
  - Two-layer Transformer encoder with learnable position embeddings.
  - Identical feature MLP + state-dependent `log_std` head.

Swap architectures via `--model-type` without touching trainer code.

---

## 8. Reproduce Results from Scratch

```bash
cd Group4_fencing_NN_submit
conda activate drl_mujoco
python main.py --model-type lstm --obstacle-mode static
```

Expect several training updates (each 2,048 steps). Rewards begin near zero and rise once successful bouts occur.

For Transformer testing:

```bash
python main.py --model-type trans --obstacle-mode static --total-timesteps 800000
```
