#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os

from obstacle_env import ObstacleEnv
from learning_agent_ppo import PPOTrainer, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fencing agent with PPO.")
    parser.add_argument(
        "--model-type",
        choices=["lstm", "trans"],
        default="lstm",
        help="Policy/value network family to use (default: lstm).",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=["static", "none", "periodic", "reactive"],
        default="periodic",
        help="Obstacle controller mode passed to the environment (default: periodic).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Override PPO total training timesteps (default: 600000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed used for training (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_type == "lstm":
        from nerual_network_lstm import PolicyNetwork, ValueNetwork
    else:
        from nerual_network_trans import PolicyNetwork, ValueNetwork

    config = PPOConfig()
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.seed is not None:
        config.seed = args.seed

    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = ObstacleEnv(xml_path=xml_path, obstacle_mode=args.obstacle_mode, render_mode=None)
    trainer = PPOTrainer(env, config, PolicyNetwork, ValueNetwork)
    trainer.train()

if __name__ == "__main__":
    main()
