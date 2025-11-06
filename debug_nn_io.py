import argparse
import os

import torch

from learning_agent_ppo import PPOConfig


def tensor_summary(tensor: torch.Tensor) -> str:
    data = tensor.detach().float()
    if data.numel() == 0:
        return f"shape={tuple(data.shape)}, empty"
    return (
        f"shape={tuple(data.shape)}, "
        f"mean={data.mean().item():.4f}, std={data.std(unbiased=False).item():.4f}, "
        f"min={data.min().item():.4f}, max={data.max().item():.4f}"
    )


def summarize(obj):
    if isinstance(obj, torch.Tensor):
        return tensor_summary(obj)
    if isinstance(obj, (list, tuple)):
        return [summarize(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Debug NN forward/backward once.")
    parser.add_argument(
        "--model-type",
        choices=["lstm", "trans"],
        default="lstm",
        help="Which policy/value network to instantiate.",
    )
    args = parser.parse_args()

    if args.model_type == "lstm":
        from nerual_network_lstm import PolicyNetwork, ValueNetwork
        policy_kwargs = {"lstm_hidden_size": PPOConfig().lstm_hidden_size}
        value_kwargs = {"lstm_hidden_size": PPOConfig().lstm_hidden_size}
        hidden_dims = PPOConfig().hidden_dims
    else:
        from nerual_network_trans import PolicyNetwork, ValueNetwork
        config = PPOConfig()
        policy_kwargs = {
            "seq_len": config.sequence_length,
            "embed_dim": config.transformer_embed_dim,
            "num_heads": config.transformer_num_heads,
            "num_layers": config.transformer_num_layers,
            "dropout": config.transformer_dropout,
        }
        value_kwargs = policy_kwargs.copy()
        hidden_dims = config.hidden_dims

    cfg = PPOConfig()
    obs_dim = cfg.sequence_length * 0 + 24  # placeholder to keep pylint calm
    obs_dim = 24
    act_dim = 8
    seq_len = cfg.sequence_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(obs_dim, act_dim, hidden_dims, **policy_kwargs).to(device)
    value = ValueNetwork(obs_dim, hidden_dims, **value_kwargs).to(device)

    obs_seq = torch.randn(1, seq_len, obs_dim, device=device, requires_grad=True)

    handles = []

    def make_forward(name):
        def hook(module, inputs, outputs):
            print(f"[DEBUG][{name}] forward input: {summarize(inputs)}")
            print(f"[DEBUG][{name}] forward output: {summarize(outputs)}")
        return hook

    def make_backward(name):
        def hook(module, grad_input, grad_output):
            print(f"[DEBUG][{name}] backward grad_input: {summarize(grad_input)}")
            print(f"[DEBUG][{name}] backward grad_output: {summarize(grad_output)}")
        return hook

    handles.append(policy.register_forward_hook(make_forward("policy")))
    handles.append(policy.register_full_backward_hook(make_backward("policy")))
    handles.append(value.register_forward_hook(make_forward("value")))
    handles.append(value.register_full_backward_hook(make_backward("value")))

    mean, log_std = policy(obs_seq)
    value_pred = value(obs_seq)
    dummy_loss = mean.pow(2).mean() + log_std.pow(2).mean() + value_pred.pow(2).mean()
    dummy_loss.backward()

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
