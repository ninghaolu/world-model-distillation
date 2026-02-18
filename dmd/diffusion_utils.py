"""
Diffusion utilities for DMD training.

Handles:
  - Noise schedule (sigmoid beta schedule, matching the pretrained model)
  - v-prediction <-> x0 conversions
  - Forward diffusion (q_sample)
  - Denoising step list generation
"""

import torch
import math


def sigmoid_beta_schedule(
    timesteps: int, start: float = -3, end: float = 3, tau: float = 1
) -> torch.Tensor:
    """
    Sigmoid beta schedule from https://arxiv.org/abs/2212.11972.
    Identical to Diffusion.sigmoid_beta_schedule in your diffusion.py.
    """
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (
        -((t * (end - start) + start) / tau).sigmoid() + v_end
    ) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


def get_alphas_cumprod(timesteps: int = 1000) -> torch.Tensor:
    """
    Compute alphas_cumprod from the sigmoid beta schedule.
    Returns a tensor of shape [timesteps].
    """
    betas = sigmoid_beta_schedule(timesteps)
    alphas = 1.0 - betas
    return alphas.cumprod(dim=0)


def get_denoising_step_list(timesteps: int, num_steps: int) -> list[int]:
    """
    Generate evenly-spaced denoising timesteps for few-step generation.

    For num_steps=4 with timesteps=1000, this produces something like [999, 749, 499, 249].
    These are the noise levels the model denoises through, from highest to lowest.

    The spacing matches what DDIM would use with `sampling_timesteps=num_steps`.
    """
    # Evenly spaced from timesteps-1 down to ~0, excluding 0
    # (the final step denoises to clean, no need for t=0 in the list)
    step_size = timesteps / num_steps
    steps = [int(timesteps - 1 - i * step_size) for i in range(num_steps)]
    # Ensure all positive
    steps = [max(s, 1) for s in steps]
    return steps


def v_pred_to_x0(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    Convert v-prediction to x0 prediction.

    Model target:  v = sqrt(α_t) * noise - sqrt(1 - α_t) * x0
    Forward:       x_t = sqrt(α_t) * x0 + sqrt(1 - α_t) * noise
    Solving:       x0 = sqrt(α_t) * x_t - sqrt(1 - α_t) * v

    Args:
        v_pred: [B, T, H, W, C] or [B*T, H, W, C]
        x_t: same shape, noisy input
        alphas_cumprod: [num_timesteps] buffer
        timestep: [B, T] or [B] integer timesteps
    Returns:
        x0_pred: same shape as v_pred
    """
    if timestep.dim() == 2:
        B, T = timestep.shape
        a_t = alphas_cumprod[timestep.reshape(-1)].view(B, T, 1, 1, 1)
    elif timestep.dim() == 1:
        a_t = alphas_cumprod[timestep].view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected timestep dim: {timestep.dim()}")

    return a_t.sqrt() * x_t - (1 - a_t).sqrt() * v_pred


def q_sample(
    x0: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    Forward diffusion process.
    x_t = sqrt(α_t) * x0 + sqrt(1 - α_t) * noise

    Args:
        x0: [B, T, H, W, C] clean data
        noise: same shape, sampled from N(0, I)
        alphas_cumprod: [num_timesteps]
        timestep: [B, T] or [B] integer timesteps
    Returns:
        x_t: same shape, noisy data
    """
    if timestep.dim() == 2:
        B, T = timestep.shape
        a_t = alphas_cumprod[timestep.reshape(-1)].view(B, T, 1, 1, 1)
    elif timestep.dim() == 1:
        a_t = alphas_cumprod[timestep].view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected timestep dim: {timestep.dim()}")

    return a_t.sqrt() * x0 + (1 - a_t).sqrt() * noise


def v_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the v-prediction target.
    v = sqrt(α_t) * noise - sqrt(1 - α_t) * x0

    Used for the critic's denoising loss.
    """
    if timestep.dim() == 2:
        B, T = timestep.shape
        a_t = alphas_cumprod[timestep.reshape(-1)].view(B, T, 1, 1, 1)
    elif timestep.dim() == 1:
        a_t = alphas_cumprod[timestep].view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected timestep dim: {timestep.dim()}")

    return a_t.sqrt() * noise - (1 - a_t).sqrt() * x0