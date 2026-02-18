"""
DMD (Distribution Matching Distillation) loss functions.

Implements:
  - KL gradient computation (DMD paper eq 7)
  - Generator loss (pseudo-MSE with stop-gradient)
  - Critic loss (standard v-prediction denoising loss)

References:
  - DMD:  https://arxiv.org/abs/2311.18828
  - DMD2: https://arxiv.org/abs/2405.14867
"""

import torch
import torch.nn.functional as F

from dmd.diffusion_utils import v_pred_to_x0, q_sample, v_target


def run_score_model(
    model: torch.nn.Module,
    noisy_input: torch.Tensor,
    timestep: torch.Tensor,
    action: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    """
    Run a score model (real or fake) and return x0 prediction.

    The score models process the FULL generated sequence (all frames at once)
    using causal temporal attention. No KV cache needed — they see the
    full sequence in a single forward pass.

    Args:
        model: DiT score model (real_score or fake_score)
        noisy_input: [B, T, H, W, C] noised generated video
        timestep: [B, T] per-frame timesteps
        action: [B, T, D] per-frame actions
        alphas_cumprod: [1000] noise schedule
        cfg_scale: classifier-free guidance scale
            0.0 or 1.0 = no guidance, >1.0 = guided
    Returns:
        x0_pred: [B, T, H, W, C]
    """
    v_cond = model(noisy_input, timestep, action)
    x0_cond = v_pred_to_x0(v_cond, noisy_input, alphas_cumprod, timestep)

    if cfg_scale > 1.0:
        null_action = model.get_null_action(action)
        v_uncond = model(noisy_input, timestep, null_action)
        x0_uncond = v_pred_to_x0(v_uncond, noisy_input, alphas_cumprod, timestep)
        x0_pred = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
    else:
        x0_pred = x0_cond

    return x0_pred


def compute_kl_grad(
    noisy_latent: torch.Tensor,
    estimated_clean: torch.Tensor,
    timestep: torch.Tensor,
    action: torch.Tensor,
    real_score: torch.nn.Module,
    fake_score: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    real_guidance_scale: float = 1.5,
    fake_guidance_scale: float = 0.0,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the KL gradient (DMD paper eq 7).

    grad = fake_score_x0(x_t) - real_score_x0(x_t)

    Intuitively:
      - real_score_x0: teacher's prediction of clean data from noisy input
      - fake_score_x0: critic's prediction (models the student's distribution)
      - Their difference tells the generator how to shift its outputs

    Args:
        noisy_latent: [B, T, H, W, C] noised generated video
        estimated_clean: [B, T, H, W, C] generator's clean output (for normalization)
        timestep: [B, T] per-frame noise timesteps
        action: [B, T, D] per-frame actions
        real_score: frozen teacher model
        fake_score: trainable critic model
        alphas_cumprod: noise schedule
        real_guidance_scale: CFG scale for teacher (>1 for sharper gradients)
        fake_guidance_scale: CFG scale for critic (typically 0 or 1)
        normalize: whether to apply gradient normalization (DMD eq 8)
    Returns:
        grad: [B, T, H, W, C] the KL gradient
        log_dict: logging info
    """
    # Fake score (critic) prediction
    x0_fake = run_score_model(
        fake_score, noisy_latent, timestep, action,
        alphas_cumprod, cfg_scale=fake_guidance_scale,
    )

    # Real score (teacher) prediction
    x0_real = run_score_model(
        real_score, noisy_latent, timestep, action,
        alphas_cumprod, cfg_scale=real_guidance_scale,
    )

    # KL gradient
    grad = x0_fake - x0_real

    if normalize:
        # Gradient normalization (DMD paper eq 8)
        p_real = estimated_clean - x0_real
        normalizer = torch.abs(p_real).mean(
            dim=list(range(1, p_real.dim())), keepdim=True
        ).clamp(min=1e-8)
        grad = grad / normalizer

    grad = torch.nan_to_num(grad)

    return grad, {
        "dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
    }


def compute_generator_loss(
    generated_video: torch.Tensor,
    actions: torch.Tensor,
    real_score: torch.nn.Module,
    fake_score: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    min_timestep: int = 20,
    max_timestep: int = 980,
    real_guidance_scale: float = 1.5,
    fake_guidance_scale: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the DMD generator loss.

    Steps:
      1. Sample random DDPM timestep for each frame
      2. Add noise to the generated video
      3. Both score networks predict x0 from the noisy video
      4. KL gradient = fake_x0 - real_x0
      5. Loss = 0.5 * MSE(generated, (generated - grad).detach())
         Only `generated_video` carries gradients; target is detached.

    Args:
        generated_video: [B, T, H, W, C] output from self-forcing rollout (has grad)
        actions: [B, T, D] corresponding actions
        real_score: frozen teacher
        fake_score: trainable critic
        alphas_cumprod: noise schedule
        min_timestep: minimum noise level for scoring
        max_timestep: maximum noise level for scoring
        real_guidance_scale: CFG for teacher
        fake_guidance_scale: CFG for critic
    Returns:
        loss: scalar
        log_dict: logging info
    """
    B, T = generated_video.shape[:2]

    with torch.no_grad():
        # Sample ONE random timestep per sample (NOT per frame)
        # The score models process the full video — all frames get the same noise level
        # This matches CausalForcing's _get_timestep which samples (B,) timesteps
        timestep_per_sample = torch.randint(
            min_timestep, max_timestep, (B,),
            device=generated_video.device, dtype=torch.long,
        )
        # Broadcast to (B, T) for q_sample
        timestep = timestep_per_sample.unsqueeze(1).expand(B, T)

        # Add noise to generated video
        noise = torch.randn_like(generated_video)
        noisy = q_sample(generated_video, noise, alphas_cumprod, timestep)

        # Compute KL gradient
        grad, log_dict = compute_kl_grad(
            noisy_latent=noisy,
            estimated_clean=generated_video,
            timestep=timestep,
            action=actions,
            real_score=real_score,
            fake_score=fake_score,
            alphas_cumprod=alphas_cumprod,
            real_guidance_scale=real_guidance_scale,
            fake_guidance_scale=fake_guidance_scale,
        )

    # Pseudo-MSE loss
    # generated_video has grad, target is detached → gradients only flow through generator
    target = (generated_video - grad).detach()
    loss = 0.5 * F.mse_loss(generated_video.float(), target.float())

    return loss, log_dict


def compute_critic_loss(
    generated_video: torch.Tensor,
    actions: torch.Tensor,
    fake_score: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    min_timestep: int = 20,
    max_timestep: int = 980,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the critic (fake_score) denoising loss.

    Standard v-prediction loss on the generator's detached outputs.
    This teaches the critic to model the student's current distribution.

    Args:
        generated_video: [B, T, H, W, C] detached generator output
        actions: [B, T, D] corresponding actions
        fake_score: trainable critic
        alphas_cumprod: noise schedule
        min_timestep: minimum noise level
        max_timestep: maximum noise level
    Returns:
        loss: scalar
        log_dict: logging info
    """
    B, T = generated_video.shape[:2]

    # Sample ONE timestep per sample, broadcast to all frames
    timestep_per_sample = torch.randint(
        min_timestep, max_timestep, (B,),
        device=generated_video.device, dtype=torch.long,
    )
    timestep = timestep_per_sample.unsqueeze(1).expand(B, T)
    noise = torch.randn_like(generated_video)
    noisy = q_sample(generated_video, noise, alphas_cumprod, timestep)

    # Critic predicts v
    v_pred = fake_score(noisy, timestep, actions)

    # v-prediction target: v = sqrt(α) * noise - sqrt(1-α) * x0
    target = v_target(generated_video, noise, alphas_cumprod, timestep)

    loss = F.mse_loss(v_pred, target)

    return loss, {"critic_timestep": timestep.detach()}