"""
Self-Forcing generation for DMD training (Algorithm 1).

Generates video autoregressively frame-by-frame using few-step denoising,
with external KV cache for temporal conditioning.

Key design: we generate MORE frames than we score (n_total_frames > n_output_frames).
The early "warmup" frames build KV cache context but are discarded. With a limited
cache size (max_cache_frames), the first frame's KV gets evicted by rolling eviction,
simulating inference-time conditions with rolling KV cache. This prevents the
flickering artifacts described in the paper (Section 3.4):

    "during training, we restrict the attention window so the model cannot attend
     to the first chunk when denoising the final chunk, thereby simulating the
     conditions encountered during long video generation."

Algorithm 1 (Self Forcing Training):
    Initialize output X_θ, KV cache
    Sample exit step s ~ Uniform(1, T)
    For each frame i = 1, ..., N:
        Initialize x^i_{t_T} ~ N(0, I)
        For j = T, ..., s:
            if j == s: enable gradient
                x̂⁰ = G_θ(x^i_{t_j}, t_j, KV)
                X_θ.append(x̂⁰)
                disable gradient
                Cache kv^i = G^KV_θ(x̂⁰, 0, KV)
                KV.append(kv^i)
            else: (no gradient)
                x̂⁰ = G_θ(x^i_{t_j}, t_j, KV)
                ε ~ N(0, I)
                x^i_{t_{j-1}} = Ψ(x̂⁰, ε, t_{j-1})
    Update θ via distribution matching loss (on last n_output_frames only)
"""

import torch
import torch.distributed as dist
from typing import Optional

from dmd.cache import create_kv_cache, clear_kv_cache, evict_oldest_block
from dmd.diffusion_utils import v_pred_to_x0, q_sample


def self_forcing_rollout(
    generator: torch.nn.Module,
    initial_latent: torch.Tensor,
    initial_action: torch.Tensor,
    actions: torch.Tensor,
    denoising_step_list: list[int],
    alphas_cumprod: torch.Tensor,
    n_total_frames: int,
    n_output_frames: int,
    max_cache_frames: int,
    grad_frames_from: Optional[int] = None,
    num_frame_per_block: int = 1,
) -> torch.Tensor:
    """
    Self-Forcing autoregressive rollout (Algorithm 1) with rolling KV cache.

    Generates `n_total_frames - 1` frames conditioned on `initial_latent` (frame 0).
    Only the last `n_output_frames` are returned for the DMD loss.
    The earlier "warmup" frames build KV cache context and are discarded.

    With max_cache_frames < n_total_frames, the rolling eviction kicks in,
    ensuring the model learns to generate without seeing the first frame's
    KV embeddings — matching inference-time rolling cache conditions.

    Example with n_total_frames=30, n_output_frames=20, max_cache_frames=20:
        - Generate frames 0..29 (frame 0 is the real initial latent)
        - After frame 20, the cache is full → frame 0's KV gets evicted
        - Frames 21..29 can no longer attend to frame 0
        - Return frames 10..29 for DMD loss
        - This simulates the rolling cache scenario at inference

    Args:
        generator: DiT model with external KV cache support
        initial_latent: [B, 1, H, W, C] first frame (clean latent from VAE)
        initial_action: [B, 1, D] action for the first frame
        actions: [B, n_total_frames, D] full action sequence
        denoising_step_list: list of timesteps [t_T, t_{T-1}, ..., t_1]
            e.g., [999, 749, 499, 249] for 4-step denoising
        alphas_cumprod: [1000] noise schedule
        n_total_frames: total frames to generate (including initial frame)
        n_output_frames: how many trailing frames to return for DMD loss
        max_cache_frames: maximum frames the KV cache can hold.
            Should be <= n_total_frames so rolling eviction occurs.
        grad_frames_from: frame index (in OUTPUT space, 0-indexed) from which
            to enable gradients. If None, no gradients (for critic training).
            E.g., grad_frames_from=19 means only the last output frame has grad.
    Returns:
        output: [B, n_output_frames, H, W, C] the last n_output_frames of the
            generated video, for use with DMD loss.
    """
    assert n_output_frames <= n_total_frames
    assert actions.shape[1] >= n_total_frames

    # n_total_frames - 1 generated frames must be divisible by num_frame_per_block
    n_gen_frames = n_total_frames - 1  # excluding the initial real frame
    assert n_gen_frames % num_frame_per_block == 0, (
        f"n_total_frames-1 ({n_gen_frames}) must be divisible by num_frame_per_block ({num_frame_per_block})"
    )

    B, _, H, W, C = initial_latent.shape
    device = initial_latent.device
    dtype = initial_latent.dtype

    num_denoising_steps = len(denoising_step_list)
    num_blocks = n_gen_frames // num_frame_per_block
    output_start_frame = n_total_frames - n_output_frames  # in full-sequence indexing

    # Sample ONE exit step, shared across all blocks (broadcast for DDP consistency)
    exit_step_idx = torch.randint(0, num_denoising_steps, (1,), device=device)
    if dist.is_initialized():
        dist.broadcast(exit_step_idx, src=0)
    exit_step_idx = exit_step_idx.item()

    # Patch dimensions for cache allocation
    patch_size = generator.patch_size
    h_patch = H // patch_size
    w_patch = W // patch_size

    # Create external KV cache
    kv_cache = create_kv_cache(
        num_layers=generator.num_layers,
        batch_size=B,
        max_frames=max_cache_frames,
        num_heads=generator.num_heads,
        head_dim=generator.head_dim,
        h_patch=h_patch,
        w_patch=w_patch,
        device=device,
        dtype=dtype,
    )

    # Write initial (real) frame into the cache at t=0
    t_zero = torch.zeros([B, 1], device=device, dtype=torch.long)
    with torch.no_grad():
        generator.forward_and_cache(
            x=initial_latent,
            t=t_zero,
            action=initial_action,
            external_kv_cache=kv_cache,
        )

    output_frames = []  # list of [B, H, W, C] tensors for the output window

    # Iterate over blocks
    for block_idx in range(num_blocks):
        # Frame indices (in the full sequence) for this block: 1-indexed since frame 0 is real
        block_start = 1 + block_idx * num_frame_per_block          # e.g., 1, 4, 7, ...
        block_end   = block_start + num_frame_per_block            # exclusive
        block_frame_indices = list(range(block_start, block_end))  # [block_start, ..., block_end-1]

        # Actions for this block: [B, num_frame_per_block, D]
        frame_actions = actions[:, block_start:block_end]

        # Determine grad eligibility for each frame in this block
        # A frame needs grad if it's in the output window AND output_idx >= grad_frames_from
        block_enable_grad = False
        for fi in block_frame_indices:
            in_output = fi >= output_start_frame
            if in_output and grad_frames_from is not None:
                out_idx = fi - output_start_frame
                if out_idx >= grad_frames_from:
                    block_enable_grad = True
                    break

        # Start from pure noise for the whole block: [B, num_frame_per_block, H, W, C]
        noisy_input = torch.randn([B, num_frame_per_block, H, W, C], device=device, dtype=dtype)

        # Timestep tensor: [B, num_frame_per_block] — same value across all frames in block
        def make_timestep(t_val):
            return torch.full([B, num_frame_per_block], t_val, device=device, dtype=torch.long)

        denoised_x0 = None

        for step_idx, current_timestep in enumerate(denoising_step_list):
            is_exit_step = (step_idx == exit_step_idx)
            timestep = make_timestep(current_timestep)

            if is_exit_step and block_enable_grad:
                # WITH gradient, generator learns from this block
                v_pred = generator(
                    x=noisy_input,
                    t=timestep,
                    action=frame_actions,
                    external_kv_cache=kv_cache,
                )
                denoised_x0 = v_pred_to_x0(v_pred, noisy_input, alphas_cumprod, timestep)
                break

            elif is_exit_step and not block_enable_grad:
                with torch.no_grad():
                    v_pred = generator(
                        x=noisy_input,
                        t=timestep,
                        action=frame_actions,
                        external_kv_cache=kv_cache,
                    )
                    denoised_x0 = v_pred_to_x0(v_pred, noisy_input, alphas_cumprod, timestep)
                break

            else:
                # Intermediate denoising step — no grad, re-noise for next step
                with torch.no_grad():
                    v_pred = generator(
                        x=noisy_input,
                        t=timestep,
                        action=frame_actions,
                        external_kv_cache=kv_cache,
                    )
                    x0_pred = v_pred_to_x0(v_pred, noisy_input, alphas_cumprod, timestep)
                    next_timestep = denoising_step_list[step_idx + 1]
                    next_t = make_timestep(next_timestep)
                    fresh_noise = torch.randn_like(x0_pred)
                    noisy_input = q_sample(x0_pred, fresh_noise, alphas_cumprod, next_t)

        # Collect frames that fall in the output window
        for local_i, fi in enumerate(block_frame_indices):
            if fi >= output_start_frame:
                output_frames.append(denoised_x0[:, local_i])  # [B, H, W, C]

        # Cache update: evict if full, then write this block's denoised output
        with torch.no_grad():
            t_zero_block = torch.zeros([B, num_frame_per_block], device=device, dtype=torch.long)
            while kv_cache[0]["end_idx"] + num_frame_per_block > max_cache_frames:
                evict_oldest_block(kv_cache, num_frame_per_block)

            generator.forward_and_cache(
                x=denoised_x0.detach(),
                t=t_zero_block,
                action=frame_actions,
                external_kv_cache=kv_cache,
            )

    output = torch.stack(output_frames, dim=1)  # [B, n_output_frames, H, W, C]
    return output