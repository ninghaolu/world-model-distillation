"""
DMD World Model Inference Wrapper.

Mirrors CausalInferencePipeline from the CausalForcing codebase, adapted for
our DiT + action conditioning (no text encoder, no CFG).
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import logging
import time
from typing import Any, Optional

import einops
import torch
import torch.nn.functional as F

from model_with_kv_cache import DiT
from vae import VAE
from dmd.cache import create_kv_cache, evict_oldest_block
from dmd.diffusion_utils import get_alphas_cumprod, get_denoising_step_list, v_pred_to_x0, q_sample

logger = logging.getLogger(__name__)

def _debug_print_memory_stats_keys():
    try:
        stats = torch.cuda.memory_stats()
        if isinstance(stats, dict):
            print("[GPU DEBUG] torch.cuda.memory_stats keys:")
            for k in sorted(stats.keys()):
                print("  ", k)
        else:
            print("[GPU DEBUG] memory_stats returned:", type(stats))
    except Exception as e:
        print("[GPU DEBUG] memory_stats not available:", e)


# GPU memory helper (adapted from demo_utils/memory.py) 
def _get_free_vram_gb(device: torch.device) -> float:
    stats = torch.cuda.memory_stats(device)
    # If memory_stats works
    if isinstance(stats, dict) and len(stats) > 0:
        allocated = stats.get("allocated_bytes.all.current")
        reserved  = stats.get("reserved_bytes.all.current")

        # Safety fallback
        if allocated is None or reserved is None:
            free, _ = torch.cuda.mem_get_info(device)
            return free / 1024**3

        free_cuda, _ = torch.cuda.mem_get_info(device)
        return (free_cuda + reserved - allocated) / 1024**3

    # If memory_stats unsupported (your case)
    else:
        free, _ = torch.cuda.mem_get_info(device)
        return free / 1024**3


def _gpu_summary(device: torch.device) -> str:
    free_gb      = _get_free_vram_gb(device)
    reserved_gb  = torch.cuda.memory_reserved(device)  / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    return (
        f"GPU {device} | free={free_gb:.2f} GB | "
        f"allocated={allocated_gb:.2f} GB | reserved={reserved_gb:.2f} GB"
    )


class DMDWorldModel:
    """
    Inference wrapper for a DMD-distilled world model.

    Follows CausalInferencePipeline.inference() step-for-step:
      Step 1: Initialize KV cache
      Step 2: Seed cache with the real initial frame at t=0
      Step 3: For each output frame:
        3.1 - Few-step denoising loop using denoising_step_list
        3.2 - Record the clean output
        3.3 - Rerun at t=context_noise to write clean KV into cache
        3.4 - Advance frame counter

    Interface matches the pretrained WorldModel:
      - __init__(checkpoint_path, config)
      - reset(x)                x: [B, H, W, C] first frame, pixel space [0,1]
      - generate_chunk(action)  yields (frame_idx, decoded_frame)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Any,
        use_ema: bool = True,
        context_noise: int = 0,
        max_cache_frames: int = 15,
        denoising_step_list: Optional[list] = None,
        num_frame_per_block: int = 1,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16
        self.context_noise = context_noise
        self.max_cache_frames = max_cache_frames
        self.chunk_size = config.chunk_size

        print(f"\n{'='*60}")
        print(f"[DMDWorldModel] Initializing...")
        print(f"  checkpoint : {checkpoint_path}")
        print(f"  device     : {self.device}")
        print(f"  use_ema    : {use_ema}")
        print(f"  max_cache_frames : {max_cache_frames}")
        print(f"  context_noise    : {context_noise}")
        _debug_print_memory_stats_keys()
        print(f"  {_gpu_summary(self.device)}")

        # VAE (frozen)
        print(f"\n[DMDWorldModel] Loading VAE...")
        self.vae = VAE().to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        in_channels = self.vae.vae.config.latent_channels
        print(f"  VAE loaded  | in_channels={in_channels} | {_gpu_summary(self.device)}")

        # DiT generator
        print(f"\n[DMDWorldModel] Building DiT...")
        print(f"  dim={config.model_dim}, layers={config.layers}, "
              f"heads={config.heads}, patch_size={config.patch_size}, "
              f"action_dim={config.action_dim}, max_frames={config.n_frames}")
        self.model = DiT(
            in_channels=in_channels,
            patch_size=config.patch_size,
            dim=config.model_dim,
            num_layers=config.layers,
            num_heads=config.heads,
            action_dim=config.action_dim,
            max_frames=config.n_frames,
            external_cond_dropout_prob=0.0,
        ).to(self.device).eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  DiT built   | params={n_params/1e6:.1f}M | {_gpu_summary(self.device)}")

        # Load checkpoint
        print(f"\n[DMDWorldModel] Loading checkpoint weights ({use_ema=})...")
        t0 = time.time()
        data = torch.load(checkpoint_path, map_location="cpu")
        weight_key = "generator_ema" if use_ema else "generator"
        if weight_key not in data:
            raise KeyError(
                f"Checkpoint missing key '{weight_key}'. "
                f"Available keys: {list(data.keys())}"
            )
        self.model.load_state_dict(data[weight_key], strict=True)
        print(f"  Loaded '{weight_key}' in {time.time()-t0:.1f}s | {_gpu_summary(self.device)}")

        # Denoising step list
        if denoising_step_list is not None:
            self.denoising_step_list = denoising_step_list
        elif "config" in data and "denoising_step_list" in data["config"]:
            self.denoising_step_list = data["config"]["denoising_step_list"]
        else:
            timesteps = getattr(config, "timesteps", 1000)
            num_steps = getattr(config, "num_denoising_steps", 4)
            self.denoising_step_list = get_denoising_step_list(timesteps, num_steps)
        print(f"  denoising_step_list: {self.denoising_step_list}")

        del data

        # Noise schedule
        timesteps = getattr(config, "timesteps", 1000)
        self.alphas_cumprod = get_alphas_cumprod(timesteps).to(self.device)

        # Runtime state
        self.kv_cache = None
        self.batch_size = None
        self.latent_shape = None
        self.curr_frame = 0
        self.num_frame_per_block = num_frame_per_block

        print(f"\n[DMDWorldModel] Ready | {_gpu_summary(self.device)}")
        print(f"{'='*60}\n")

    # Public API
    def reset(self, x: torch.Tensor):
        """
        Initialize with the first frame.

        Args:
            x: [B, H, W, C] first frame, pixel space, values in [0, 1].
        """
        print(f"\n[reset] Starting new episode | batch_size={x.shape[0]} | "
              f"frame_shape={x.shape[1:]} | {_gpu_summary(self.device)}")

        x = einops.repeat(x, "b h w c -> b t h w c", t=1)
        self.batch_size = x.shape[0]

        # Encode first frame
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                initial_latent = self.vae.encode(x.to(self.device))

        _, _, H_l, W_l, C_l = initial_latent.shape
        self.latent_shape = (H_l, W_l, C_l)
        print(f"[reset] VAE encode done | latent_shape=({H_l},{W_l},{C_l}) | "
              f"{_gpu_summary(self.device)}")

        # Step 1: Initialize KV cache
        patch_size = self.model.patch_size
        h_patch = H_l // patch_size
        w_patch = W_l // patch_size
        self.kv_cache = create_kv_cache(
            num_layers=self.model.num_layers,
            batch_size=self.batch_size,
            max_frames=self.max_cache_frames,
            num_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            h_patch=h_patch,
            w_patch=w_patch,
            device=self.device,
            dtype=self.dtype,
        )
        # Estimate cache memory: 2 (K+V) * num_layers * B*H*W * num_heads * max_frames * head_dim * 2 bytes
        cache_bytes = (
            2 * self.model.num_layers
            * (self.batch_size * h_patch * w_patch)
            * self.model.num_heads
            * self.max_cache_frames
            * self.model.head_dim
            * 2  # bfloat16
        )
        print(f"[reset] KV cache created | "
              f"layers={self.model.num_layers}, max_frames={self.max_cache_frames}, "
              f"spatial={h_patch}x{w_patch} | "
              f"~{cache_bytes/1024**3:.3f} GB | {_gpu_summary(self.device)}")

        # Step 2: Seed cache with initial frame at t=0
        t_zero = torch.zeros([self.batch_size, 1], device=self.device, dtype=torch.long)
        initial_action = torch.zeros(
            [self.batch_size, 1, self.model.action_dim],
            device=self.device, dtype=self.dtype,
        )
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                self.model.forward_and_cache(
                    x=initial_latent.to(dtype=self.dtype),
                    t=t_zero,
                    action=initial_action,
                    external_kv_cache=self.kv_cache,
                )
        self.curr_frame = 1
        print(f"[reset] Cache seeded with frame 0 | curr_frame=1 | "
              f"{_gpu_summary(self.device)}")

    @torch.no_grad()
    def generate_chunk(self, action_vec: torch.Tensor):
        """
        Generate the next frame given an action.

        Yields:
            (frame_idx, decoded): int and [B, 1, H_pix, W_pix, C] in [0,1].
        """
        H_l, W_l, C_l = self.latent_shape
        frame_idx = self.curr_frame
        t0 = time.time()

        # Normalize to [B, n, D]
        if action_vec.dim() == 2:
            action_vec = action_vec.unsqueeze(1)  # [B, D] → [B, 1, D]
        # action_vec is now [B, n, D]
        n = action_vec.shape[1]

        action_vec = action_vec.to(device=self.device, dtype=self.dtype)
        if action_vec.shape[-1] < self.model.action_dim:
            action_vec = F.pad(action_vec, (0, self.model.action_dim - action_vec.shape[-1]))

        t_zero_block = torch.zeros([self.batch_size, n], device=self.device, dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            noisy_input = torch.randn(
                [self.batch_size, n, H_l, W_l, C_l],
                device=self.device, dtype=self.dtype,
            )
            denoised_x0 = None
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.full([self.batch_size, n], current_timestep, device=self.device, dtype=torch.long)
                v_pred = self.model(x=noisy_input, t=timestep, action=action_vec, external_kv_cache=self.kv_cache)
                denoised_x0 = v_pred_to_x0(v_pred, noisy_input, self.alphas_cumprod, timestep)
                if step_idx < len(self.denoising_step_list) - 1:
                    next_t = torch.full([self.batch_size, n], self.denoising_step_list[step_idx + 1], device=self.device, dtype=torch.long)
                    noisy_input = q_sample(denoised_x0, torch.randn_like(denoised_x0), self.alphas_cumprod, next_t)

            while self.kv_cache[0]["end_idx"] + n > self.max_cache_frames:
                evict_oldest_block(self.kv_cache, n)
            self.model.forward_and_cache(x=denoised_x0, t=t_zero_block, action=action_vec, external_kv_cache=self.kv_cache)

            decoded = self.vae.decode(denoised_x0)  # [B, n, H, W, C]

        self.curr_frame += n
        for i in range(n):
            yield frame_idx + i, decoded[:, i:i+1]