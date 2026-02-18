"""
External KV cache for Self-Forcing training.

Mirrors the Wan codebase's approach: pre-allocated cache tensors passed into
every model call, rather than storing cache internally on attention layers.

The cache stores K and V representations at the **frame level** for temporal
attention. Spatial attention does not use caching (it operates per-frame).

Cache layout per layer:
    {
        "k": [B*H_patch*W_patch, num_heads, max_frames, head_dim],
        "v": [B*H_patch*W_patch, num_heads, max_frames, head_dim],
        "end_idx": 0,  # number of frames currently stored
    }

Here B*H_patch*W_patch is the batch dimension for temporal attention,
since spatial dims are folded into the batch. num_heads and head_dim
come from the model's attention configuration.

Usage:
    cache = create_kv_cache(num_layers=16, batch_size=2, ...)
    # ... pass cache into model calls ...
    clear_kv_cache(cache)
"""

import torch
from typing import Optional


def create_kv_cache(
    num_layers: int,
    batch_size: int,
    max_frames: int,
    num_heads: int,
    head_dim: int,
    h_patch: int,
    w_patch: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """
    Create a pre-allocated external KV cache.

    Args:
        num_layers: number of transformer blocks (each has one temporal attn layer)
        batch_size: B
        max_frames: maximum number of frames the cache can hold
        num_heads: number of attention heads
        head_dim: dimension per head (= dim // num_heads)
        h_patch: height after patchification (= H // patch_size)
        w_patch: width after patchification (= W // patch_size)
        device: torch device
        dtype: data type for cache tensors
    Returns:
        List of cache dicts, one per layer.
    """
    # Temporal attention operates on shape (B*H_patch*W_patch, num_heads, T, head_dim)
    spatial_batch = batch_size * h_patch * w_patch

    cache = []
    for _ in range(num_layers):
        cache.append({
            "k": torch.zeros(
                [spatial_batch, num_heads, max_frames, head_dim],
                device=device, dtype=dtype,
            ),
            "v": torch.zeros(
                [spatial_batch, num_heads, max_frames, head_dim],
                device=device, dtype=dtype,
            ),
            "end_idx": 0,
        })
    return cache


def clear_kv_cache(cache: list[dict]) -> None:
    """Reset all cache entries to zero and reset indices."""
    for layer_cache in cache:
        layer_cache["k"].zero_()
        layer_cache["v"].zero_()
        layer_cache["end_idx"] = 0


def evict_oldest_frame(cache: list[dict]) -> None:
    """
    Rolling KV cache eviction (Algorithm 2, Line 10-11).

    Removes the oldest frame's KV entries by shifting everything left by 1.
    Used for long video generation beyond max_frames.
    """
    for layer_cache in cache:
        end_idx = layer_cache["end_idx"]
        if end_idx <= 0:
            return

        # Shift frames left by 1
        layer_cache["k"][:, :, :end_idx - 1, :] = layer_cache["k"][:, :, 1:end_idx, :].clone()
        layer_cache["v"][:, :, :end_idx - 1, :] = layer_cache["v"][:, :, 1:end_idx, :].clone()

        # Zero out the now-free slot
        layer_cache["k"][:, :, end_idx - 1, :] = 0
        layer_cache["v"][:, :, end_idx - 1, :] = 0

        layer_cache["end_idx"] = end_idx - 1


def get_cache_info(cache: list[dict]) -> dict:
    """Get summary info about cache state (for debugging)."""
    if not cache:
        return {"num_layers": 0}
    return {
        "num_layers": len(cache),
        "max_frames": cache[0]["k"].shape[2],
        "end_idx": cache[0]["end_idx"],
        "spatial_batch": cache[0]["k"].shape[0],
        "num_heads": cache[0]["k"].shape[1],
        "head_dim": cache[0]["k"].shape[3],
        "device": str(cache[0]["k"].device),
        "dtype": str(cache[0]["k"].dtype),
    }