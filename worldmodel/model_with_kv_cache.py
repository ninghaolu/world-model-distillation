"""
DiT model with external KV cache support for Self-Forcing / DMD training.

This extends the original DiT (model.py) with an external KV cache path
for temporal attention, following the Wan/CausalForcing codebase pattern.

Key changes from the original model.py:
  1. SelfAttention gains `_forward_with_external_cache()` method
  2. SelfAttentionBlock.forward() accepts optional `ext_cache` kwarg
  3. DiTBlock.forward() accepts optional `ext_cache` kwarg
  4. DiT.forward() accepts optional `external_kv_cache` + `current_frame_idx`

When `external_kv_cache` is None, the model behaves exactly as the original.
When provided, the external cache is used instead of the internal cache.
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops
import math
import functools
from typing import Sequence, Optional
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class AttentionType(StrEnum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class RotaryType(StrEnum):
    STANDARD = "standard"
    PIXEL = "pixel"


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


@functools.lru_cache
def rope_nd(
    shape: Sequence[int],
    dim: int = 64,
    base: float = 10_000.0,
    rotary_type: RotaryType = RotaryType.STANDARD,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    D = len(shape)
    assert dim % (2 * D) == 0

    dim_per_axis = dim // D
    half = dim_per_axis // 2
    if rotary_type == RotaryType.STANDARD:
        inv_freq = 1.0 / (
            base ** (torch.arange(half, device=device, dtype=dtype) / half)
        )
        coords = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    elif rotary_type == RotaryType.PIXEL:
        inv_freq = (
            torch.linspace(1.0, 256.0 / 2, half, device=device, dtype=dtype) * math.pi
        )
        coords = [
            torch.linspace(-1, +1, steps=n, device=device, dtype=dtype) for n in shape
        ]
    else:
        raise NotImplementedError(f"invalid rotary type: {rotary_type}")

    mesh = torch.meshgrid(*coords, indexing="ij")
    embeddings = []
    for pos in mesh:
        theta = pos.unsqueeze(-1) * inv_freq
        emb_axis = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embeddings.append(emb_axis)
    return torch.cat(embeddings, dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos + rotate_half(x) * sin


def apply_rope_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    shape: tuple[int, ...],
    rotary_type: RotaryType,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.shape[-1]
    rope = rope_nd(
        shape, dim, base, rotary_type=rotary_type, dtype=q.dtype, device=q.device
    )
    rope = rope.view(*shape, len(shape), 2, -1)
    cos, sin = rope.unbind(-2)
    cos = cos.reshape(*shape, -1)
    sin = sin.reshape(*shape, -1)

    k_rot = rope_mix(k, cos, sin)

    if len(shape) == 1:
        q_offset = k.shape[2] - q.shape[2]
        if q_offset > 0:
            cos_q = cos[q_offset:, :]
            sin_q = sin[q_offset:, :]
            q_rot = rope_mix(q, cos_q, sin_q)
        else:
            q_rot = rope_mix(q, cos, sin)
    else:
        q_rot = rope_mix(q, cos, sin)

    return q_rot, k_rot


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 2, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(2, dim=-1)
        x = self.linear(self.norm(x) * (1 + m[1]) + m[0])
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        is_causal: bool,
        attention_type: AttentionType,
        rotary_type: RotaryType = RotaryType.STANDARD,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.is_causal = is_causal
        self.attention_type = attention_type
        self.rotary_type = rotary_type
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

        # Internal cache (preserved for original inference path)
        self.k_cache_cond: torch.Tensor | None = None
        self.v_cache_cond: torch.Tensor | None = None
        self.k_cache_null: torch.Tensor | None = None
        self.v_cache_null: torch.Tensor | None = None
        self.cache_start: int | None = None
        self.cache_end: int | None = None

    def clear_cache(self):
        self.k_cache_cond = None
        self.v_cache_cond = None
        self.k_cache_null = None
        self.v_cache_null = None
        self.cache_start = None
        self.cache_end = None

    def _forward_with_external_cache(
        self,
        x: torch.Tensor,
        ext_cache: dict,
    ) -> torch.Tensor:
        """
        Forward pass using external KV cache for temporal attention.

        This is the Self-Forcing training path. The external cache stores
        K, V representations of all previously generated (clean) frames.
        Current frame's K, V are computed and concatenated with cached ones.

        The cache is written back DETACHED — no gradients flow through cache,
        as specified in the Self-Forcing paper (Section 3.2).

        Args:
            x: [B, T, H, W, D] where T is the number of NEW frames (typically 1)
            ext_cache: dict with keys "k", "v" (pre-allocated tensors), "end_idx" (int)
                k, v shape: [B*H_patch*W_patch, num_heads, max_frames, head_dim]
        Returns:
            out: [B, T, H, W, D]
        """
        B, T, H, W, D = x.shape

        # Temporal attention: fold spatial dims into batch
        x_flat = einops.rearrange(x, "b t h w d -> (b h w) t d")

        # Compute Q, K, V for new frames
        qkv = self.qkv_proj(x_flat)
        q_new, k_new, v_new = qkv.chunk(3, dim=-1)
        q_new = self.norm_q(q_new)
        k_new = self.norm_k(k_new)

        q_new = einops.rearrange(q_new, "B T (head d) -> B head T d", head=self.num_heads)
        k_new = einops.rearrange(k_new, "B T (head d) -> B head T d", head=self.num_heads)
        v_new = einops.rearrange(v_new, "B T (head d) -> B head T d", head=self.num_heads)

        # Read cached K, V
        end_idx = ext_cache["end_idx"]
        if end_idx > 0:
            k_cached = ext_cache["k"][:, :, :end_idx, :]
            v_cached = ext_cache["v"][:, :, :end_idx, :]
            k_full = torch.cat([k_cached, k_new], dim=2)
            v_full = torch.cat([v_cached, v_new], dim=2)
        else:
            k_full = k_new
            v_full = v_new

        # Apply RoPE with full sequence length
        full_seq_len = k_full.shape[2]
        sequence_shape = (full_seq_len,)
        q_rot, k_rot = apply_rope_nd(q_new, k_full, sequence_shape, rotary_type=self.rotary_type)

        # Attention (not causal — we manually control what's in the cache)
        out = F.scaled_dot_product_attention(q_rot, k_rot, v_full, is_causal=False)
        out = einops.rearrange(out, "B head seq d -> B seq (head d)")
        out = self.out_proj(out)

        # Unfold spatial dims
        out = einops.rearrange(out, "(b h w) t d -> b t h w d", h=H, w=W)
        return out

    def write_to_external_cache(
        self,
        x: torch.Tensor,
        ext_cache: dict,
    ) -> None:
        """
        Compute K, V for the given frames and write them to the external cache.
        This is called during the cache update pass (Algorithm 1, Line 13)
        where we run the model at t=0 on clean frames to store their representations.

        All writes are detached — no gradient flows through the cache.

        Args:
            x: [B, T, H, W, D] clean frame representations (post-patchify, post-adaLN)
            ext_cache: the layer's cache dict
        """
        B, T, H, W, D = x.shape
        x_flat = einops.rearrange(x, "b t h w d -> (b h w) t d")

        qkv = self.qkv_proj(x_flat)
        _, k_new, v_new = qkv.chunk(3, dim=-1)
        k_new = self.norm_k(k_new)

        k_new = einops.rearrange(k_new, "B T (head d) -> B head T d", head=self.num_heads)
        v_new = einops.rearrange(v_new, "B T (head d) -> B head T d", head=self.num_heads)

        end_idx = ext_cache["end_idx"]
        new_end = end_idx + T

        ext_cache["k"][:, :, end_idx:new_end, :] = k_new.detach()
        ext_cache["v"][:, :, end_idx:new_end, :] = v_new.detach()
        ext_cache["end_idx"] = new_end

    # Original internal cache path

    def _forward_with_cache(
        self, x: torch.Tensor, start_frame: int, cache_idx: int, cache_type: str = "cond"
    ) -> torch.Tensor:
        B, T, H, W, D = x.shape
        x = einops.rearrange(x, "b t h w d -> (b h w) t d")

        k_cache = self.k_cache_cond if cache_type == "cond" else self.k_cache_null
        v_cache = self.v_cache_cond if cache_type == "cond" else self.v_cache_null

        qkv_new = self.qkv_proj(x)
        q_new, k_new, v_new = qkv_new.chunk(3, dim=-1)
        q = self.norm_q(q_new)
        k_new = self.norm_k(k_new)
        q = einops.rearrange(q, "B T (head d) -> B head T d", head=self.num_heads)
        k_new = einops.rearrange(k_new, "B T (head d) -> B head T d", head=self.num_heads)
        v_new = einops.rearrange(v_new, "B T (head d) -> B head T d", head=self.num_heads)

        if k_cache is not None:
            rel_start = start_frame - self.cache_start
            rel_end = cache_idx - self.cache_start
            if rel_start < 0 or rel_end > k_cache.shape[2]:
                raise ValueError(
                    f"Cache bounds error: cache[{rel_start}:{rel_end}] "
                    f"but cache len {k_cache.shape[2]}"
                )
            k_cached = k_cache[:, :, rel_start:rel_end, :]
            v_cached = v_cache[:, :, rel_start:rel_end, :]
            k = torch.cat([k_cached, k_new], dim=2)
            v = torch.cat([v_cached, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        if cache_type == "cond":
            self.k_cache_cond = k.detach()
            self.v_cache_cond = v.detach()
        else:
            self.k_cache_null = k.detach()
            self.v_cache_null = v.detach()

        self.cache_start = start_frame
        self.cache_end = start_frame + k.shape[2]

        sequence_shape = (k.shape[2],)
        q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = einops.rearrange(out, "B head seq d -> B seq (head d)")
        out = self.out_proj(out)
        out = einops.rearrange(out, "(b h w) t d -> b t h w d", h=H, w=W)
        return out

    def forward(
        self,
        x: torch.Tensor,
        cache_idx: int | None = None,
        start_frame: int | None = None,
        cache_type: str = "cond",
        ext_cache: Optional[dict] = None,
    ):
        """
        Args:
            x: [B, T, H, W, D]
            cache_idx, start_frame, cache_type: for internal cache path
            ext_cache: for external cache path (Self-Forcing training)
        """
        B, T, H, W, D = x.shape

        # External cache path (Self-Forcing training)
        if ext_cache is not None and self.attention_type == AttentionType.TEMPORAL:
            return self._forward_with_external_cache(x, ext_cache)

        # Internal cache path (original inference)
        use_internal_cache = (
            cache_idx is not None
            and start_frame is not None
            and cache_idx > start_frame
            and self.attention_type == AttentionType.TEMPORAL
        )
        if use_internal_cache:
            return self._forward_with_cache(x, start_frame, cache_idx, cache_type)

        # No cache path (standard forward)
        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "b t h w d -> (b h w) t d")
        else:
            raise NotImplementedError(f"invalid attention type: {self.attention_type}")
        sequence_shape = x.shape[1:-1]

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = einops.rearrange(q, "B ... (head d) -> B head ... d", head=self.num_heads)
        k = einops.rearrange(k, "B ... (head d) -> B head ... d", head=self.num_heads)
        v = einops.rearrange(v, "B ... (head d) -> B head ... d", head=self.num_heads)

        q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)
        q = einops.rearrange(q, "B head ... d -> B head (...) d")
        k = einops.rearrange(k, "B head ... d -> B head (...) d")
        v = einops.rearrange(v, "B head ... d -> B head (...) d")

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        x = einops.rearrange(x, "B head seq d -> B seq (head d)")
        x = self.out_proj(x)

        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "(b t) (h w) d -> b t h w d", t=T, h=H, w=W)
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "(b h w) t d -> b t h w d", h=H, w=W)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: AttentionType,
        rotary_type: RotaryType,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6, bias=True)
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim,
            num_heads,
            is_causal=is_causal,
            attention_type=attention_type,
            rotary_type=rotary_type,
        )
        self.ffwd = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache_idx: int | None = None,
        start_frame: int | None = None,
        cache_type: str = "cond",
        ext_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(6, dim=-1)
        x = x + self.attn(
            self.norm1(x) * (1 + m[1]) + m[0],
            cache_idx, start_frame, cache_type,
            ext_cache=ext_cache,
        ) * m[2]
        x = x + self.ffwd(self.norm2(x) * (1 + m[4]) + m[3]) * m[5]
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, H, W, D = x.shape
        h = self.num_heads
        q_in = einops.rearrange(x, "b t h w d -> (b t) (h w) d")
        kv_in = einops.repeat(context, "b lc d -> (b t) lc d", t=T)

        q = self.q_proj(q_in)
        k, v = self.kv_proj(kv_in).chunk(2, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.view(B * T, H * W, h, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B * T, kv_in.size(1), h, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B * T, kv_in.size(1), h, self.head_dim).transpose(1, 2).contiguous()

        attn_mask = None
        if mask is not None:
            key_pad = ~mask.bool()
            key_pad = einops.repeat(key_pad, "b lc -> (b t) lc", t=T).to(kv_in.device)
            attn_mask = key_pad.unsqueeze(1).unsqueeze(1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        x = x.transpose(1, 2).contiguous().view(B * T, H * W, self.dim)
        x = self.out_proj(x)
        x = einops.rearrange(x, "(b t) (h w) d -> b t h w d", b=B, t=T, h=H, w=W)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 3, bias=True)
        )
        self.ffwd = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, cross_context: dict[str, torch.Tensor]):
        B, T, H, W, C = x.shape
        q_in = self.norm1(x)
        cross_out = self.cross_attn(q_in, cross_context["seq"], cross_context.get("mask"))
        x = x + cross_out

        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(3, dim=-1)
        y = self.ffwd(self.norm2(x) * (1 + m[1]) + m[0])
        x = x + y * m[2]
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_config: dict[AttentionType, RotaryType] | None = None,
    ) -> None:
        super().__init__()
        self.s_block = SelfAttentionBlock(
            dim, num_heads,
            is_causal=False,
            attention_type=AttentionType.SPATIAL,
            rotary_type=rope_config[AttentionType.SPATIAL] if rope_config else RotaryType.STANDARD,
        )
        self.t_block = SelfAttentionBlock(
            dim, num_heads,
            is_causal=True,
            attention_type=AttentionType.TEMPORAL,
            rotary_type=rope_config[AttentionType.TEMPORAL] if rope_config else RotaryType.STANDARD,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache_idx: int | None = None,
        start_frame: int | None = None,
        cache_type: str = "cond",
        ext_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        x = self.s_block(x, c)
        x = self.t_block(x, c, cache_idx, start_frame, cache_type, ext_cache=ext_cache)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        action_dim: int = 256,
        max_frames: int = 16,
        rope_config: dict[AttentionType, RotaryType] | None = None,
        external_cond_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.action_dim = action_dim
        self.external_cond_dropout_prob = external_cond_dropout_prob

        self.x_proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(256, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.action_embedder = nn.Linear(action_dim, dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, rope_config) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(dim, patch_size, in_channels)
        self.max_frames = max_frames
        self.initialize_weights()

    def timestep_embedding(self, t: torch.Tensor, dim: int = 256, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_proj.bias, 0)

        nn.init.normal_(self.timestep_mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.s_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x = self.x_proj(x)
        x = einops.rearrange(x, "(b t) d h w -> b t h w d", t=T)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            x, "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.patch_size, p2=self.patch_size, c=self.in_channels,
        )

    def get_null_action(self, action: torch.Tensor) -> torch.Tensor:
        null_action = torch.zeros_like(action)
        null_action[..., -1] = 1
        return null_action

    def get_adaln_cond(self, t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, T = t.shape
        t = einops.rearrange(t, "b t -> (b t)")
        t_freq = self.timestep_embedding(t)
        c = self.timestep_mlp(t_freq)
        c = einops.rearrange(c, "(b t) d -> b t d", t=T)
        if self.training and self.external_cond_dropout_prob > 0:
            should_drop = torch.rand((B, 1, 1), device=action.device) < self.external_cond_dropout_prob
            null_action = self.get_null_action(action)
            action = torch.where(should_drop, null_action, action)
        c += self.action_embedder(action)
        return c

    def clear_kv_cache(self):
        for block in self.blocks:
            block.t_block.attn.clear_cache()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        cache_idx: int | None = None,
        start_frame: int | None = None,
        cache_type: str = "cond",
        external_kv_cache: Optional[list[dict]] = None,
    ) -> torch.Tensor:
        """
        Diffusion model forward pass.

        Args:
            x: [B, T, H, W, C] input latent frames
            t: [B, T] timesteps per frame
            action: [B, T, D] action conditioning per frame
            cache_idx, start_frame, cache_type: for internal cache (original inference)
            external_kv_cache: list of per-layer cache dicts (Self-Forcing training)
                When provided, the model uses external cache instead of internal cache.
                The input x should contain ONLY the new frames (not the full sequence).
        Returns:
            output: [B, T, H, W, C] predicted v
        """
        B, T, H, W, C = x.shape

        x = self.patchify(x)
        c = self.get_adaln_cond(t, action)

        if external_kv_cache is not None:
            # External cache path: x already contains only new frames
            for layer_idx, block in enumerate(self.blocks):
                x = block(x, c, ext_cache=external_kv_cache[layer_idx])
        else:
            # Original path: handle internal cache slicing
            has_cache = any(
                block.t_block.attn.k_cache_cond is not None if cache_type == "cond"
                else block.t_block.attn.k_cache_null is not None
                for block in self.blocks
            )
            if cache_idx is not None and start_frame is not None and cache_idx > start_frame and has_cache:
                start_rel = cache_idx - start_frame
                x = x[:, start_rel:, ...]
                c = c[:, start_rel:, ...]
                T = x.shape[1]

            for block in self.blocks:
                x = block(x, c, cache_idx, start_frame, cache_type)

        x = self.final_layer(x, c)
        x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)
        x = einops.rearrange(x, "(b t) h w c -> b t h w c", t=T)
        return x

    def forward_and_cache(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        external_kv_cache: list[dict],
    ) -> torch.Tensor:
        """
        Forward pass that ALSO writes K, V into the external cache.

        This is used for:
          1. The initial frame seeding (Algorithm 1: cache context)
          2. The cache update pass after each frame (Algorithm 1, Line 13)

        The forward computation itself runs normally. After each temporal
        attention layer, the new frame's K, V are written to the cache.

        Args:
            x: [B, T, H, W, C] frames to process and cache
            t: [B, T] timesteps (typically 0 for clean frames)
            action: [B, T, D] actions
            external_kv_cache: list of per-layer cache dicts
        Returns:
            output: [B, T, H, W, C] model output (usually discarded)
        """
        B, T, H, W, C = x.shape

        x_hidden = self.patchify(x)
        c = self.get_adaln_cond(t, action)

        for layer_idx, block in enumerate(self.blocks):
            # Run spatial attention (no caching needed)
            x_hidden = block.s_block(x_hidden, c)

            # For temporal attention: compute output using external cache
            _, _, Hp, Wp, _ = x_hidden.shape
            m = block.t_block.adaLN_modulation(c)
            m = einops.repeat(m, "b t d -> b t h w d", h=Hp, w=Wp).chunk(6, dim=-1)

            attn_input = block.t_block.norm1(x_hidden) * (1 + m[1]) + m[0]

            # Run attention with external cache (reads + attends)
            attn_out = block.t_block.attn._forward_with_external_cache(
                attn_input, external_kv_cache[layer_idx]
            )
            x_hidden = x_hidden + attn_out * m[2]

            # Write the NEW frame's K, V into cache (detached)
            block.t_block.attn.write_to_external_cache(
                attn_input, external_kv_cache[layer_idx]
            )

            # FFN
            x_hidden = x_hidden + block.t_block.ffwd(
                block.t_block.norm2(x_hidden) * (1 + m[4]) + m[3]
            ) * m[5]

        x_out = self.final_layer(x_hidden, c)
        x_out = einops.rearrange(x_out, "b t h w d -> (b t) h w d")
        x_out = self.unpatchify(x_out)
        x_out = einops.rearrange(x_out, "(b t) h w c -> b t h w c", t=T)
        return x_out