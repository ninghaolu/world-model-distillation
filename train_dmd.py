"""
DMD post-training for autoregressive world models via Self-Forcing.

Distills a pretrained multi-step DDPM DiT into a few-step generator using
the DMD2 framework with Self-Forcing autoregressive rollout.

Usage:
    # Single GPU
    python train_dmd.py --pretrained_checkpoint outputs/20250101_120000/ckpt_000100000.pt

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train_dmd.py \
        --pretrained_checkpoint outputs/20250101_120000/ckpt_000100000.pt \
        --batch_size 2 --n_frames 20

Key hyperparameters:
    --num_denoising_steps 4     Few-step denoising per frame (down from 10)
    --dfake_gen_update_ratio 2  Critic trains 2x per generator update
    --real_guidance_scale 1.5   CFG for teacher scoring
    --n_grad_frames 1           How many trailing frames carry generator gradients
"""

import fire
import os
import datetime
import logging
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import einops
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

import worldmodel.model_with_kv_cache
from worldmodel.model_with_kv_cache import DiT
from worldmodel.vae import VAE
from dmd.diffusion_utils import get_alphas_cumprod, get_denoising_step_list
from dmd.generation import self_forcing_rollout
from dmd.losses import compute_generator_loss, compute_critic_loss

import wandb
from utils import Logger, update_ema, requires_grad, init_distributed, pad_actions


def main(
    # Checkpoint
    pretrained_checkpoint: str = None,
    checkpoint_dir: Path | None = None,
    # Dataset (LeRobot)
    dataset_dir: Path = Path("sample_data"),
    subset_names: str = "bridge",
    input_h: int = 256,
    input_w: int = 256,
    n_frames: int = 20,  # kept for dataloader compat; overridden by n_total_frames for generation
    frame_skip: int = 1,
    action_dim: int = 21,
    num_workers: int = 16,
    dataset_fps: int = 10,
    # Training
    batch_size: int = 1,
    timesteps: int = 1_000,
    lr_generator: float = 2e-6,
    lr_critic: float = 4e-7,
    warmup_steps: int = 200,
    ema_decay: float = 0.99,
    max_train_steps: int = 100_000,
    weight_decay: float = 0.02,
    grad_clip_norm: float = 5.0,
    beta1: float = 0.0,
    beta2: float = 0.999,
    beta1_critic: float = 0.0,
    beta2_critic: float = 0.999,
    # DMD hyperparameters
    num_denoising_steps: int = 4,
    dfake_gen_update_ratio: int = 5,
    real_guidance_scale: float = 3.0,
    fake_guidance_scale: float = 0.0,
    gen_cfg: float = 1.0,
    min_timestep_pct: float = 0.02,         # for sampling t
    max_timestep_pct: float = 0.98,
    n_grad_frames: int = 1,
    num_frame_per_block: int = 1,
    # Rolling KV cache training (Section 3.4):
    # Generate n_total_frames but only score the last n_output_frames.
    # The early "warmup" frames build KV cache context and get evicted,
    # so the model learns to generate without seeing the first frame.
    # max_cache_frames should be < n_total_frames to trigger eviction.
    n_total_frames: int = 24,
    n_output_frames: int = 15,
    max_cache_frames: int = 15,
    # Architecture (must match pretrained model)
    patch_size: int = 2,
    model_dim: int = 1024,
    layers: int = 16,
    heads: int = 16,
    # Logging
    validate_every: int = 2000,
    log_every: int = 50,
    # Sampling / eval
    cfg: float = 3.0,
    # WandB
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_project: str | None = None,
) -> None:
    # Distributed setup 
    local_rank, rank, world_size, distributed = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    # Output dir
    if checkpoint_dir is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_dmd")
        checkpoint_dir = Path("outputs") / run_name
    else:
        checkpoint_dir = Path(checkpoint_dir)

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoint dir: {checkpoint_dir}")

    # Dataset (LeRobot)
    deltas = [i * (1 / dataset_fps) for i in range(n_total_frames)]

    if isinstance(subset_names, str):
        subset_names_list = subset_names.split(",") if "," in subset_names else [subset_names]
    else:
        subset_names_list = list(subset_names)

    train_dataset = MultiLeRobotDataset(
        repo_ids=[f"{s}/train" for s in subset_names_list],
        root=dataset_dir,
        image_transforms=transforms.Resize((input_h, input_w)),
        delta_timestamps={
            "observation.image": deltas,
            "action": deltas,
        },
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_iter = iter(train_loader)

    # VAE (frozen)
    vae = VAE().to(device)
    vae.eval()
    requires_grad(vae, False)

    # Build 3 DiTs 
    def make_dit():
        return DiT(
            in_channels=vae.vae.config.latent_channels,
            patch_size=patch_size,
            dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            action_dim=action_dim,
            max_frames=n_total_frames,
            external_cond_dropout_prob=0.0,  # no dropout during distillation
        )

    generator = make_dit().to(device)
    real_score = make_dit().to(device)
    fake_score = make_dit().to(device)

    # Load pretrained weights
    if pretrained_checkpoint is not None:
        ckpt_path = Path(pretrained_checkpoint)
        if rank == 0:
            logging.info(f"Loading pretrained weights from {ckpt_path}")

        data = torch.load(str(ckpt_path), map_location="cpu")

        # Support both "ema" and "model" keys
        if "ema" in data:
            state_dict = data["ema"]
        elif "model" in data:
            state_dict = data["model"]
        else:
            state_dict = data

        generator.load_state_dict(state_dict, strict=True)
        real_score.load_state_dict(state_dict, strict=True)
        fake_score.load_state_dict(state_dict, strict=True)
        del state_dict, data
        torch.cuda.empty_cache()

        if rank == 0:
            logging.info("Pretrained weights loaded into generator, real_score, fake_score")
    else:
        if rank == 0:
            logging.warning("No pretrained_checkpoint! Models are randomly initialized.")

    # Freeze teacher
    requires_grad(real_score, False)
    real_score.eval()

    # Generator + critic are trainable
    requires_grad(generator, True)
    requires_grad(fake_score, True)

    # EMA of generator
    generator_ema = deepcopy(generator).to(device)
    requires_grad(generator_ema, False)
    update_ema(generator_ema, generator, 0.0)  # exact copy

    if rank == 0:
        n_params = sum(p.numel() for p in generator.parameters())
        logging.info(f"Params per model: {n_params:,}  (x3 = {n_params * 3:,} total)")

    # DDP wrapping
    if distributed:
        generator = torch.nn.parallel.DistributedDataParallel(
            generator, device_ids=[local_rank], output_device=local_rank,
        )
        fake_score = torch.nn.parallel.DistributedDataParallel(
            fake_score, device_ids=[local_rank], output_device=local_rank,
        )
        # real_score is frozen, no DDP needed
        gen_no_ddp = generator.module
        fake_no_ddp = fake_score.module
    else:
        gen_no_ddp = generator
        fake_no_ddp = fake_score

    # Noise schedule and denoising steps 
    alphas_cumprod = get_alphas_cumprod(timesteps).to(device)
    denoising_step_list = get_denoising_step_list(timesteps, num_denoising_steps)
    min_timestep = int(min_timestep_pct * timesteps)
    max_timestep = int(max_timestep_pct * timesteps)

    if rank == 0:
        logging.info(f"Denoising step list ({num_denoising_steps} steps): {denoising_step_list}")
        logging.info(f"Timestep range for scoring: [{min_timestep}, {max_timestep})")
        logging.info(
            f"Rolling cache training: generate {n_total_frames} frames, "
            f"score last {n_output_frames}, cache holds {max_cache_frames}. "
            f"First frame evicted after frame {max_cache_frames}."
        )

    # Optimizers
    gen_opt = optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=lr_generator, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    critic_opt = optim.AdamW(
        [p for p in fake_score.parameters() if p.requires_grad],
        lr=lr_critic, weight_decay=weight_decay, betas=(beta1_critic, beta2_critic),
    )

    # LR warmup 
    def get_warmup_lr(step, base_lr):
        if warmup_steps <= 0:
            return base_lr
        if step < warmup_steps:
            return 1e-7 + (base_lr - 1e-7) * step / warmup_steps
        return base_lr

    def set_lr(optimizer, new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    # Resume
    train_steps = 0
    ckpts = sorted(checkpoint_dir.glob("dmd_ckpt_*.pt")) if checkpoint_dir.exists() else []
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
        data = torch.load(latest, map_location=device)
        gen_no_ddp.load_state_dict(data["generator"])
        fake_no_ddp.load_state_dict(data["fake_score"])
        generator_ema.load_state_dict(data["generator_ema"])
        gen_opt.load_state_dict(data["gen_optimizer"])
        critic_opt.load_state_dict(data["critic_optimizer"])
        train_steps = int(data.get("step", 0))
        if rank == 0:
            logging.info(f"Resumed from {latest} at step {train_steps}")
        del data

    # WandB
    if rank == 0 and wandb is not None:
        run_name = wandb_name or os.environ.get("SLURM_JOB_NAME")
        wandb.init(
            entity=wandb_entity or "hand2gripper",
            project=wandb_project or "dmd-distillation",
            config={
                "n_total_frames": n_total_frames,
                "n_output_frames": n_output_frames,
                "max_cache_frames": max_cache_frames,
                "batch_size": batch_size,
                "lr_generator": lr_generator,
                "lr_critic": lr_critic,
                "num_denoising_steps": num_denoising_steps,
                "denoising_step_list": denoising_step_list,
                "dfake_gen_update_ratio": dfake_gen_update_ratio,
                "real_guidance_scale": real_guidance_scale,
                "n_grad_frames": n_grad_frames,
                "max_train_steps": max_train_steps,
                "world_size": world_size,
            },
            name=run_name,
            reinit=True,
            dir="/tmp/wandb",
            save_code=False,
        )
        logger = Logger(use_wandb=True)
    else:
        logger = Logger(use_wandb=False)

    # Training loop
    # grad_frames_from is in OUTPUT space (0-indexed within n_output_frames)
    # e.g., n_output_frames=20, n_grad_frames=1 → grad on output frame 19 only
    grad_frames_from = max(0, n_output_frames - n_grad_frames)

    running_gen_loss = 0.0
    running_critic_loss = 0.0
    num_gen_batches = 0
    num_critic_batches = 0

    pbar = tqdm(total=max_train_steps, desc="DMD Training") if rank == 0 else None
    if pbar is not None:
        pbar.n = train_steps
        pbar.refresh()

    while train_steps < max_train_steps:
        train_gen = (train_steps % dfake_gen_update_ratio == 0)

        # LR warmup 
        if train_steps < warmup_steps:
            set_lr(gen_opt, get_warmup_lr(train_steps, lr_generator))
            set_lr(critic_opt, get_warmup_lr(train_steps, lr_critic))

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["observation.image"]
        x = einops.rearrange(x, "b t c h w -> b t h w c").to(device)
        actions = pad_actions(batch["action"], action_dim).to(device)

        # VAE encode the first frame (context)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                initial_latent = vae.encode(x[:, :1])  # [B, 1, H, W, C_lat]

        initial_action = actions[:, :1]  # [B, 1, D]

        # Generator update
        if train_gen:
            generator.train()
            gen_opt.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Self-Forcing rollout (Algorithm 1)
                # Generate n_total_frames but only return last n_output_frames
                generated = self_forcing_rollout(
                    generator=gen_no_ddp,
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    actions=actions,
                    denoising_step_list=denoising_step_list,
                    alphas_cumprod=alphas_cumprod,
                    n_total_frames=n_total_frames,
                    n_output_frames=n_output_frames,
                    max_cache_frames=max_cache_frames,
                    grad_frames_from=grad_frames_from,
                    num_frame_per_block=num_frame_per_block,
                )       # [B, n_output_frames, H, W, C]

                # DMD loss on the output window
                # Actions for the output window: last n_output_frames of the full sequence
                output_actions = actions[:, n_total_frames - n_output_frames:]

                # DMD loss on all generated frames (including context for scoring)
                gen_loss, gen_log = compute_generator_loss(
                    generated_video=generated,
                    actions=output_actions,
                    real_score=real_score,
                    fake_score=fake_no_ddp,
                    alphas_cumprod=alphas_cumprod,
                    min_timestep=min_timestep,
                    max_timestep=max_timestep,
                    real_guidance_scale=real_guidance_scale,
                    fake_guidance_scale=fake_guidance_scale,
                )

            if not gen_loss.requires_grad:
                # print generator parameters state
                for name, param in generator.named_parameters():
                    if param.requires_grad and param.grad is None:
                         pass # Expected before backward
                    if not param.requires_grad:
                        print(f"[DEBUG] Parameter {name} does not require grad!", flush=True)
            else:
                gen_loss.backward()

            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)
            gen_opt.step()

            # EMA update
            update_ema(generator_ema, gen_no_ddp, ema_decay)

            running_gen_loss += gen_loss.detach().item()
            num_gen_batches += 1

            # Print score diagnostics (first step only, then every log_every)
            if rank == 0 and (train_steps == 0 or train_steps % log_every == 0):
                def _v(key):
                    v = gen_log.get(key, 0)
                    return v.item() if hasattr(v, 'item') else v
                print(
                    f"[step {train_steps}] gen_loss={gen_loss.item():.6f}  "
                    f"real_x0: mean={_v('real_score_x0_mean'):.4f} std={_v('real_score_x0_std'):.4f}  "
                    f"fake_x0: mean={_v('fake_score_x0_mean'):.4f} std={_v('fake_score_x0_std'):.4f}  "
                    f"score_diff_l2={_v('score_diff_l2'):.4f}  "
                    f"kl_grad_norm={_v('dmd_gradient_norm'):.4f}  "
                    f"gen_video: mean={_v('gen_video_mean'):.4f} std={_v('gen_video_std'):.4f}",
                    flush=True,
                )

        # Critic update (every step)
        fake_score.train()
        critic_opt.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate without grad for critic training
            with torch.no_grad():
                gen_for_critic = self_forcing_rollout(
                    generator=gen_no_ddp,
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    actions=actions,
                    denoising_step_list=denoising_step_list,
                    alphas_cumprod=alphas_cumprod,
                    n_total_frames=n_total_frames,
                    n_output_frames=n_output_frames,
                    max_cache_frames=max_cache_frames,
                    grad_frames_from=None,  # no grad
                    num_frame_per_block=num_frame_per_block, 
                )

            # Actions for the output window
            output_actions_critic = actions[:, n_total_frames - n_output_frames:]

            critic_loss, critic_log = compute_critic_loss(
                generated_video=gen_for_critic.detach(),
                actions=output_actions_critic,
                fake_score=fake_score,
                alphas_cumprod=alphas_cumprod,
                min_timestep=min_timestep,
                max_timestep=max_timestep,
            )

        critic_loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(fake_score.parameters(), grad_clip_norm)
        critic_opt.step()

        running_critic_loss += critic_loss.detach().item()
        num_critic_batches += 1


        # Logging
        if train_steps % log_every == 0 and rank == 0:
            log_dict = {}
            if num_critic_batches > 0:
                log_dict["critic_loss"] = running_critic_loss / num_critic_batches
            if num_gen_batches > 0:
                log_dict["generator_loss"] = running_gen_loss / num_gen_batches
            if train_gen and gen_log:
                for k, v in gen_log.items():
                    log_dict[k] = v.item() if hasattr(v, 'item') else v

            log_dict["lr_generator"] = gen_opt.param_groups[0]["lr"]
            log_dict["lr_critic"] = critic_opt.param_groups[0]["lr"]

            for k, v in log_dict.items():
                logger.log_scalar(f"train/{k}", v, train_steps)

            if pbar is not None:
                pbar.set_postfix({
                    "g": f"{log_dict.get('generator_loss', 0):.4f}",
                    "c": f"{log_dict.get('critic_loss', 0):.4f}",
                })

            running_gen_loss = 0.0
            running_critic_loss = 0.0
            num_gen_batches = 0
            num_critic_batches = 0

        # MEMORY CLEANUP
        if train_steps % 20 == 0:
            torch.cuda.empty_cache()

        # CHECKPOINT
        if train_steps > 0 and train_steps % validate_every == 0 and rank == 0:
            ckpt_path = checkpoint_dir / f"dmd_ckpt_{train_steps:09d}.pt"
            torch.save(
                {
                    "generator": gen_no_ddp.state_dict(),
                    "generator_ema": generator_ema.state_dict(),
                    "fake_score": fake_no_ddp.state_dict(),
                    "gen_optimizer": gen_opt.state_dict(),
                    "critic_optimizer": critic_opt.state_dict(),
                    "step": train_steps,
                    "config": {
                        "n_total_frames": n_total_frames,
                        "n_output_frames": n_output_frames,
                        "max_cache_frames": max_cache_frames,
                        "num_denoising_steps": num_denoising_steps,
                        "denoising_step_list": denoising_step_list,
                        "real_guidance_scale": real_guidance_scale,
                    },
                },
                ckpt_path,
            )
            logging.info(f"Saved checkpoint to {ckpt_path}")

            # Keep only latest 5
            ckpts = sorted(checkpoint_dir.glob("dmd_ckpt_*.pt"))
            for old in ckpts[:-5]:
                logging.info(f"Removing old checkpoint {old}")
                os.remove(old)

        train_steps += 1
        if pbar is not None:
            pbar.update(1)

    # Cleanup 
    if distributed:
        dist.destroy_process_group()
    if wandb is not None and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)