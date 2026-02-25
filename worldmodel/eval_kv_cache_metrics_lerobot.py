import einops
import os
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

# Local imports
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Model imports 
from dmd_world_model import DMDWorldModel 


try:
    from eval_utils import (
        VideoEvaluator,
        save_video_mp4,
        save_comparison_gif,
        save_metadata,
        plot_mse_over_time,
        create_output_structure,
        pad_actions
    )
    EVAL_UTILS_AVAILABLE = True
except ImportError as e:
    EVAL_UTILS_AVAILABLE = False
    print(f"WARNING: eval_utils not found. Metrics computation will be disabled.")
    print(f"Import error details: {e}")

logger = logging.getLogger(__name__)


# Configuration 
@dataclass
class Config:
    # Paths
    data_dir: str = "/projects/work/yang-lab/projects/pretrain_world_model/final_datasets"
    output_gif_path: str = "comparison_sample.gif"

    # Data
    subset_name: str = "oxe_converting2"
    n_frames: int = 10
    action_dim: int = 21
    input_h: int = 256
    input_w: int = 256
    dataset_fps = 10
    batch_size: int = 1
    num_workers: int = 4

    # Model Architecture
    patch_size: int = 2
    model_dim: int = 2304
    layers: int = 32
    heads: int = 16
    external_cond_dropout_prob: float = 0.1

    # Diffusion & Inference
    timesteps: int = 1_000
    sampling_timesteps: int = 4
    cfg: float = 3.0
    n_context_frames: int = 1
    window_len: Optional[int] = None
    horizon: int = 1
    chunk_size: int = 1

    # DMD-specific (only used when model_type="dmd")
    num_denoising_steps: int = 4
    max_cache_frames: int = 15
    context_noise: int = 0

    # Which model to use: "pretrained" or "dmd"
    model_type: str = "dmd"

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 42


def update_config_from_dict(cfg: Config, cfg_dict: Dict[str, Any]) -> Config:
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            logger.debug(f"Ignoring unknown config key '{k}'")
    return cfg


# Model Setup
def setup_models(ckpt_path: str, config: Config, norm_stats_path=None):
    """
    Initialize the world model.

    Uses DMDWorldModel if config.model_type == "dmd", else pretrained WorldModel.
    Both have the same interface: reset() and generate_chunk().
    """
    logger.info(f"Initializing model (type={config.model_type})...")

    if config.model_type == "dmd":
        world_model = DMDWorldModel(
            checkpoint_path=ckpt_path,
            config=config,
            use_ema=True,
            context_noise=config.context_noise,
            max_cache_frames=config.max_cache_frames,
        )
    else:
        # Pretrained model (original pyramid-scheduling WorldModel)
        world_model = WorldModel(
            checkpoint_path=ckpt_path,
            config=config,
            norm_stats_path=norm_stats_path,
        )

    return world_model


# Visualization Utilities
def tensor_to_video_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a channel-last video tensor to uint8 numpy array for visualization.
    Input: (H, W, C) or (T, H, W, C), values in [0, 1].
    Output: (T, H, W, C) in [0, 255].
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    video = tensor.cpu().detach().float().clamp(0, 1).numpy()
    return (video * 255).astype(np.uint8)


def visualize_input_static(observation_tensor: torch.Tensor, n_frames: int = 5):
    frames = tensor_to_video_uint8(observation_tensor)
    n_cols = min(n_frames, frames.shape[0])
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))
    if n_cols == 1:
        axes = [axes]
    for i in range(n_cols):
        axes[i].imshow(frames[i])
        axes[i].axis("off")
        axes[i].set_title(f"Input Frame {i}")
    plt.tight_layout()
    plt.show()


# Main eval function
def eval_checkpoint_single_dataset(
    ckpt_path: str,
    config: Dict[str, Any],
    *,
    device: Optional[str] = None,
    output_base_dir: str = "/scratch/nl2752/eval_outputs_clothfolding",
    split: str = "test",
    max_samples: Optional[int] = None,
    compute_metrics: bool = True,
    save_videos: bool = True,
    save_gifs: bool = True,
    fps: int = 8,
    logger_override: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger_override or logger

    cfg = Config()
    cfg = update_config_from_dict(cfg, config)
    if device is not None:
        cfg.device = device

    if cfg.seed is not None:
        log.info(f"Setting seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    subset_name = cfg.subset_name
    ckpt_leaf_raw = os.path.basename(os.path.normpath(ckpt_path))
    ckpt_exp = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(ckpt_path))))
    ckpt_leaf = f"{ckpt_leaf_raw}_cfg{cfg.cfg}"
    log.info(f"Experiment: {ckpt_exp}, Checkpoint: {ckpt_leaf}, Model type: {cfg.model_type}")

    # Initialize model
    world_model = setup_models(ckpt_path=ckpt_path, config=cfg, norm_stats_path=None)

    if compute_metrics and EVAL_UTILS_AVAILABLE:
        evaluator = VideoEvaluator(
            device=cfg.device,
            compute_lpips=True,
            compute_mse=True,
            compute_ssim=True,
            compute_fid=True,
        )
    else:
        evaluator = None
        if compute_metrics:
            log.warning("Metrics computation disabled (eval_utils not available)")

    ckpt_output_dir = os.path.join(output_base_dir, ckpt_exp, ckpt_leaf)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    log.info(f"Output directory: {ckpt_output_dir}")

    if EVAL_UTILS_AVAILABLE:
        output_paths = create_output_structure(output_base_dir, ckpt_exp, ckpt_leaf, subset_name)
    else:
        subset_output_dir = os.path.join(ckpt_output_dir, subset_name)
        os.makedirs(subset_output_dir, exist_ok=True)
        output_paths = {'root': subset_output_dir}

    # DataLoader
    log.info("Initializing LeRobot DataLoader...")
    deltas = [i * (1 / cfg.dataset_fps) for i in range(cfg.n_frames)]
    eval_dataset = MultiLeRobotDataset(
        repo_ids=[f"{subset_name}/validation"],
        root=cfg.data_dir,
        image_transforms=transforms.Resize((int(cfg.input_h), int(cfg.input_w))),
        delta_timestamps={
            "observation.image": deltas,
            "action": deltas,
        },
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    log.info(f"Validation dataset size: {len(eval_dataset)}")
    subset_metrics = []
    processed = 0

    for sample_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {subset_name}")):
        if max_samples and sample_idx >= max_samples:
            break

        video_name = f"lerobot_{sample_idx:06d}"

        x = batch["observation.image"]
        x = einops.rearrange(x, "b t c h w -> b t h w c")
        x = x.to(cfg.device)

        actions = batch["action"]
        assert actions.dim() == 3, f"Expected (B,T,D), got {actions.shape}"
        actions = pad_actions(actions, cfg.action_dim)
        actions = actions.to(cfg.device)

        with torch.no_grad():
            init_frame = x[:, 0]      # [B, H, W, C]
            action_seq = actions      # [B, T, action_dim]
            world_model.reset(init_frame)

            pred_frames = {}
            T = action_seq.shape[1]
            for t in range(T):
                action_t = action_seq[:, t]   # [B, D]
                for frame_idx, decoded in world_model.generate_chunk(action_t):
                    pred_frames[frame_idx] = decoded[0, 0].cpu()  # [H, W, C]

            pred_video = torch.stack(
                [pred_frames[i] for i in sorted(pred_frames.keys())],
                dim=0,
            )  # [T_pred, H, W, C]

        gt_video_uint8   = tensor_to_video_uint8(x[0])
        pred_video_uint8 = tensor_to_video_uint8(pred_video)

        safe_video_name = str(video_name).replace("/", "_").replace(" ", "_")
        sample_name = f"{safe_video_name}_{split}"

        if save_videos and 'videos_gt' in output_paths and 'videos_pred' in output_paths:
            try:
                save_video_mp4(gt_video_uint8,   os.path.join(output_paths['videos_gt'],   f"{sample_name}_gt.mp4"),   fps=fps)
                save_video_mp4(pred_video_uint8, os.path.join(output_paths['videos_pred'], f"{sample_name}_pred.mp4"), fps=fps)
            except Exception as e:
                log.warning(f"Failed to save videos: {e}")

        if save_gifs:
            try:
                gif_path = os.path.join(
                    output_paths.get('gifs', output_paths['root']),
                    f"{sample_name}.gif",
                )
                if EVAL_UTILS_AVAILABLE:
                    save_comparison_gif(gt_video_uint8, pred_video_uint8, gif_path, fps=fps, add_labels=True)
                else:
                    comparison = np.concatenate([gt_video_uint8, pred_video_uint8], axis=2)
                    imageio.mimsave(gif_path, comparison, fps=fps, loop=0)
            except Exception as e:
                log.warning(f"Failed to save GIF: {e}")

        sample_metrics = {}
        if compute_metrics and evaluator:
            try:
                sample_metrics = evaluator.evaluate_single(gt_video_uint8, pred_video_uint8)
                if 'metrics' in output_paths:
                    save_metadata(
                        os.path.join(output_paths['metrics'], f"{sample_name}_metrics.json"),
                        **sample_metrics,
                    )
                if 'mse_per_frame' in sample_metrics and 'plots' in output_paths:
                    plot_mse_over_time(
                        sample_metrics['mse_per_frame'],
                        os.path.join(output_paths['plots'], f"{sample_name}_mse.png"),
                        title=f"MSE over Time - {sample_name}",
                    )
                subset_metrics.append(sample_metrics)
            except Exception as e:
                log.warning(f"Failed to compute metrics: {e}")

        if 'metadata' in output_paths:
            try:
                save_metadata(
                    os.path.join(output_paths['metadata'], f"{sample_name}.json"),
                    sample_idx=sample_idx,
                    sample_name=sample_name,
                    subset_name=subset_name,
                    ckpt_exp=ckpt_exp,
                    ckpt_step=ckpt_leaf,
                    model_type=cfg.model_type,
                    n_frames=cfg.n_frames,
                    image_size=[cfg.input_h, cfg.input_w],
                    actions=actions[0].cpu().numpy().tolist(),
                )
            except Exception as e:
                log.warning(f"Failed to save metadata: {e}")

        processed += 1

    if compute_metrics and evaluator and subset_metrics and EVAL_UTILS_AVAILABLE:
        summary_path = os.path.join(output_paths['root'], 'summary.json')
        evaluator.save_summary(subset_metrics, summary_path, include_fid=True)
        log.info(f"Subset {subset_name} complete. Results: {output_paths['root']}")

    return {
        "subset_name": subset_name,
        "num_samples": processed,
        "output_dir": output_paths['root'] if output_paths else None,
        "summary_path": os.path.join(output_paths['root'], "summary.json"),
    }


def submit_eval_checkpoint_single_dataset(
    ckpt_path: str,
    config: Dict[str, Any],
    *,
    device: Optional[str] = None,
    output_base_dir: str = "/scratch/nl2752/cloth_folding_dmd_100_new_inference",
    split: str = "test",
    max_samples: Optional[int] = None,
    compute_metrics: bool = True,
    save_videos: bool = True,
    save_gifs: bool = True,
    fps: int = 8,
    job_name: str = "eval_ckpt",
    partition: str = "h200",
    account: str = "torch_pr_147_courant",
    time_limit: str = "4:00:00",
    mem: str = "200G",
    cpus_per_task: int = 12,
    gpus: int = 1,
    constraint: str = "h200",
    log_dir: str = "logs_cf_dmd_100_new_inference",
) -> str:
    import sys
    import pickle
    import subprocess
    import time
    import uuid

    os.makedirs(log_dir, exist_ok=True)
    args = dict(
        ckpt_path=ckpt_path, config=config, device=device,
        output_base_dir=output_base_dir, split=split,
        max_samples=max_samples, compute_metrics=compute_metrics,
        save_videos=save_videos, save_gifs=save_gifs, fps=fps,
    )

    job_id_str = f"{job_name.replace('/', '_')}_{uuid.uuid4()}"
    pkl_path   = os.path.abspath(os.path.join(log_dir, f"{job_id_str}_args.pkl"))
    with open(pkl_path, "wb") as f:
        pickle.dump(args, f)

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    runner_path  = os.path.abspath(os.path.join(log_dir, f"{job_id_str}_runner.py"))
    runner_content = f"""
import sys, os, pickle, logging
os.chdir("{current_dir}")
sys.path.insert(0, "{current_dir}")
from eval_kv_cache_metrics_lerobot import eval_checkpoint_single_dataset

def main():
    logging.basicConfig(level=logging.INFO)
    with open("{pkl_path}", "rb") as f:
        kwargs = pickle.load(f)
    results = eval_checkpoint_single_dataset(**kwargs)
    print("Done:", results)

if __name__ == "__main__":
    main()
"""
    with open(runner_path, "w") as f:
        f.write(runner_content)

    sbatch_path = os.path.abspath(os.path.join(log_dir, f"{job_id_str}.sbatch"))
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_id_str}_%j.out
#SBATCH --error={log_dir}/{job_id_str}_%j.err
#SBATCH --account={account}
#SBATCH --nodes=1 --tasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH --gres=gpu:l40s:1

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Date: $(date)"
export PYTHONPATH=$PYTHONPATH:{current_dir}
"{sys.executable}" "{runner_path}"
"""
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)

    result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Job submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    else:
        raise RuntimeError(f"sbatch failed: {result.stderr}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir = "/projects/work/yang-lab/users/yz12129"
    ckpt600m_dmd = "/scratch/nl2752/world-model-distillation/outputs/20260221_234405_dmd/dmd_ckpt_000010000.pt"

    subsets = [
        "converted_rollout_data/folding_rollout_jan15_ds1",
        "lerobot_dataset/fold_clothes_lerobot_split_realaction_214ep_ds1",
        "converted_rollout_data/folding_rollout_jan16_ds1",
        "converted_rollout_data/folding_other_teleop_demos",
    ]

    for subset_name in subsets:
        print(f"\n{'='*80}\nSubmitting eval for: {subset_name}\n{'='*80}")
        submit_eval_checkpoint_single_dataset(
            ckpt_path=ckpt600m_dmd,
            config={
                "data_dir":    data_dir,
                "subset_name": subset_name,
                "model_type":  "dmd",       
                "model_dim":   1024,
                "layers":      16,
                "heads":       16,
                "n_frames":    100,
                "max_cache_frames": 13,
            },
            max_samples=100,
            compute_metrics=True,
            save_videos=True,
            save_gifs=True,
            fps=10,
            job_name=f"eval_{subset_name}",
            time_limit="04:00:00",
        )