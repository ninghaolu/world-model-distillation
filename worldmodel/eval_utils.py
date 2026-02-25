"""
Evaluation Utilities for World Model Assessment

Provides comprehensive metrics for evaluating world models:
- LPIPS: Learned Perceptual Image Patch Similarity
- MSE: Mean Squared Error (pixel-level)
- SSIM: Structural Similarity Index
- FID: Fréchet Inception Distance (for video distribution quality)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import lpips
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


# Video I/O Utilities
def save_video_mp4(
    video: np.ndarray,
    output_path: str,
    fps: int = 8,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
) -> None:
    """Save video as MP4 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=codec,
        pixelformat=pixel_format,
        quality=8,
    )
    
    for frame in video:
        writer.append_data(frame)
    writer.close()


def save_comparison_gif(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    output_path: str,
    fps: int = 8,
    add_labels: bool = True,
) -> None:
    """Save side-by-side comparison GIF (GT | Pred)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    comparison = np.concatenate([gt_video, pred_video], axis=2)
    
    if add_labels:
        labeled_frames = []
        for frame in comparison:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            
            h, w = frame.shape[:2]
            draw.text((10, 10), "Ground Truth", fill=(255, 255, 255), font=font)
            draw.text((w//2 + 10, 10), "Generated", fill=(255, 255, 255), font=font)
            
            labeled_frames.append(np.array(img))
        comparison = np.array(labeled_frames)
    
    imageio.mimsave(output_path, comparison, fps=fps, loop=0)


def save_metadata(output_path: str, **kwargs) -> None:
    """Save metadata as JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_data = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_data[key] = value.cpu().numpy().tolist()
        else:
            serializable_data[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


# Metric Computation: LPIPS
class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity."""
    
    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
        logger.info(f"Initialized LPIPS with {net} backbone")
    
    def compute(
        self,
        gt_video: Union[np.ndarray, torch.Tensor],
        pred_video: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Compute LPIPS between ground truth and predicted videos."""
        if isinstance(gt_video, np.ndarray):
            gt_video = torch.from_numpy(gt_video).float()
        if isinstance(pred_video, np.ndarray):
            pred_video = torch.from_numpy(pred_video).float()
        
        if gt_video.max() > 1:
            gt_video = gt_video / 255.0
        if pred_video.max() > 1:
            pred_video = pred_video / 255.0
        
        gt_video = gt_video.permute(0, 3, 1, 2).to(self.device)
        pred_video = pred_video.permute(0, 3, 1, 2).to(self.device)
        
        if normalize:
            gt_video = gt_video * 2 - 1
            pred_video = pred_video * 2 - 1
        
        lpips_values = []
        with torch.no_grad():
            for gt_frame, pred_frame in zip(gt_video, pred_video):
                gt_frame = gt_frame.unsqueeze(0)
                pred_frame = pred_frame.unsqueeze(0)
                lpips_val = self.loss_fn(gt_frame, pred_frame).item()
                lpips_values.append(lpips_val)
        
        return {
            'lpips_mean': float(np.mean(lpips_values)),
            'lpips_std': float(np.std(lpips_values)),
            'lpips_per_frame': lpips_values,
        }



# Metric Computation: MSE
def compute_mse(
    gt_video: Union[np.ndarray, torch.Tensor],
    pred_video: Union[np.ndarray, torch.Tensor],
    per_frame: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """Compute Mean Squared Error between videos."""
    if isinstance(gt_video, torch.Tensor):
        gt_video = gt_video.cpu().numpy()
    if isinstance(pred_video, torch.Tensor):
        pred_video = pred_video.cpu().numpy()
    
    if gt_video.max() > 1:
        gt_video = gt_video / 255.0
    if pred_video.max() > 1:
        pred_video = pred_video / 255.0
    
    mse_per_frame = []
    for gt_frame, pred_frame in zip(gt_video, pred_video):
        mse = np.mean((gt_frame - pred_frame) ** 2)
        mse_per_frame.append(float(mse))
    
    result = {
        'mse_mean': float(np.mean(mse_per_frame)),
        'mse_std': float(np.std(mse_per_frame)),
    }
    
    if per_frame:
        result['mse_per_frame'] = mse_per_frame
    
    return result


def plot_mse_over_time(
    mse_per_frame: List[float],
    output_path: str,
    title: str = "MSE over Time",
) -> None:
    """Plot MSE over time."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(mse_per_frame, label='MSE', linewidth=2)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()



# Metric Computation: SSIM
def compute_ssim(
    gt_video: Union[np.ndarray, torch.Tensor],
    pred_video: Union[np.ndarray, torch.Tensor],
    per_frame: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """Compute Structural Similarity Index between videos."""
    if isinstance(gt_video, torch.Tensor):
        gt_video = gt_video.cpu().numpy()
    if isinstance(pred_video, torch.Tensor):
        pred_video = pred_video.cpu().numpy()
    
    if gt_video.max() > 1:
        gt_video = gt_video / 255.0
    if pred_video.max() > 1:
        pred_video = pred_video / 255.0
    
    ssim_per_frame = []
    for gt_frame, pred_frame in zip(gt_video, pred_video):
        ssim_val = ssim(
            gt_frame,
            pred_frame,
            channel_axis=2,
            data_range=1.0
        )
        ssim_per_frame.append(float(ssim_val))
    
    result = {
        'ssim_mean': float(np.mean(ssim_per_frame)),
        'ssim_std': float(np.std(ssim_per_frame)),
    }
    
    if per_frame:
        result['ssim_per_frame'] = ssim_per_frame
    
    return result


# Metric Computation: FID for Video
class FIDMetric:
    """Fréchet Inception Distance for video quality."""
    
    def __init__(self, device: str = "cuda", feature_dim: int = 2048):
        self.device = device
        self.feature_dim = feature_dim
        
        self.inception = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            transform_input=False
        ).to(device)
        self.inception.eval()
        self.inception.fc = torch.nn.Identity()
        
        logger.info("Initialized FID metric with InceptionV3")
    
    def extract_features(
        self,
        video: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract InceptionV3 features from video frames."""
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()
        
        if video.max() > 1:
            video = video / 255.0
        
        video = video.permute(0, 3, 1, 2).to(self.device)
        video = F.interpolate(video, size=(299, 299), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        video = (video - mean) / std
        
        features = []
        with torch.no_grad():
            for i in range(0, len(video), batch_size):
                batch = video[i:i + batch_size]
                feat = self.inception(batch)
                features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        return features
    
    def compute(
        self,
        gt_videos: List[np.ndarray],
        pred_videos: List[np.ndarray],
    ) -> float:
        """Compute FID between ground truth and predicted video distributions."""
        logger.info(f"Computing FID for {len(gt_videos)} video pairs...")
        
        gt_features = []
        pred_features = []
        
        for gt_video, pred_video in tqdm(zip(gt_videos, pred_videos), 
                                          total=len(gt_videos),
                                          desc="Extracting features"):
            gt_feat = self.extract_features(gt_video)
            pred_feat = self.extract_features(pred_video)
            gt_features.append(gt_feat)
            pred_features.append(pred_feat)
        
        gt_features = np.concatenate(gt_features, axis=0)
        pred_features = np.concatenate(pred_features, axis=0)
        
        fid = self._calculate_fid(gt_features, pred_features)
        
        logger.info(f"FID: {fid:.3f}")
        return float(fid)
    
    def _calculate_fid(
        self,
        features_real: np.ndarray,
        features_fake: np.ndarray,
    ) -> float:
        """Calculate FID given feature arrays."""
        mu_real = np.mean(features_real, axis=0)
        sigma_real = np.cov(features_real, rowvar=False)
        
        mu_fake = np.mean(features_fake, axis=0)
        sigma_fake = np.cov(features_fake, rowvar=False)
        
        diff = mu_real - mu_fake
        
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)



# Aggregated Evaluation
class VideoEvaluator:
    """Comprehensive video evaluation combining all metrics."""
    
    def __init__(
        self,
        device: str = "cuda",
        compute_lpips: bool = True,
        compute_mse: bool = True,
        compute_ssim: bool = True,
        compute_fid: bool = False,
    ):
        self.device = device
        self.compute_lpips_flag = compute_lpips
        self.compute_mse_flag = compute_mse
        self.compute_ssim_flag = compute_ssim
        self.compute_fid_flag = compute_fid
        
        if compute_lpips:
            self.lpips_metric = LPIPSMetric(device=device)
        
        if compute_fid:
            self.fid_metric = FIDMetric(device=device)
            self.gt_videos_cache = []
            self.pred_videos_cache = []
    
    def evaluate_single(
        self,
        gt_video: np.ndarray,
        pred_video: np.ndarray,
    ) -> Dict[str, any]:
        """Evaluate a single video pair."""
        metrics = {}
        
        if self.compute_mse_flag:
            mse_results = compute_mse(gt_video, pred_video, per_frame=True)
            metrics.update(mse_results)
        
        if self.compute_ssim_flag:
            ssim_results = compute_ssim(gt_video, pred_video, per_frame=True)
            metrics.update(ssim_results)
        
        if self.compute_lpips_flag:
            lpips_results = self.lpips_metric.compute(gt_video, pred_video)
            metrics.update(lpips_results)
        
        if self.compute_fid_flag:
            self.gt_videos_cache.append(gt_video)
            self.pred_videos_cache.append(pred_video)
        
        return metrics
    
    def compute_fid(self) -> Optional[float]:
        """Compute FID over all cached videos."""
        if not self.compute_fid_flag:
            return None
        
        if len(self.gt_videos_cache) < 2:
            logger.warning("Not enough videos to compute FID (need >= 2)")
            return None
        
        fid = self.fid_metric.compute(
            self.gt_videos_cache,
            self.pred_videos_cache
        )
        
        return fid
    
    def save_summary(
        self,
        all_metrics: List[Dict],
        output_path: str,
        include_fid: bool = True,
    ) -> None:
        """Save summary statistics of all metrics."""
        summary = {}
        
        for key in ['mse_mean', 'ssim_mean', 'lpips_mean']:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                summary[f'{key}_avg'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_min'] = float(np.min(values))
                summary[f'{key}_max'] = float(np.max(values))
        
        if include_fid and self.compute_fid_flag:
            fid = self.compute_fid()
            if fid is not None:
                summary['fid'] = fid
        
        summary['num_videos'] = len(all_metrics)
        
        save_metadata(output_path, **summary)
        
        logger.info(f"Evaluation Summary:")
        logger.info(f"  Number of videos: {summary['num_videos']}")
        if 'mse_mean_avg' in summary:
            logger.info(f"  MSE (avg): {summary['mse_mean_avg']:.4f} ± {summary['mse_mean_std']:.4f}")
        if 'ssim_mean_avg' in summary:
            logger.info(f"  SSIM (avg): {summary['ssim_mean_avg']:.4f} ± {summary['ssim_mean_std']:.4f}")
        if 'lpips_mean_avg' in summary:
            logger.info(f"  LPIPS (avg): {summary['lpips_mean_avg']:.4f} ± {summary['lpips_mean_std']:.4f}")
        if 'fid' in summary:
            logger.info(f"  FID: {summary['fid']:.3f}")



# Create output directory structure
def create_output_structure(
    base_dir: str,
    exp_name: str,       # experiment folder
    ckpt_step: str,      # checkpoint step folder
    subset_name: str,    # dataset subset
) -> Dict[str, str]:
    """Create organized output directory structure:
       base_dir / exp_name / ckpt_step / subset_name / ...
    """
    subset_dir = Path(base_dir) / exp_name / ckpt_step / subset_name

    paths = {
        'root': str(subset_dir),
        'videos_gt': str(subset_dir / 'videos' / 'gt'),
        'videos_pred': str(subset_dir / 'videos' / 'pred'),
        'gifs': str(subset_dir / 'gifs'),
        'metadata': str(subset_dir / 'metadata'),
        'metrics': str(subset_dir / 'metrics'),
        'plots': str(subset_dir / 'plots'),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    logger.info(f"Created output structure at {paths['root']}")
    return paths


# Action padding, used in eval_kv_cache_metrics_lerobot.py
def pad_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Pad actions to the specified action dimension."""
    return torch.nn.functional.pad(actions, (0, action_dim - actions.shape[-1]))