import wandb
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

class Logger:
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb and wandb is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_video(self, tag: str, video: np.ndarray, step: int, fps: int = 8,
                  caption: str | None = None) -> None:
        if self.use_wandb:
            video_uint8 = (video[0] * 255).astype(np.uint8)
            video_ch_first = np.transpose(video_uint8, (0, 3, 1, 2))
            wandb.log(
                {tag: wandb.Video(video_ch_first, fps=fps, format="gif", caption=caption)},
                step=step,
            )

@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def init_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True
    return 0, 0, 1, False


def pad_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.pad(actions, (0, action_dim - actions.shape[-1]))