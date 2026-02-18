#!/bin/bash

#SBATCH --job-name=dmd
#SBATCH -D .
#SBATCH --output=/scratch/nl2752/dmd_logs/%j.out
#SBATCH --error=/scratch/nl2752/dmd_logs/%j.err

# --- Updated Flags ---
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --constraint="l40s"
#SBATCH --gres=gpu:2
#SBATCH --mem=300GB

# --- Task Configuration ---
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# --- Previous/Standard Flags ---
#SBATCH --time=1:00:00
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"


cd /scratch/nl2752/world-model-distillation
source ~/.bashrc
conda activate pretrain

torchrun --nproc_per_node=2 train_dmd.py \
  --pretrained_checkpoint outputs/20250101_120000/ckpt_000100000.pt \
  --dataset_dir /projects/work/yang-lab/users/yz12129 \
  --subset_names converted_rollout_data/folding_rollout_jan15_ds1,lerobot_dataset/fold_clothes_lerobot_split_realaction_214ep_ds1,converted_rollout_data/folding_rollout_jan16_ds1,converted_rollout_data/folding_other_teleop_demos \
  --batch_size 2 \
  --n_total_frames 24 \
  --n_output_frames 15 \
  --max_cache_frames 15 \
  --wandb_entity nl2752-new-york-university \
  --wandb_project dmd \
  --wandb_name folding_dmd_l40s_2g