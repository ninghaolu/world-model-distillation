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
#SBATCH --time=24:00:00


cd /scratch/nl2752/world-model-distillation
source ~/.bashrc
conda activate wmnv

torchrun --nproc_per_node=2 train_dmd.py \
  --pretrained_checkpoint /projects/work/yang-lab/projects/pretrain_world_model/fold_clothes_600m_scratch_ds1/ckpt_000232000.pt \
  --dataset_dir /projects/work/yang-lab/users/yz12129 \
  --subset_names converted_rollout_data/folding_rollout_jan15_ds1,lerobot_dataset/fold_clothes_lerobot_split_realaction_214ep_ds1,converted_rollout_data/folding_rollout_jan16_ds1,converted_rollout_data/folding_other_teleop_demos \
  --batch_size 2 \
  --n_total_frames 49 \
  --n_output_frames 13 \
  --max_cache_frames 13 \
  --wandb_entity nl2752-new-york-university \
  --wandb_project dmd \
  --wandb_name folding_dmd_l40s_2g \
  --validate_every 5000 \
  --num_frame_per_block 1
  