# Few-Step Distillation for Action-Conditioned World Models

A few-step distillation pipeline for training **action-conditioned autoregressive video diffusion world models**, based on **Self-Forcing** and **Distribution Matching Distillation (DMD)**.

This repository is designed for efficient world model inference within a world-model evaluation pipeline.

---

## Few-Step Distillation Results

### Qualitative Comparison

| Original Multi-Step Model | Few-Step Distilled Model         |
|---------------------------|----------------------------------|
| <img src="assets/origin0.gif" width="360"> | <img src="assets/dmd0.gif" width="360"> |

Left: Original multi-step action-conditioned diffusion model  
Right: Few-step distilled model trained with DMD  

The distilled model preserves motion consistency and action responsiveness
while significantly reducing the number of sampling steps.

---

## Requirements

This repository has been tested with::

- **Python 3.12**
- **PyTorch 2.7.1 (CUDA 12.6)**
- NVIDIA GPU (L40S and H200 tested)

---

## Setup

### Create environment

```bash
conda create -n dmd python=3.12 -y
conda activate dmd
pip install -r requirements.txt
```

### Quick Start
```bash
sbatch scripts/train_dmd_framewise.bash
# or
sbatch scripts/train_dmd_chunkwise.bash
```

## Integration with Action-Conditioned World Models

This repository is designed to integrate with:

- Worldgym
  https://world-model-eval.github.io/
  https://arxiv.org/abs/2506.00613

It enables efficient evaluation of action-conditioned video world models as environment simulators for policy evaluation and planning.


## Acknowledgements

This codebase builds upon:

- Self-Forcing  
  https://github.com/guandeh17/Self-Forcing

- World-Model-Eval  
  https://github.com/world-model-eval/world-model-eval


## Citation

If you find this repository useful, please consider citing the following works:

@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}

@misc{quevedo2025worldgymworldmodelenvironment,
  title={WorldGym: World Model as An Environment for Policy Evaluation},
  author={Julian Quevedo and Ansh Kumar Sharma and Yixiang Sun and Varad Suryavanshi and Percy Liang and Sherry Yang},
  year={2025},
  eprint={2506.00613},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2506.00613}
}

@misc{worldmodeleval2025,
  title={World-Model-Eval},
  author={World-Model-Eval Contributors},
  year={2025},
  howpublished={\url{https://github.com/world-model-eval/world-model-eval}}
}
