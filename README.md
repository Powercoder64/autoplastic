# Autoplastic Neural Network PyTorch

PyTorch implementation of the **Autoplastic Neural Network (ANN)**.  The model learns structure and weights jointly:

- **P** — layer inclusion (depth control)  
- **C** — per-channel connectivity (neuron-level sparsity)  
- **α** — per-block conv↔attention mixing

Training is coordinated by a **stability gate** `S = exp(-||ΔW||_F)` and a **two-timescale** schedule (weights fast, architecture slow).

---

## Highlights

- **Self-evolving architecture:** learns depth (**P**) and connectivity (**C**) during training; exports a static checkpoint at the end.  
- **Conv + attention mixing:** per-block **α** adapts between convolution and (light) spatial attention; **α** trains only when a block is unstable (`S < ε`).  
- **Two-timescale learning:** default `η_a = η_w / 20` and `ε = 0.85`.  
- **Practical extras:** cosine LR with warm-up, AMP (mixed precision), label smoothing, optional MixUp, token-downsampled attention.

---

## Requirements

- Python 3.10+  
- PyTorch 2.x  
- torchvision

---

## Quick Start


**1) Create and activate an environment (example)**
```bash
conda create -n autoplastic python=3.10 -y
conda activate autoplastic
```

**2) Install dependencies (choose the CUDA build that matches your driver if needed)**
pip install torch torchvision

**3) Train (auto-downloads CIFAR to ./data)**
python autoplastic.py

## Configuration

Edit `TrainConfig` at the top of `autoplastic.py`:

- **Core:** `epochs`, `batch_size`, `base_lr`, `weight_decay`
- **Architecture cadence:** `epsilon` (stability gate), `arch_lr_ratio` (`η_a / η_w`)
- **Gating init & bounds:** `p_init`, `c_init`, `p_min`, `c_min`, `c_max`
- **Training extras:** `label_smoothing`, `use_mixup`, `amp`, `warmup_epochs`

**Recommended for images:** `epsilon ≈ 0.80–0.85`, `arch_lr_ratio ≈ 1/20`.

---

## Training Notes

- **Stability:** `S = exp(-||W − W_EMA||_F)` is evaluated each epoch using an EMA snapshot. If `S ≥ ε`, `α` is frozen next epoch.  
- **Initialization:** `p_init = c_init = 0.9` with floors (`p_min = 0.50`, `c_min = 0.40`) avoids early under-capacity; pruning still proceeds during training.  
- **Speed tips:** Enable AMP, use channels-last memory format, and keep attention tokens ≤ `8×8`.

---

## Expected Logs

Per-epoch console output includes:

- Train CE / Acc, Val CE / Acc  
- Effective depth `L_eff` and active-channel ratio  
- Per-block **P**, **α**, mean **C**, and prune %

**Example:**
```text
Epoch 007/040
Train CE: 0.6123 | Train Acc: 78.40% | Val CE: 0.4851 | Val Acc: 83.72%
Effective depth (L_eff): 4.21 | Active-channel ratio: 0.86
P: [0.93, 0.91, 0.90, 0.88, 0.86, 0.85]
α: [0.58, 0.55, 0.52, 0.50, 0.49, 0.47]
mean C: [0.91, 0.89, 0.87, 0.84, 0.82, 0.80]
pruned %: [9.0, 11.0, 13.0, 16.0, 18.0, 20.0]
```

## Citation

If you use this repository, please cite:

```bibtex
@article{KorbanAutoplasticNN,
  title   = {Autoplastic Neural Network: Self-Evolving Networks with Dynamic Connectivity and Modality Adaptation},
  author  = {Matthew Korban and Peter Youngs and Scott T. Acton},
  year    = {2025}
}


