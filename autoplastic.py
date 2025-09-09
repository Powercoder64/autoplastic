import math, random, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# -----------------------
# Repro & backend
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# -----------------------
# Light spatial self-attention with token downsampling
# -----------------------
class SpatialSelfAttentionLite(nn.Module):
    """
    Single-head attention; computes attention on a downsampled grid (<= 8x8 tokens),
    then projects back. Greatly reduces O(N^2) while keeping the conv-attn "flavor".
    """
    def __init__(self, in_ch: int, out_ch: int, attn_dim: int = None, stride: int = 1, max_tokens: int = 64):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.stride = stride
        self.max_tokens = max_tokens
        if attn_dim is None:
            attn_dim = max(8, in_ch // 4)

        self.q = nn.Conv2d(in_ch, attn_dim, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(in_ch, attn_dim, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(in_ch, attn_dim, 1, 1, 0, bias=False)
        self.proj = nn.Conv2d(attn_dim, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

        # Optional stride via avg pool on the attention path
        self.path_pool = nn.AvgPool2d(2, 2) if stride == 2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.path_pool is not None:
            x = self.path_pool(x)
        B, C, H, W = x.shape
        # Downsample to at most max_tokens
        # Keep aspect ratio: pick ds such that (H//ds)*(W//ds) <= max_tokens
        ds = 1
        while (H // (ds+1)) * (W // (ds+1)) >= 1 and (H // (ds+1)) * (W // (ds+1)) > self.max_tokens:
            ds += 1
        if ds > 1:
            x_small = F.avg_pool2d(x, kernel_size=ds, stride=ds)
        else:
            x_small = x

        Bh, Ch, Hs, Ws = x_small.shape
        N = Hs * Ws

        q = self.q(x_small).reshape(Bh, -1, N).transpose(1, 2)  # (B, N, d)
        k = self.k(x_small).reshape(Bh, -1, N)                  # (B, d, N)
        v = self.v(x_small).reshape(Bh, -1, N).transpose(1, 2)  # (B, N, d)
        d = k.shape[1]
        attn = torch.softmax((q @ k) / math.sqrt(d), dim=-1)    # (B, N, N)
        out_small = attn @ v                                    # (B, N, d)
        out_small = out_small.transpose(1, 2).reshape(Bh, -1, Hs, Ws)

        # Upsample back to H,W if needed
        if (Hs, Ws) != (H, W):
            out = F.interpolate(out_small, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = out_small

        out = self.bn(self.proj(out))
        return out

# -----------------------
# Autoplastic Block (deterministic gates)
# -----------------------
class ANNBlock(nn.Module):
    """
    Residual block:
      - Conv path: 3x3 -> BN -> ReLU -> 3x3 -> BN
      - Attention path: SpatialSelfAttentionLite
      - Mix via alpha in [0,1] (stored as logits), then residual add
      - Architecture gates (deterministic):
          P: scalar in [0,1] applied to the residual branch (like StochasticDepth mean)
          C: per-output-channel gate in [0,1]
    We *do not* sample Bernoulli masks during training to avoid high variance on CIFAR.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, init_alpha: float = 0.5,
                 p_init: float = 0.9, c_init: float = 0.9):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride

        # Conv path
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Attention path
        self.attn = SpatialSelfAttentionLite(in_ch, out_ch, stride=stride)

        # Residual projection if needed
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        # Alpha (mix) stored as logits; alpha = sigmoid(alpha_logit)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(init_alpha, dtype=torch.float32)))

        # Architectural probabilities as buffers (updated by custom rules)
        self.register_buffer("P", torch.tensor(float(p_init)))
        self.register_buffer("C", torch.full((out_ch,), float(c_init)))

        # Accumulators for epoch-wise stability/grad
        self._grad_accum = torch.zeros(out_ch)  # per-channel grad norms for conv2
        self._prev_weights_ema = None           # EMA snapshot for stability
        self.relu = nn.ReLU(inplace=True)

        # Training-time toggle for alpha updates
        self.alpha_trainable = True

    # alpha in [0,1]
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic gates (mean-field)
        Mi = self.P.clamp(0.0, 1.0)
        m = self.C.clamp(0.0, 1.0)

        # Conv path
        y_conv = self.relu(self.bn1(self.conv1(x)))
        y_conv = self.bn2(self.conv2(y_conv))

        # Attention path
        y_att = self.attn(x)

        # Mix conv/attn
        a = self.alpha
        y = a * y_conv + (1.0 - a) * y_att

        # Apply gates on residual branch only
        y = y * m.view(1, -1, 1, 1) * Mi

        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(identity + y)
        return out

    # --------- Utilities for architecture updates ----------
    def reset_grad_accum(self, device=None):
        self._grad_accum = torch.zeros(self.out_ch, device=device or self.conv2.weight.device)

    def accumulate_channel_grad_norms(self):
        if self.conv2.weight.grad is None:
            return
        g = self.conv2.weight.grad.detach()
        g_per_out = g.flatten(1).norm(p=2, dim=1)  # (out_ch,)
        self._grad_accum = self._grad_accum + g_per_out

    def snapshot_weights(self) -> Dict[str, torch.Tensor]:
        snap = {}
        for n, p in self.named_parameters(recurse=True):
            if p.requires_grad:
                snap[n] = p.detach().clone()
        return snap

    def update_ema_snapshot(self, snap: Dict[str, torch.Tensor], ema: float = 0.9):
        # Maintain EMA of parameters for stab computation
        if self._prev_weights_ema is None:
            self._prev_weights_ema = {k: v.clone() for k, v in snap.items()}
        else:
            for k, v in snap.items():
                self._prev_weights_ema[k] = ema * self._prev_weights_ema[k].to(v.device) + (1 - ema) * v

    @torch.no_grad()
    def stability_from_ema(self) -> Tuple[float, torch.Tensor]:
        """
        S_block = exp(-||W - W_ema||_F), S_edge from conv2 per-output-channel deltas
        """
        if self._prev_weights_ema is None:
            return 1.0, torch.ones(self.out_ch, device=self.conv2.weight.device)

        total_sq = 0.0
        for n, p in self.named_parameters(recurse=True):
            if p.requires_grad and (n in self._prev_weights_ema):
                diff = (p.detach() - self._prev_weights_ema[n].to(p.device))
                total_sq += (diff * diff).sum().item()
        deltaW = math.sqrt(total_sq + 1e-12)
        S_block = math.exp(-deltaW)

        if "conv2.weight" in self._prev_weights_ema:
            diff = (self.conv2.weight.detach() - self._prev_weights_ema["conv2.weight"].to(self.conv2.weight.device))
            per_ch = diff.flatten(1).norm(p=2, dim=1)
            S_edge = torch.exp(-per_ch)
        else:
            S_edge = torch.ones(self.out_ch, device=self.conv2.weight.device)

        return S_block, S_edge

# -----------------------
# Whole model
# -----------------------
class AutoplasticNet(nn.Module):
    def __init__(self, num_classes: int = 10, init_alpha: float = 0.5, p_init=0.9, c_init=0.9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.b1 = ANNBlock(32, 64, stride=2, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.b2 = ANNBlock(64, 64, stride=1, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.b3 = ANNBlock(64, 128, stride=2, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.b4 = ANNBlock(128, 128, stride=1, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.b5 = ANNBlock(128, 256, stride=2, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.b6 = ANNBlock(256, 256, stride=1, init_alpha=init_alpha, p_init=p_init, c_init=c_init)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, num_classes))
        self.blocks: List[ANNBlock] = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    # ---- Arch helpers ----
    def reset_grad_accum(self, device=None):
        for b in self.blocks: b.reset_grad_accum(device=device)

    def accumulate_channel_grad_norms(self):
        for b in self.blocks: b.accumulate_channel_grad_norms()

    def snapshot_and_update_ema(self, ema=0.9):
        snaps = []
        for b in self.blocks:
            s = b.snapshot_weights()
            b.update_ema_snapshot(s, ema=ema)
            snaps.append(s)  # keep returning for completeness if needed
        return snaps

    def arch_state_summary(self) -> Dict[str, List[float]]:
        P = [float(b.P.item()) for b in self.blocks]
        alpha = [float(b.alpha.detach().item()) for b in self.blocks]
        meanC = [float(b.C.mean().item()) for b in self.blocks]
        return {"P": P, "alpha": alpha, "meanC": meanC}

    def effective_depth_and_activity(self) -> Tuple[float, float]:
        leff, act = 0.0, 0.0
        for b in self.blocks:
            leff += float(b.P.item()) * float(b.C.mean().item())
            act += float(b.C.mean().item())
        act /= len(self.blocks)
        return leff, act

# -----------------------
# Config
# -----------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    base_lr: float = 0.10          # SGD LR for weights
    alpha_lr_ratio: float = 1/20   # slower timescale for alpha (paper)
    momentum: float = 0.9
    weight_decay: float = 5e-4
    epsilon: float = 0.85          # stability gate
    arch_lr_ratio: float = 1/20    # eta_a = lambda * eta_w
    delta_grad_thresh: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
    print_every: int = 50
    init_alpha: float = 0.5
    p_init: float = 0.9
    c_init: float = 0.9
    p_min: float = 0.50            # keep some capacity
    c_min: float = 0.40
    c_max: float = 0.995
    label_smoothing: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    warmup_epochs: int = 3         # for cosine LR
    amp: bool = True               # automatic mixed precision

# -----------------------
# Data
# -----------------------
def make_dataloaders(cfg: TrainConfig):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers>0)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers>0)
    return train_loader, test_loader

# -----------------------
# MixUp helpers (optional)
# -----------------------
def mixup_data(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# -----------------------
# Train / Eval
# -----------------------
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def train_one_epoch(model: AutoplasticNet,
                    opt_w: torch.optim.Optimizer,
                    opt_alpha: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    train_loader: DataLoader,
                    device: str,
                    cfg: TrainConfig,
                    epoch: int,
                    alpha_trainable_mask: List[bool]) -> Tuple[float, float]:
    model.train()
    model.reset_grad_accum(device=device)

    total_loss, total_correct, total = 0.0, 0, 0
    use_amp = cfg.amp and device.startswith("cuda")

    for it, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Optional MixUp (do not combine with label smoothing if you prefer simplicity)
        if cfg.use_mixup:
            x_in, y_a, y_b, lam = mixup_data(x, y, cfg.mixup_alpha)
        else:
            x_in, y_a, y_b, lam = x, y, y, 1.0

        opt_w.zero_grad(set_to_none=True)
        opt_alpha.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x_in)
                if cfg.use_mixup and lam < 1.0:
                    loss = lam * F.cross_entropy(logits, y_a, label_smoothing=cfg.label_smoothing) + \
                           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=cfg.label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        else:
            logits = model(x_in)
            if cfg.use_mixup and lam < 1.0:
                loss = lam * F.cross_entropy(logits, y_a, label_smoothing=cfg.label_smoothing) + \
                       (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=cfg.label_smoothing)
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate per-channel grads (for C update)
        model.accumulate_channel_grad_norms()

        # Freeze alpha grads for stable blocks
        for trainable, b in zip(alpha_trainable_mask, model.blocks):
            if not trainable and b.alpha_logit.grad is not None:
                b.alpha_logit.grad.zero_()

        # Optimizer steps
        if use_amp:
            scaler.step(opt_w)
            scaler.step(opt_alpha)
            scaler.update()
        else:
            opt_w.step()
            opt_alpha.step()

        # Metrics
        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            # For training acc, use the non-mixed labels
            logits_eval = logits if not cfg.use_mixup else model(x)
            total_correct += (logits_eval.argmax(dim=1) == y).sum().item()
        total += x.size(0)

        if cfg.print_every and it % cfg.print_every == 0:
            print(f"  [epoch {epoch:03d} | iter {it:04d}] train CE: {total_loss/total:.4f}")

    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model: AutoplasticNet, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

# -----------------------
# Architecture updates (epoch-wise, stability-gated) — Eq. (3)–(5) w/ practical bounds
# -----------------------
def update_architecture_epoch(model: AutoplasticNet, cfg: TrainConfig) -> List[bool]:
    """
    For each block:
      - S_block, S_edge from EMA snapshots
      - P update: increase if unstable (S<eps), else decrease (small), with [p_min,1]
      - C update: increase where unstable channels; decrease where grad small; clamp [c_min,c_max]
      - alpha trainable next epoch iff unstable
    """
    arch_lr = cfg.base_lr * cfg.arch_lr_ratio
    stable_flags = []

    for b in model.blocks:
        S_block, S_edge = b.stability_from_ema()
        AGT = 1.0 - S_block

        # P update (mean-field residual depth)
        if S_block < cfg.epsilon:
            newP = float(b.P.item()) + arch_lr * max(0.05, AGT)
        else:
            newP = float(b.P.item()) - arch_lr * max(0.05, (1.0 - AGT))
        newP = max(cfg.p_min, min(1.0, newP))
        b.P = torch.tensor(newP, device=b.P.device)

        # C update
        # Normalize grad accum to [0,1]
        g = b._grad_accum
        if g.numel() == 0:
            g = torch.zeros_like(b.C)
        else:
            g = g / (g.abs().max().clamp(min=1e-8))

        NGT = 1.0 - S_edge  # vector
        C_new = b.C.clone().detach()
        # Increase where unstable channels
        inc_mask = (S_edge < cfg.epsilon)
        C_new[inc_mask] = (C_new[inc_mask] + arch_lr * torch.maximum(NGT[inc_mask], torch.tensor(0.05, device=C_new.device))).clamp(cfg.c_min, cfg.c_max)
        # Decrease where grads tiny
        # Use a soft percentile threshold for "insignificant"
        thr = torch.quantile(g.detach(), 0.20).item() if g.numel() > 0 else cfg.delta_grad_thresh
        dec_mask = (g < max(cfg.delta_grad_thresh, thr))
        C_new[dec_mask] = (C_new[dec_mask] - arch_lr * torch.maximum(1.0 - NGT[dec_mask], torch.tensor(0.05, device=C_new.device))).clamp(cfg.c_min, cfg.c_max)
        b.C = C_new

        stable_flags.append(bool(S_block >= cfg.epsilon))

    return stable_flags

# -----------------------
# Warmup+Cosine scheduler
# -----------------------
class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup = max(0, int(warmup_epochs))
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            warm_frac = (self.last_epoch + 1) / max(1, self.warmup)
            return [base_lr * warm_frac for base_lr in self.base_lrs]
        # cosine decay to 0
        t = (self.last_epoch - self.warmup + 1) / max(1, (self.max_epochs - self.warmup))
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return [base_lr * cos for base_lr in self.base_lrs]

# -----------------------
# Main
# -----------------------
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = cfg.device
    print(f"Using device: {device} | AMP: {cfg.amp}")

    train_loader, val_loader = make_dataloaders(cfg)

    # Model
    model = AutoplasticNet(num_classes=10, init_alpha=cfg.init_alpha, p_init=cfg.p_init, c_init=cfg.c_init).to(device)
    if device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)  # small throughput win

    # Param groups
    alpha_params = [b.alpha_logit for b in model.blocks]
    weight_params = [p for n, p in model.named_parameters() if not n.endswith("alpha_logit")]

    opt_w = torch.optim.SGD(weight_params, lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    opt_alpha = torch.optim.SGD(alpha_params, lr=cfg.base_lr * cfg.alpha_lr_ratio, momentum=0.0, weight_decay=0.0)

    # LR schedulers
    sched_w = CosineWithWarmup(opt_w, warmup_epochs=cfg.warmup_epochs, max_epochs=cfg.epochs)
    sched_alpha = CosineWithWarmup(opt_alpha, warmup_epochs=cfg.warmup_epochs, max_epochs=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))

    # Initialize EMA snapshots for stability (after 1 fwd pass)
    model.snapshot_and_update_ema(ema=0.9)

    alpha_trainable_mask = [True] * len(model.blocks)
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, opt_w, opt_alpha, scaler, train_loader, device, cfg, epoch, alpha_trainable_mask)
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Architecture updates (epoch-wise, stability-gated)
        stable_flags = update_architecture_epoch(model, cfg)
        alpha_trainable_mask = [not s for s in stable_flags]  # train alpha only if unstable

        # Update EMA snapshots for next-epoch stability measurement
        model.snapshot_and_update_ema(ema=0.9)

        # LR steps
        sched_w.step()
        sched_alpha.step()

        # Metrics
        leff, active_ratio = model.effective_depth_and_activity()
        arch = model.arch_state_summary()

        def fmt(a): return "[" + ", ".join(f"{v:.2f}" for v in a) + "]"
        pruned_pct = [(1.0 - c) * 100.0 for c in arch["meanC"]]

        print("\n" + "-" * 92)
        print(f"Epoch {epoch:03d}/{cfg.epochs:03d}")
        print(f"Train CE: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val CE: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Effective depth (L_eff): {leff:.2f} | Active-channel ratio: {active_ratio:.3f}")
        print(f"P (layer inclusion):        {fmt(arch['P'])}")
        print(f"alpha (conv↔attn mix):      {fmt(arch['alpha'])}")
        print(f"mean C (per-block):         {fmt(arch['meanC'])}")
        print(f"pruned % (per-block):       {fmt(pruned_pct)}")
        print("-" * 92 + "\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict()}, "autoplastic_cifar_best.pt")

    print(f"Finished. Best Val Acc: {best_acc*100:.2f}%")
    print("Saved best checkpoint to autoplastic_cifar_best.pt")

if __name__ == "__main__":
    main()
