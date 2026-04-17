"""
Training script for Amyloid PET Centiloid Prediction.

New vs previous version
────────────────────────
--augment        enables the full augmentation pipeline during training
--no_tracer_aug  disables per-tracer strength (all tracers get standard aug)

EDA-informed: NAV (n=85) and PIB (small n) get STRONG augmentation;
              FBP and FBB get standard augmentation.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler

# Make the top-level repo importable when running this script directly.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import PETDataset, PETResNet, get_criterion

try:
    from mygo_centiloid import build_train_transform
except ImportError:
    build_train_transform = None


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Sampler
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(centiloids: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampler across 6 CL bins — fixes 64.8 % negative bias."""
    bin_edges = [-np.inf, 0, 25, 50, 75, 100, np.inf]
    bin_idx   = np.digitize(centiloids, bin_edges) - 1
    bin_idx   = np.clip(bin_idx, 0, len(bin_edges) - 2)

    counts  = np.bincount(bin_idx, minlength=len(bin_edges) - 1).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)

    weights = 1.0 / counts[bin_idx]
    weights = weights / weights.sum() * len(weights)

    return WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(centiloids),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    vx = pred   - pred.mean()
    vy = target - target.mean()
    return ((vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, clip_grad):
    model.train()
    running_loss = 0.0

    for images, targets, tracers in loader:
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            preds = model(images, tracers)
            loss  = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * len(images)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for images, targets, tracers in loader:
        images  = images.to(device,  non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            preds = model(images, tracers)
        all_preds.append(preds.cpu())
        all_targets.append(targets)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return (preds - targets).abs().mean().item(), pearson_r(preds, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train mygo_centiloid")

    # Paths
    parser.add_argument("--train_csv",      required=True)
    parser.add_argument("--val_csv",        required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")

    # Training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad",    type=float, default=1.0)
    parser.add_argument("--patience",     type=int,   default=20)

    # Augmentation ← NEW
    parser.add_argument("--augment",         action="store_true", default=True,
                        help="Enable training augmentation (default: on)")
    parser.add_argument("--no_augment",      dest="augment", action="store_false",
                        help="Disable all augmentation")
    parser.add_argument("--no_tracer_aug",   action="store_true",
                        help="Use same aug strength for all tracers "
                             "(disables strong aug for NAV/PIB)")

    # Loss
    parser.add_argument("--loss",  default="centiloid",
                        choices=["centiloid", "huber", "mse", "mae"])
    parser.add_argument("--delta", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=0.7)

    # Model
    parser.add_argument("--emb_dim",      type=int,   default=32)
    parser.add_argument("--dropout_high", type=float, default=0.4)
    parser.add_argument("--dropout_low",  type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}\n")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    # Load training dataset first (no transform yet — need tracer_map first)
    train_ds = PETDataset(args.train_csv, cache=False)
    val_ds   = PETDataset(
        args.val_csv, tracer_map=train_ds.tracer_map, cache=False
        # transform=None → no augmentation on validation (deterministic)
    )

    # ── Augmentation setup ────────────────────────────────────────────────
    # EDA: NAV n=85 and PIB are small → use stronger augmentation for those.
    # FBP (n~large) and FBB (n~large) use standard augmentation.
    #
    # Reference: Pérez-García et al. TorchIO [1] recommend stronger spatial
    # augmentation for datasets with fewer than ~100 subjects per class.

    if args.augment and build_train_transform is None:
        print("Augmentation: requested but augmentation.py not found → disabled\n")
        args.augment = False

    if args.augment:
        # Tracers that benefit from stronger augmentation (from EDA)
        SMALL_N_TRACERS = {"NAV", "PIB"}

        tid_to_name = {v: k for k, v in train_ds.tracer_map.items()}

        if args.no_tracer_aug:
            # Same standard augmentation for all tracers
            train_ds.transform = build_train_transform(strong=False)
            print("Augmentation: uniform standard (tracer-aware disabled)\n")
        else:
            # Per-tracer dict: stronger for NAV/PIB, standard for FBP/FBB
            train_ds.transform = {
                tid: build_train_transform(
                    strong=(tid_to_name.get(tid, "") in SMALL_N_TRACERS)
                )
                for tid in set(train_ds.tracers.tolist())
            }
            print("Augmentation: per-tracer")
            for tid, name in tid_to_name.items():
                strength = "STRONG" if name in SMALL_N_TRACERS else "standard"
                n = (train_ds.tracers == tid).sum().item()
                print(f"  {name:>4}  n={n:>4}  → {strength}")
            print()
    else:
        print("Augmentation: disabled\n")

    # ── Sampler + DataLoaders ─────────────────────────────────────────────
    sampler = build_weighted_sampler(train_ds.centiloids.numpy())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=args.num_workers,
        pin_memory=use_amp,
        drop_last=True,           # stable BN stats
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mean_cl     = float(train_ds.centiloids.mean())
    num_tracers = len(train_ds.tracer_map)

    model = PETResNet(
        num_tracers    = num_tracers,
        emb_dim        = args.emb_dim,
        dropout_high   = args.dropout_high,
        dropout_low    = args.dropout_low,
        mean_centiloid = mean_cl,
    ).to(device)

    model.summary(input_size=(args.batch_size, 1, 128, 128, 128))

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_kwargs = {}
    if args.loss in ("centiloid", "huber"):
        loss_kwargs["delta"] = args.delta
    if args.loss == "centiloid":
        loss_kwargs["alpha"] = args.alpha
    criterion = get_criterion(args.loss, **loss_kwargs)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler    = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_mae     = float("inf")
    patience_cnt = 0
    best_ckpt    = os.path.join(args.checkpoint_dir, "best_model.pt")
    last_ckpt    = os.path.join(args.checkpoint_dir, "last_model.pt")

    header = f"{'Epoch':>6} | {'Train Loss':>11} | {'Val MAE':>8} | {'Pearson r':>10} | {'LR':>9}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, args.clip_grad,
        )
        val_mae, val_r = validate(model, val_loader, device)
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {train_loss:>11.4f} | {val_mae:>8.2f} | {val_r:>10.4f} | {lr:>9.2e}")

        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "val_mae":          val_mae,
            "val_r":            val_r,
            "tracer_map":       train_ds.tracer_map,
            "num_tracers":      num_tracers,
            "emb_dim":          args.emb_dim,
            "dropout_high":     args.dropout_high,
            "dropout_low":      args.dropout_low,
        }
        torch.save(ckpt, last_ckpt)

        if val_mae < best_mae:
            best_mae     = val_mae
            patience_cnt = 0
            torch.save(ckpt, best_ckpt)
            print(f"  ✓ New best MAE: {best_mae:.2f} CL → {best_ckpt}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\nDone. Best Val MAE: {best_mae:.2f} CL  |  checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()