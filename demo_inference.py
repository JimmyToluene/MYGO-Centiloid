"""
demo_inference.py — minimal end-to-end demo for PETResNet (Centiloid prediction).

Loads the 4 synthetic samples in pseudodata/demo.csv (one per tracer:
FBP / FBB / NAV / PIB), runs them through PETResNet, and prints the
predicted Centiloid score for each.

Use cases
─────────
  - Verify that a fresh checkout of the repo is functional.
  - Verify a trained checkpoint loads correctly before submission.

Usage
─────
    # Without a checkpoint — uses random-initialised weights
    # (predictions are NOT meaningful; the demo only proves the pipeline runs)
    python demo_inference.py

    # With a trained checkpoint
    python demo_inference.py --checkpoint checkpoints/best_model.pt

    # On custom synthetic data
    python demo_inference.py --csv my_demo.csv
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from abpet import PETDataset, PETResNet


# ── Defaults match the training config in train.py ────────────────────────
DEFAULT_TRACER_MAP = {"FBB": 0, "FBP": 1, "NAV": 2, "PIB": 3}
DEFAULT_EMB_DIM    = 32
DEFAULT_DROPOUT_HI = 0.4
DEFAULT_DROPOUT_LO = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",        default="pseudodata/demo.csv",
                   help="Demo CSV (columns: ID, npy_path, TRACER.AMY)")
    p.add_argument("--checkpoint", default=None,
                   help="Optional checkpoint path; uses random init if absent")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def build_model(args: argparse.Namespace, device: torch.device):
    """Instantiate PETResNet, optionally loading a trained checkpoint."""
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = dict(
            tracer_map   = ckpt["tracer_map"],
            num_tracers  = ckpt["num_tracers"],
            emb_dim      = ckpt.get("emb_dim",      DEFAULT_EMB_DIM),
            dropout_high = ckpt.get("dropout_high", DEFAULT_DROPOUT_HI),
            dropout_low  = ckpt.get("dropout_low",  DEFAULT_DROPOUT_LO),
        )
        model = PETResNet(
            num_tracers  = cfg["num_tracers"],
            emb_dim      = cfg["emb_dim"],
            dropout_high = cfg["dropout_high"],
            dropout_low  = cfg["dropout_low"],
        ).to(device).eval()
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        print(f"  Loaded checkpoint : {args.checkpoint}")
        return model, cfg["tracer_map"]

    cfg = dict(
        tracer_map  = DEFAULT_TRACER_MAP,
        num_tracers = len(DEFAULT_TRACER_MAP),
    )
    model = PETResNet(
        num_tracers  = cfg["num_tracers"],
        emb_dim      = DEFAULT_EMB_DIM,
        dropout_high = DEFAULT_DROPOUT_HI,
        dropout_low  = DEFAULT_DROPOUT_LO,
    ).to(device).eval()
    print("  ⚠  No checkpoint — RANDOM weights, predictions are not meaningful")
    return model, cfg["tracer_map"]


@torch.no_grad()
def main() -> int:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  PETResNet — demo inference")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Demo CSV   : {args.csv}")
    if not os.path.exists(args.csv):
        print(f"\n  ✗ CSV not found: {args.csv}", file=sys.stderr)
        return 1

    model, tracer_map = build_model(args, device)

    dataset = PETDataset(args.csv, tracer_map=tracer_map)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)

    print()
    print(f"  {'ID':<10} {'TRACER':<8} {'PREDICTED_CL':>14}")
    print("  " + "-" * 36)

    row = 0
    for batch in loader:
        # Dataset has no CENTILOIDS → returns (image, tracer)
        images, tracers = batch[0], batch[-1]
        images  = images.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)
        preds = model(images, tracers).cpu().tolist()
        for p in preds:
            r = dataset.df.iloc[row]
            print(f"  {r['ID']:<10} {r['TRACER.AMY']:<8} {p:>+11.2f} CL")
            row += 1

    print("=" * 60)
    print("  Done.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
