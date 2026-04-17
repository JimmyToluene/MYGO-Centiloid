"""
Prediction script for Amyloid PET Centiloid Prediction.

Usage:
    python predict.py \
        --csv        /projectnb/medaihack/ABPET/data/val.csv \
        --checkpoint checkpoints/best_model.pt \
        --output     predictions.csv
"""

import argparse
import os, sys
import torch
from torch.utils.data import DataLoader

# Make the top-level repo importable when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import PETDataset, PETResNet, PETResNetNoFiLM
from mygo_centiloid.utils import (
    make_run_dir, write_config, write_metrics, append_registry,
)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []

    for batch in loader:
        images, tracers = batch[0], batch[-1]
        images  = images.to(device,  non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            preds = model(images, tracers)

        all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--output",      type=str, default="results/predictions.csv")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name",    type=str, default=None)
    parser.add_argument("--log_dir",     type=str, default="logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tracer_map   = ckpt["tracer_map"]
    num_tracers  = ckpt["num_tracers"]
    model_name   = ckpt.get("model", "petresnet")   # back-compat for old ckpts
    emb_dim      = ckpt.get("emb_dim", 32)
    dropout_high = ckpt.get("dropout_high", 0.4)
    dropout_low  = ckpt.get("dropout_low",  0.2)

    # ── Run folder + registry setup ───────────────────────────────────────
    run_dir = make_run_dir(
        run_type="predict", model=model_name,
        base=args.log_dir, run_name=args.run_name,
    )
    write_config(run_dir, {"run_type": "predict", "model": model_name, **vars(args)})
    print(f"Run dir: {run_dir}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = PETDataset(args.csv, tracer_map=tracer_map)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model (dispatch on checkpoint's recorded architecture) ────────────
    if model_name == "petresnet_no_film":
        model = PETResNetNoFiLM(
            num_tracers  = num_tracers,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    else:
        model = PETResNet(
            num_tracers  = num_tracers,
            emb_dim      = emb_dim,
            dropout_high = dropout_high,
            dropout_low  = dropout_low,
        ).to(device)
    print(f"Model: {model_name}\n")

    # Strip torch.compile prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v
                  for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}\n")

    model.summary(input_size=(args.batch_size, 1, 128, 128, 128))

    # ── Predict ───────────────────────────────────────────────────────────
    predictions = predict(model, loader, device)

    out_df = dataset.df[["npy_path", "TRACER.AMY"]].copy()
    if "ID" in dataset.df.columns:
        out_df.insert(0, "ID", dataset.df["ID"])
    out_df["PREDICTED_CENTILOIDS"] = predictions

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} predictions → {args.output}")

    # ── Registry ──────────────────────────────────────────────────────────
    metrics = {
        "run_type":   "predict",
        "model":      model_name,
        "n_samples":  int(len(out_df)),
        "checkpoint": args.checkpoint,
        "output":     args.output,
    }
    write_metrics(run_dir, metrics)
    append_registry({
        "type":    "predict",
        "model":   model_name,
        "run_dir": str(run_dir),
        "ckpt":    args.checkpoint,
        "output":  args.output,
        "metrics": {"n_samples": int(len(out_df))},
    })


if __name__ == "__main__":
    main()