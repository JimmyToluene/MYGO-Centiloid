"""
gradcam.py  —  EXPERIMENTAL, not part of the main pipeline

3D Grad-CAM overlays for PETResNet / PETResNetNoFiLM checkpoints.

For a regression model the "target" is just y_pred itself — gradients tell you
which voxels, when perturbed, would move the predicted centiloid up. High-
attention regions should overlap cortical grey matter (precuneus, frontal
cortex, posterior cingulate) if the model is doing something sensible.

Default target module:
    petresnet           → model.film4    (last FiLM-modulated conv feature)
    petresnet_no_film   → model.stage4   (last conv feature)

Usage:
    # reuses dev/config/predict.yaml for paths + checkpoint
    python dev/gradcam.py --config dev/config/predict.yaml \\
        --n_samples 8 \\
        --out_dir results/gradcam/baseline

    # override the target layer to see finer maps (coarser → finer):
    #   film4/stage4 = 4³    film3/stage3 = 8³    film2/stage2 = 16³
    python dev/gradcam.py --target_layer stage3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import PETDataset, PETResNet, PETResNetNoFiLM


# ─────────────────────────────────────────────────────────────────────────────
# Config + model
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model", "petresnet")
    num_tracers = ckpt["num_tracers"]

    if name == "petresnet_no_film":
        model = PETResNetNoFiLM(
            num_tracers  = num_tracers,
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)
    else:
        model = PETResNet(
            num_tracers  = num_tracers,
            emb_dim      = ckpt.get("emb_dim", 32),
            dropout_high = ckpt.get("dropout_high", 0.4),
            dropout_low  = ckpt.get("dropout_low",  0.2),
        ).to(device)

    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, name, ckpt


def resolve_target_layer(model, model_name: str, name: str | None):
    """Map a layer name to the actual nn.Module."""
    if name is None:
        name = "film4" if model_name == "petresnet" else "stage4"
    if not hasattr(model, name):
        available = [n for n in ("film1", "film2", "film3", "film4",
                                 "stage1", "stage2", "stage3", "stage4")
                     if hasattr(model, n)]
        raise ValueError(f"Model {model_name} has no attribute {name!r}. "
                         f"Available: {available}")
    return name, getattr(model, name)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM core
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM3D:
    """3D Grad-CAM for regression models.

    Registers forward + full-backward hooks on a target module; .compute(x, tid)
    does one forward-backward pass and returns (y_pred, heatmap) where heatmap
    is a numpy array of shape x.shape[2:] normalised to [0, 1].
    """

    def __init__(self, model, target_module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        self._h_fwd = target_module.register_forward_hook(self._fwd)
        self._h_bwd = target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _module, _inp, out):
        self.activations = out.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()

    def compute(self, x: torch.Tensor, tracer_idx: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        y = self.model(x, tracer_idx)            # (1,)
        y.sum().backward()

        A = self.activations                      # (1, C, d, h, w)
        G = self.gradients                        # (1, C, d, h, w)
        alpha = G.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((alpha * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:],
                            mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return float(y.item()), cam


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_overlay(volume: np.ndarray, heatmap: np.ndarray, meta: dict,
                 out_path: str) -> None:
    """Save 3×3 grid: axial/coronal/sagittal of PET + heatmap + overlay."""
    vol = volume.squeeze()                        # (128, 128, 128)
    D, H, W = vol.shape
    slices = {
        "Axial (z)":    (vol[D // 2, :, :], heatmap[D // 2, :, :]),
        "Coronal (y)":  (vol[:, H // 2, :], heatmap[:, H // 2, :]),
        "Sagittal (x)": (vol[:, :, W // 2], heatmap[:, :, W // 2]),
    }

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    fig.suptitle(
        f"Grad-CAM | tracer={meta['tracer']} | "
        f"CL_true={meta['cl_true']:.1f} | CL_pred={meta['cl_pred']:.1f} | "
        f"target={meta['target_layer']} | model={meta['model']}",
        fontsize=11, fontweight="bold",
    )

    for row, (title, (bg, hm)) in enumerate(slices.items()):
        axes[row, 0].imshow(bg, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{title} — PET", fontsize=10)
        axes[row, 1].imshow(hm, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{title} — Grad-CAM", fontsize=10)
        axes[row, 2].imshow(bg, cmap="gray", origin="lower")
        axes[row, 2].imshow(hm, cmap="hot", origin="lower",
                            alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title(f"{title} — overlay", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def pick_sample_indices(dataset, n_samples: int) -> list[int]:
    """Evenly-spaced across the CL range so we cover negative + high-CL cases."""
    if not dataset.has_targets:
        return list(range(min(n_samples, len(dataset))))
    cl = dataset.centiloids.numpy()
    order = np.argsort(cl)
    if n_samples >= len(dataset):
        return order.tolist()
    # linspace picks evenly spaced *ranks*, so we span the full CL distribution
    picks = np.linspace(0, len(order) - 1, n_samples).round().astype(int)
    return order[picks].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="dev/config/predict.yaml",
                    help="YAML with paths.csv + paths.checkpoint")
    ap.add_argument("--csv",          default=None, help="Override paths.csv")
    ap.add_argument("--checkpoint",   default=None, help="Override paths.checkpoint")
    ap.add_argument("--out_dir",      default="results/gradcam")
    ap.add_argument("--n_samples",    type=int, default=8,
                    help="Number of samples to visualize (evenly across CL range)")
    ap.add_argument("--target_layer", default=None,
                    help="Which module to hook. Default: film4 (petresnet) / "
                         "stage4 (no_film). Options: stage1-4, film1-4.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    csv_path  = args.csv        or cfg["paths"]["csv"]
    ckpt_path = args.checkpoint or cfg["paths"]["checkpoint"]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model, model_name, ckpt = load_model(ckpt_path, device)
    layer_name, target_module = resolve_target_layer(
        model, model_name, args.target_layer
    )
    print(f"Model        : {model_name}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Target layer : {layer_name}  ({type(target_module).__name__})")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = PETDataset(csv_path, tracer_map=ckpt["tracer_map"])
    indices = pick_sample_indices(dataset, args.n_samples)
    print(f"Visualising {len(indices)} samples → {args.out_dir}\n")

    inv_tracer = {v: k for k, v in dataset.tracer_map.items()}

    cam = GradCAM3D(model, target_module)
    try:
        for rank, idx in enumerate(indices):
            sample = dataset[idx]
            if len(sample) == 3:
                image, cl_true, tid = sample
                cl_true = float(cl_true)
            else:
                image, tid = sample
                cl_true = float("nan")

            x = image.unsqueeze(0).to(device).float()   # (1, 1, D, H, W)
            t = tid.unsqueeze(0).to(device)             # (1,)

            y_pred, heatmap = cam.compute(x, t)
            tracer_name = inv_tracer[int(tid.item())]

            out_name = (f"{rank:02d}_{tracer_name}_"
                        f"cl{cl_true:+.0f}_pred{y_pred:+.0f}.png".replace("+", "p")
                                                                 .replace("-", "n"))
            save_overlay(
                volume=x.squeeze(0).detach().cpu().numpy(),
                heatmap=heatmap,
                meta={
                    "tracer":       tracer_name,
                    "cl_true":      cl_true,
                    "cl_pred":      y_pred,
                    "target_layer": layer_name,
                    "model":        model_name,
                },
                out_path=os.path.join(args.out_dir, out_name),
            )
            print(f"  [{rank+1:>2}/{len(indices)}] {tracer_name:>3}  "
                  f"true={cl_true:+7.2f}  pred={y_pred:+7.2f}  → {out_name}")
    finally:
        cam.remove()

    print(f"\n✓ Done → {args.out_dir}")


if __name__ == "__main__":
    main()
