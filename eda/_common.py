"""
Shared constants and helpers for the EDA suite.

Single source of truth for cross-script constants (amyloid threshold,
tracer ordering, color palette, plot style). Importing from this module
eliminates the drift risk of redefining these in every script.

Usage in any eda/NN_*.py:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import (
        AMYLOID_POS_THRESHOLD, TRACER_ORDER, TRACER_NAMES,
        TRACER_COLORS, FOREGROUND_THRESH, setup_style,
    )
"""

import seaborn as sns

# ── Clinical constants ────────────────────────────────────────────────────────

# Centiloid threshold for amyloid positivity.
# Source: Klunk et al., Alzheimer's & Dementia 2015 (≥24.4 CL ≈ moderate plaque).
AMYLOID_POS_THRESHOLD: float = 24.4

# Foreground voxel threshold used for "brain" mask in calibration analysis.
# Volumes are min-max normalized to [0,1]; voxels above 0.05 are treated as
# brain tissue rather than air / zero-padded background.
FOREGROUND_THRESH: float = 0.05


# ── Tracer metadata ───────────────────────────────────────────────────────────

TRACER_ORDER: list[str] = ["FBP", "FBB", "NAV", "PIB"]

TRACER_NAMES: dict[str, str] = {
    "FBP": "Florbetapir",
    "FBB": "Florbetaben",
    "NAV": "Florbetanav",
    "PIB": "Pittsburgh\nCompound B",
}

TRACER_COLORS: dict[str, tuple] = dict(
    zip(TRACER_ORDER, sns.color_palette("Set1", 4))
)


# ── Plot style ────────────────────────────────────────────────────────────────

def setup_style(font_scale: float = 1.2) -> None:
    """Apply the project-wide seaborn theme. Call once per script in main()."""
    sns.set_theme(style="whitegrid", font_scale=font_scale)


# ── Logger ────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "info") -> None:
    """Lightweight logger used across EDA scripts (avoids stdlib logging setup)."""
    prefix = {
        "info": "   ",
        "ok"  : "  ✓",
        "warn": "  ⚠",
        "save": "  →",
        "skip": "  ·",
    }.get(level, "   ")
    print(f"{prefix} {msg}")
