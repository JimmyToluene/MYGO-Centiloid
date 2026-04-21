"""Model architecture subpackage."""

from mygo_centiloid.model.petresnet_film import PETResNet, BaselineCNN, TracerNorm, FiLMBlock, ResBlock3D
from mygo_centiloid.model.petresnet_no_film import PETResNetNoFiLM
from mygo_centiloid.model.petresnet_no_head_emb import PETResNetNoHeadEmb
from mygo_centiloid.model.petresnet_attn import PETResNetAttn, SpatialAttention3D
from mygo_centiloid.model.petresnet_attn_gated import PETResNetAttnGated, SpatialAttention3DGated

__all__ = [
    "PETResNet", "BaselineCNN", "TracerNorm", "FiLMBlock", "ResBlock3D",
    "PETResNetNoFiLM",
    "PETResNetNoHeadEmb",
    "PETResNetAttn", "SpatialAttention3D",
    "PETResNetAttnGated", "SpatialAttention3DGated",
]
