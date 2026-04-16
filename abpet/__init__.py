"""
abpet — Amyloid β-PET Centiloid Prediction package.

Public API (all callers should import from here, not from submodules):

    from abpet import PETResNet, BaselineCNN          # architecture
    from abpet import PETDataset                       # data
    from abpet import build_train_transform            # augmentation
    from abpet import CentiloidLoss, get_criterion     # losses
"""

from abpet.data.dataset      import PETDataset
from abpet.data.augmentation import build_train_transform
from abpet.model.petresnet   import PETResNet, BaselineCNN
from abpet.nn.losses         import CentiloidLoss, get_criterion

__all__ = [
    "PETDataset",
    "build_train_transform",
    "PETResNet",
    "BaselineCNN",
    "CentiloidLoss",
    "get_criterion",
]

__version__ = "0.1.0"
