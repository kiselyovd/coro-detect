"""Deterministic seeding across libraries."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore[import-not-found]  # template stub; revisit in backport

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
