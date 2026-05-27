"""Dataset loading utilities for SAM‑LLaVA.

This package currently exposes a single dataset class, :class:`DefectDataset`, which
reads images, binary segmentation masks and textual descriptions from a common
directory structure.  Additional datasets can be added here with the same
PyTorch dataset interface.
"""

from .defect_dataset import DefectDataset  # noqa: F401