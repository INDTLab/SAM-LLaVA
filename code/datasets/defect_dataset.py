"""DefectDataset for SAM‑LLaVA.

The SAM‑LLaVA model requires paired image, mask and textual description inputs.
This dataset expects the following directory layout under a common root:

```
root/
├── images/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── masks/
│   ├── img_0001_mask.png
│   ├── img_0002_mask.png
│   └── ...
└── descriptions.json
```

* The `images/` directory contains RGB images.
* The `masks/` directory contains single‑channel binary masks with the same base
  filenames as the images.  Pixels with value 1 correspond to defects; 0
  otherwise.
* `descriptions.json` is a dictionary mapping image filenames to natural
  language descriptions.

You can create your own dataset following this layout.  The dataset returns a
dictionary with keys `image` (a tensor of shape (3,H,W) in the range [0,1]),
`mask` (tensor of shape (1,H,W) with values 0 or 1) and `description` (string).
Optionally, you can provide a list of indices for a support set used in
few‑shot settings; the dataset will return an additional key `support` containing
pairs of support images and masks.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def default_transform(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Resize a PIL image and convert it to a float tensor in [0,1]."""
    if image.size != size[::-1]:
        image = image.resize(size, Image.BILINEAR)
    arr = np.array(image).astype("float32") / 255.0
    # If grayscale, expand channel dimension
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # Rearrange to (C,H,W)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class DefectDataset(Dataset):
    """Dataset of defect images, masks and descriptions.

    Parameters
    ----------
    root : str
        Path to the dataset root.  Must contain `images`, `masks` and
        `descriptions.json`.
    transform : Optional[callable]
        Optional transform applied to both images and masks.  Defaults to
        converting images to tensors and resizing to 256×256.
    support_indices : Optional[List[int]]
        If provided, a list of indices used as the support set for few‑shot
        calibration.  When not empty the dataset returns an additional
        `support` field.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[callable] = None,
        support_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.root = root
        # Load descriptions
        desc_path = os.path.join(root, "descriptions.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                self.descriptions = json.load(f)
        else:
            self.descriptions = {}
        # List all image files
        img_dir = os.path.join(root, "images")
        self.img_files = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        # Set transform
        if transform is None:
            # Use our default transform
            self.transform = lambda img: default_transform(img, size=(256, 256))
        else:
            self.transform = transform
        # Support set
        self.support_indices = support_indices or []
        # Preload support samples to avoid repeated disk IO
        self._support_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if self.support_indices:
            for idx in self.support_indices:
                sample = self._load_sample(idx)
                self._support_cache.append((sample['image'], sample['mask']))

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_sample(self, index: int) -> dict:
        img_name = self.img_files[index]
        img_path = os.path.join(self.root, "images", img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(self.root, "masks", mask_name)
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Apply same transform to both
        image_t = self.transform(image)
        # Resize mask to match image
        if mask.size != (image_t.shape[2], image_t.shape[1]):
            mask = mask.resize((image_t.shape[2], image_t.shape[1]), Image.NEAREST)
        mask_arr = np.array(mask).astype("float32") / 255.0
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[:, :, None]
        mask_arr = np.transpose(mask_arr, (2, 0, 1))
        mask_t = torch.from_numpy(mask_arr)
        mask_t = (mask_t > 0.5).float()
        # Load description
        description = self.descriptions.get(img_name, "")
        return {
            "image": image_t,
            "mask": mask_t,
            "description": description,
        }

    def __getitem__(self, index: int) -> dict:
        sample = self._load_sample(index)
        # Add support set if required
        if self._support_cache:
            sample['support'] = self._support_cache
        return sample