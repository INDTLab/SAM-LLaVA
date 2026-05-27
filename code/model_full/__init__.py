"""Full SAM‑LLaVA model components.

This package contains classes that use the official CLIP ViT‑B/16, SAM ViT‑H
and Vicuna‑7B‑v1.5 models as described in the SAM‑LLaVA paper【409945085282613†L540-L546】.
They rely on external libraries (`transformers`, `segment_anything`, `peft`) and
assume that the necessary pretrained weights have been downloaded or will be
fetched automatically at runtime.

If you wish to run the full model, please install the required packages and
download the checkpoints as outlined in the README.
"""

from .clip_sam_cascade import ClipSamCascadeFull
from .llm import VicunaLLM
from .sam_llava_model import SamLlavaModelFull

__all__ = [
    "ClipSamCascadeFull",
    "VicunaLLM",
    "SamLlavaModelFull",
]