"""Model components for SAM‑LLaVA.

The high‑level architecture of SAM‑LLaVA consists of several loosely coupled
modules:

* **CLIP–SAM cascade** – Generates a coarse defect mask using a CLIP‑like
  encoder and refines it with a SAM‑like network【409945085282613†L324-L336】.
* **Multi‑scale prompt learner** – Builds multi‑scale contextual prompts from
  the coarse mask【409945085282613†L392-L405】.
* **Mask decoder** – Produces segmentation outputs at multiple resolutions【409945085282613†L392-L425】.
* **Segmentation‑Aware Semantic Alignment (SASA)** – Aligns mask‑filtered
  visual features with token embeddings via bidirectional cross‑attention【409945085282613†L423-L455】.
* **Language model** – Generates textual descriptions conditioned on the aligned
  features【409945085282613†L456-L464】.

Each module is implemented in a separate file for clarity.  See the individual
modules for detailed documentation and usage.
"""

from .clip_sam_cascade import ClipSamCascade
from .multi_scale_prompt import MultiScalePromptLearner
from .mask_decoder import MaskDecoder
from .sasa_module import SASAModule
from .llm import SimpleTokenizer, SimpleLLM
from .sam_llava_model import SamLlavaModel

__all__ = [
    "ClipSamCascade",
    "MultiScalePromptLearner",
    "MaskDecoder",
    "SASAModule",
    "SimpleTokenizer",
    "SimpleLLM",
    "SamLlavaModel",
]