"""Multimodal fusion module.

Canonical location for fusion logic.  The legacy ``utils/fusion_engine``
module is a thin re-export shim for backward compatibility.
"""

from modules.fusion.fusion_engine import (  # noqa: F401
    FusionWeights,
    PRESETS,
    fusion_engine,
    fuse_scores,
    save_fusion_output,
)
