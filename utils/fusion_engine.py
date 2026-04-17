"""Backward-compatibility shim.

Fusion logic has moved to ``modules.fusion.fusion_engine``.
This module re-exports everything so existing callers don't break.
"""

from modules.fusion.fusion_engine import (  # noqa: F401
    FusionWeights,
    PRESETS,
    fusion_engine,
    fuse_scores,
    save_fusion_output,
    load_json,
    _normalize,
)