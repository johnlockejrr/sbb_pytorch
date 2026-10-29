"""Image and checkpoint I/O -- the single source of truth for both.

Before the refactor, ``resize_image`` existed three times, ``cv2.imread`` was called without
a ``None`` check at 30+ sites, and three different checkpoint layouts were in circulation at
once (``model_state`` from ``model_utils.save_model``, ``model_state_dict`` from ``train.py``,
and bare state dicts). Training a model with one module and loading it with another failed on
a key name. Both problems are resolved here: one module per concern, one layout each.
"""

from __future__ import annotations

from sbb_pytorch.storage.checkpoint import (
    Checkpoint,
    latest_checkpoint,
    load_checkpoint,
    load_into_model,
    prune_checkpoints,
    save_checkpoint,
    write_sidecar_config,
)
from sbb_pytorch.storage.images import (
    BlurKind,
    ShiftDirection,
    brighten,
    downscale_upscale,
    imwrite,
    read_image,
    read_mask,
    remap_mask,
    resize,
    resize_mask,
    rotate90,
    rotate_not_90,
    rotated_rect_max_area,
    shift_image_and_label,
    shuffle_channels,
    to_uint8,
)

__all__ = [
    "BlurKind",
    "Checkpoint",
    "ShiftDirection",
    "brighten",
    "downscale_upscale",
    "imwrite",
    "latest_checkpoint",
    "load_checkpoint",
    "load_into_model",
    "prune_checkpoints",
    "read_image",
    "read_mask",
    "remap_mask",
    "resize",
    "resize_mask",
    "rotate90",
    "rotate_not_90",
    "rotated_rect_max_area",
    "save_checkpoint",
    "shift_image_and_label",
    "shuffle_channels",
    "to_uint8",
    "write_sidecar_config",
]
