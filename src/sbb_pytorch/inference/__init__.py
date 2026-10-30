"""Inference: load a trained checkpoint, predict on pages, render the result.

Replaces ``inference_pytorch.py``. Two pieces, one job each:

* :class:`~sbb_pytorch.inference.predictor.Predictor` -- build the model from a run directory
  (weights *and* the config recorded beside them), then predict on one page given as a file or an
  array, either whole-page or through a sliding grid of tiles. It returns a
  :class:`~sbb_pytorch.inference.predictor.Prediction`. A directory of pages is the CLI's loop
  over ``predict_file``, not a method on the predictor.
* :mod:`~sbb_pytorch.inference.visualize` -- every rendering the CLI writes (overlay, colourised
  layout, binarised page, enhanced page) from the one palette the ``gt`` package also uses.
"""

from __future__ import annotations

from sbb_pytorch.inference.predictor import Prediction, Predictor
from sbb_pytorch.inference.visualize import (
    OVERLAY_ALPHA,
    binarization_image,
    colorize,
    enhancement_image,
    overlay,
    render,
    save_images,
)

__all__ = [
    "OVERLAY_ALPHA",
    "Prediction",
    "Predictor",
    "binarization_image",
    "colorize",
    "enhancement_image",
    "overlay",
    "render",
    "save_images",
]
