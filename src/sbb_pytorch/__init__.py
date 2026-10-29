"""PyTorch implementation of the SBB pixelwise document-image segmentation models.

The package covers the five SBB use cases (layout, textline, binarization, printspace,
glyph/word) plus the two adjacent tasks (image enhancement, machine-based reading order),
exposed as one :program:`sbb` command with ``train``, ``infer`` and ``gt`` subcommands.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
