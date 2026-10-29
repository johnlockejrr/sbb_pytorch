"""Data loading, augmentation and patch generation.

Four modules, one job each -- where the repository previously had the same job in five files:

===========================  ==========================================================
Module                       Replaces
===========================  ==========================================================
:mod:`~sbb_pytorch.data.dataset`       ``dataset.py`` (``SegmentationDataset``, ``get_dirs_or_files``)
:mod:`~sbb_pytorch.data.augmentations` ``augmentation.py`` *and* the inline ``A.Compose``
                                       inside ``dataset.DataModule``
:mod:`~sbb_pytorch.data.datamodule`    ``dataset.DataModule`` (the class, its preprocessing
                                       call, and its two loose-argument loaders)
:mod:`~sbb_pytorch.data.patches`       ``utils.provide_patches`` + its five patch writers
                                       *and* ``dataset.preprocess_and_augment_data``
===========================  ==========================================================

:func:`~sbb_pytorch.data.dataset.make_loader` is enough for training without Lightning;
:class:`~sbb_pytorch.data.datamodule.SBBDataModule` is what a ``pl.Trainer`` is handed.
"""

from __future__ import annotations

from sbb_pytorch.data.augmentations import (
    DEFAULT_PROBABILITY,
    PHOTOMETRIC_PROBABILITY,
    augment_pair,
    build_transform,
    geometric_transforms,
    normalizing_transforms,
    photometric_transforms,
    training_transform,
    validation_transform,
)
from sbb_pytorch.data.datamodule import SBBDataModule
from sbb_pytorch.data.dataset import (
    SegmentationDataset,
    label_path_for,
    list_image_pairs,
    make_loader,
    train_val_split,
)
from sbb_pytorch.data.patches import (
    BORDER_WIDTH,
    PatchConfig,
    Variants,
    composited_variants,
    dataset_for,
    generate,
    output_name,
    pad_to,
    pad_with_border,
    patch_slices,
    patches,
    preprocess_and_augment_data,
    variants,
    with_red_elements,
    write_patches,
)

__all__ = [
    "BORDER_WIDTH",
    "DEFAULT_PROBABILITY",
    "PHOTOMETRIC_PROBABILITY",
    "PatchConfig",
    "SBBDataModule",
    "SegmentationDataset",
    "Variants",
    "augment_pair",
    "build_transform",
    "composited_variants",
    "dataset_for",
    "generate",
    "geometric_transforms",
    "label_path_for",
    "list_image_pairs",
    "make_loader",
    "normalizing_transforms",
    "output_name",
    "pad_to",
    "pad_with_border",
    "patch_slices",
    "patches",
    "photometric_transforms",
    "preprocess_and_augment_data",
    "train_val_split",
    "training_transform",
    "variants",
    "validation_transform",
    "with_red_elements",
    "write_patches",
]
