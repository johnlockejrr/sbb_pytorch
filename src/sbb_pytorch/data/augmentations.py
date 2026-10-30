"""One albumentations factory, and the reason the old two could not both survive.

The repository built augmentation pipelines in **two** places that never talked to each other:
``augmentation.py`` (``get_training_augmentations`` / ``get_validation_augmentations`` /
``get_enhancement_augmentations`` / ``get_augmentations``) and a hard-coded ``A.Compose`` inside
``dataset.DataModule.__init__``. Training used the second one, so the first was dead code -- and
the second ignored every augmentation flag the config exposes, applying flips and rotations even
to a run that asked for ``augmentation=false``.

**All three of ``augmentation.py``'s pipelines raise on the installed albumentations 2.0.8**,
verified against the package rather than from memory:

=========================================  ================================================
legacy call                                albumentations 2.0.8
=========================================  ================================================
``A.Flip(p=0.5)``                          ``AttributeError`` -- removed; ``HorizontalFlip``
                                           / ``VerticalFlip`` replace it
``A.RandomResizedCrop(height=224, ...)``   ``ValueError: size Field required`` -- now ``size=``
``A.GaussNoise(var_limit=(x, y))``         accepted, deprecated; ``std_range`` is the new name
``A.ElasticTransform(alpha_affine=...)``   accepted, deprecated
``A.OpticalDistortion(shift_limit=0.5)``   accepted, deprecated
``A.set_seed(7)``                          ``AttributeError`` -- gone; ``A.Compose(seed=7)``
=========================================  ================================================

So "preserve the behaviour" here cannot mean copying the constructor call: the pipeline it built
has never once run in this environment. What is preserved is the *set of transforms and their
probabilities* -- ``augmentation.py``'s transform list at ``p=0.5`` (and ``0.2`` for
brightness/contrast, the one value ``DataModule`` and ``augmentation.py`` shared) -- while each
transform is now gated on the config field that asks for it. The crop size, flip codes, blur
kinds, rotation angles and degradation strengths all come from :class:`~sbb_pytorch.config.models.SBBConfig`
instead of a parallel ``AugmentationConfig`` class with its own defaults, which is what made the
two modules disagree in the first place.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from sbb_pytorch.config.models import BlurKind, SBBConfig, TaskType
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.storage.images import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "DEFAULT_PROBABILITY",
    "PHOTOMETRIC_PROBABILITY",
    "augment_pair",
    "build_transform",
    "geometric_transforms",
    "normalizing_transforms",
    "training_transform",
    "validation_transform",
]

#: Probability used for every stochastic transform. ``augmentation.py`` used ``0.5`` for all of
#: them and ``DataModule`` used ``0.5`` for its flips/rotations -- the single shared value.
DEFAULT_PROBABILITY = 0.5

#: ``DataModule`` and ``augmentation.py`` both used ``0.2`` for brightness/contrast specifically.
PHOTOMETRIC_PROBABILITY = 0.2

_BLUR_TRANSFORMS: dict[BlurKind, type[A.BasicTransform]] = {}


def _blur_transform(kind: BlurKind) -> A.BasicTransform:
    """Instantiate one :class:`~sbb_pytorch.config.models.BlurKind` as its albumentations class.

    The legacy config spelled this value ``"guass"`` and the code that read it hit an
    unbound-local; here the enum member names the class, so a new blur kind is one enum entry.
    """
    global _BLUR_TRANSFORMS
    if not _BLUR_TRANSFORMS:
        _BLUR_TRANSFORMS = {
            BlurKind.GAUSS: A.GaussianBlur,
            BlurKind.MEDIAN: A.MedianBlur,
            BlurKind.BLUR: A.Blur,
        }
    return _BLUR_TRANSFORMS[kind](blur_limit=3, p=DEFAULT_PROBABILITY)


def normalizing_transforms(config: SBBConfig) -> list[A.BasicTransform]:
    """``Normalize`` + ``ToTensorV2`` -- the tail both legacy pipelines agreed on exactly.

    The statistics are the ImageNet ones the models were pretrained with
    (:data:`~sbb_pytorch.storage.images.IMAGENET_MEAN`), the same pair
    ``storage.images.normalize`` applies to a numpy array; going through albumentations here
    keeps the tensor layout (``ToTensorV2``) that the old ``F.interpolate`` fix-ups assumed.
    """
    return [
        A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist(), max_pixel_value=255.0),
        ToTensorV2(),
    ]


def _flips_for(codes: tuple[int, ...], p: float) -> list[A.BasicTransform]:
    """Translate the legacy ``cv2.flip`` codes in ``config.flip_index`` to albumentations flips.

    ``cv2.flip`` had three codes -- ``1`` horizontal, ``0`` vertical, ``-1`` both -- while
    albumentations splits them into two transforms. Emitting ``-1`` therefore produces *both*
    transforms here, which is what ``cv2.flip(img, -1)`` did. An unknown code is a config error
    rather than a silent no-op, matching the validator's stance on the same field.
    """
    flips: list[A.BasicTransform] = []
    for code in codes:
        if code == 1:
            flips.append(A.HorizontalFlip(p=p))
        elif code == 0:
            flips.append(A.VerticalFlip(p=p))
        elif code == -1:
            flips.extend([A.HorizontalFlip(p=p), A.VerticalFlip(p=p)])
        else:
            raise ConfigError(
                f"flip_index code {code} is not one of 1 (horizontal), 0 (vertical), -1 (both)"
            )
    return flips


def geometric_transforms(config: SBBConfig, *, training: bool = True) -> list[A.BasicTransform]:
    """Every spatial transform the config asks for, in the order albumentations expects.

    Split out from :func:`build_transform` because the offline path
    (:func:`augment_pair`, used by ``data.patches``) needs the geometry *without* the
    ``Normalize``/``ToTensorV2`` tail -- normalising an image and then writing it back to disk is
    what turned ``provide_patches`` output into unreusable files.

    The crop/resize choice is the one place ``augmentation.py`` and ``DataModule`` disagreed:
    the former cropped a random region scaled by ``scale``, the latter resized exactly. The
    config's own ``scaling`` flag decides now -- with ``scaling`` on, ``RandomResizedCrop`` at
    ``(min(scales), 1.0)``; without it, a plain ``Resize`` to the configured size.

    ``scaling_flip`` and ``scaling_bluring`` are absent on purpose. In the legacy code both live
    in the patch-writing branch only (``utils.py:1033`` and ``utils.py:1048``, each emitting one
    file per scale/blurring or scale/flip combination, which :mod:`sbb_pytorch.data.patches`
    reproduces at ``patches.py:646`` and ``patches.py:681``); neither has a runtime-augmentation
    meaning, so this function ignores them rather than inventing one.
    """
    height, width = config.image_size
    if not training:
        return [A.Resize(height=height, width=width)]

    transforms: list[A.BasicTransform] = []
    if config.scaling:
        transforms.append(
            A.RandomResizedCrop(size=(height, width), scale=(min(config.scales), 1.0), p=1.0)
        )
    else:
        transforms.append(A.Resize(height=height, width=width))

    if config.flip_aug:
        transforms.extend(_flips_for(config.flip_index, DEFAULT_PROBABILITY))
    if config.rotation:
        transforms.append(A.RandomRotate90(p=DEFAULT_PROBABILITY))
    if config.rotation_not_90:
        angles = config.thetha
        transforms.append(
            A.Rotate(limit=(min(angles), max(angles)), p=DEFAULT_PROBABILITY, border_mode=0)
        )
    if config.channels_shuffling:
        transforms.append(A.ChannelShuffle(p=DEFAULT_PROBABILITY))
    return transforms


def photometric_transforms(config: SBBConfig) -> list[A.BasicTransform]:
    """Intensity-side transforms: blur, brightness/contrast, noise, enhancement extras.

    ``config.degrade_scales`` are fractions of the pixel range (``0.2`` = 20%), which is exactly
    how albumentations reads ``GaussNoise(std_range=...)`` -- the legacy ``do_degrading`` added
    ``normal(0, 0.2)`` in 0-255 units instead, i.e. ~nothing. See ``docs/migration.md``.
    """
    transforms: list[A.BasicTransform] = []
    if config.blur_aug:
        kinds = list(dict.fromkeys(config.blur_k))  # de-duplicate, keep order
        if len(kinds) == 1:
            transforms.append(_blur_transform(kinds[0]))
        else:
            transforms.append(A.OneOf([_blur_transform(k) for k in kinds], p=DEFAULT_PROBABILITY))

    if config.brightening or config.task is TaskType.ENHANCEMENT:
        limit = max(config.brightness) - 1.0  # factors 1.3..2.0 -> a 0.2..1.0 relative limit
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(0.0, limit),
                contrast_limit=0.2,
                p=PHOTOMETRIC_PROBABILITY,
            )
        )

    if config.degrading:
        transforms.append(
            A.GaussNoise(
                std_range=(min(config.degrade_scales), max(config.degrade_scales)),
                p=DEFAULT_PROBABILITY,
            )
        )
    return transforms


def training_transform(config: SBBConfig, *, seed: int | None = None) -> A.Compose:
    """The training pipeline for ``config`` (``augmentation=false`` yields the eval pipeline).

    Raising on ``seed`` handling: albumentations 2.x removed ``A.set_seed``; the per-compose
    ``seed=`` argument is the supported way to make a pipeline reproducible, and it is passed
    through whenever the caller (or ``config.seed``) supplies one.
    """
    if not config.augmentation:
        return validation_transform(config, seed=seed)
    transforms = [
        *geometric_transforms(config),
        *photometric_transforms(config),
        *normalizing_transforms(config),
    ]
    return _compose(transforms, seed=seed if seed is not None else config.seed)


def validation_transform(config: SBBConfig, *, seed: int | None = None) -> A.Compose:
    """Resize + normalise + tensor. Identical in both legacy modules, so nothing to reconcile."""
    return _compose(
        [*geometric_transforms(config, training=False), *normalizing_transforms(config)],
        seed=seed if seed is not None else config.seed,
    )


def build_transform(
    config: SBBConfig, *, training: bool = True, seed: int | None = None
) -> A.Compose:
    """The single entry point: ``training_transform`` or ``validation_transform``.

    ``get_augmentations(config, task, is_training)`` was the legacy equivalent; it branched on a
    ``task`` *string* (``"enhancement"``) that bypassed its own ``AugmentationConfig`` defaults,
    so the same config produced different augmentation in two call sites. The task now arrives
    through ``config.task`` and only refines the photometric set.
    """
    if training:
        return training_transform(config, seed=seed)
    return validation_transform(config, seed=seed)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: SBBConfig,
    *,
    training: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the *geometric* part of the pipeline to a pair, staying in numpy.

    Returns ``(image_uint8, mask_uint8)`` at the configured size -- no normalisation, no tensor,
    because the offline augmentation path writes files back to disk. Masks keep their stored
    values: every spatial transform here resizes masks with nearest-neighbour interpolation,
    which is the invariant ``storage.images.resize_mask`` was written to guarantee (the legacy
    ``rotation_not_90_func`` warped the label with the default bilinear interpolation and
    invented class ids between the real ones).
    """
    composed = _compose(
        geometric_transforms(config, training=training),
        seed=seed if seed is not None else config.seed,
    )
    out = composed(image=np.asarray(image), mask=np.asarray(mask))
    return out["image"], out["mask"]


def _compose(transforms: list[A.BasicTransform], *, seed: int | None) -> A.Compose:
    """``A.Compose`` with the optional seed passed only when there is one (``None`` means random)."""
    if seed is None:
        return A.Compose(transforms)
    return A.Compose(transforms, seed=int(seed))
