"""Augmentation pipelines -- the ``augmentation.py`` + ``DataModule`` pair, merged into one.

The module docstring claims all three legacy pipelines raise on the installed albumentations
2.0.8; the first tests here are that claim in executable form. The rest cover what neither
legacy module did: gating each transform on the config field that asks for it, and keeping the
offline (write-to-disk) path out of the ``Normalize``/``ToTensorV2`` tail.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from sbb_pytorch.config.models import BlurKind, SBBConfig, TaskType
from sbb_pytorch.data.augmentations import (
    DEFAULT_PROBABILITY,
    PHOTOMETRIC_PROBABILITY,
    _flips_for,
    augment_pair,
    build_transform,
    geometric_transforms,
    normalizing_transforms,
    photometric_transforms,
    training_transform,
    validation_transform,
)
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.storage.images import IMAGENET_MEAN, IMAGENET_STD

SIZE = 32
ALL_FLAGS = {
    "augmentation": True,
    "flip_aug": True,
    "flip_index": (1, 0, -1),
    "blur_aug": True,
    "blur_k": ("gauss", "median"),
    "brightening": True,
    "brightness": (1.8,),
    "degrading": True,
    "degrade_scales": (0.2, 0.5),
    "rotation": True,
    "rotation_not_90": True,
    "thetha": (5.0, -5.0),
    "channels_shuffling": True,
    "scaling": True,
    "scales": (0.6,),
}


def make_config(**overrides: object) -> SBBConfig:
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": SIZE,
        "input_width": SIZE,
        "augmentation": True,
        "dir_train": "/data/train",  # required by the config whenever augmentation is on
        **overrides,
    }
    return SBBConfig.model_validate(data)


#: Every optional augmentation, off -- ``plain()`` switches them off and then re-enables the one
#: a test is about, so a test proves the field it names and not the defaults around it.
OFF: dict[str, object] = {
    "flip_aug": False,
    "rotation": False,
    "rotation_not_90": False,
    "channels_shuffling": False,
    "scaling": False,
    "blur_aug": False,
    "brightening": False,
    "degrading": False,
}


def plain(**overrides: object) -> SBBConfig:
    """A config with every optional augmentation switched off, then ``overrides`` on."""
    return make_config(**{**OFF, **overrides})


@pytest.fixture
def pair() -> tuple[np.ndarray, np.ndarray]:
    image = np.tile(np.linspace(0, 255, SIZE, dtype=np.uint8), (SIZE, 1))
    image = np.stack([image, image // 2, image // 4], axis=-1)
    mask = np.tile(np.arange(SIZE, dtype=np.uint8) % 3, (SIZE, 1))
    return image, mask


def classes(transforms: list[A.BasicTransform]) -> set[str]:
    return {type(transform).__name__ for transform in transforms}


# ------------------------------------------------------------- the albumentations 2.0.8 reality


def test_the_legacy_constructor_calls_do_not_exist_anymore() -> None:
    """``A.Flip(p=0.5)`` and ``A.set_seed(7)`` are gone in 2.0.8 and
    ``A.RandomResizedCrop(height=, width=)`` now wants ``size=`` -- so the legacy pipelines
    raised ``AttributeError``/``ValueError`` before a single image was read. The module uses the
    surviving spellings, which is why it builds at all."""
    assert not hasattr(A, "Flip"), "A.Flip came back -- the module's replacement can be simplified"
    assert not hasattr(A, "set_seed"), "A.set_seed came back -- _compose could stop branching on None"
    with pytest.raises(ValueError, match="size"):
        A.RandomResizedCrop(height=SIZE, width=SIZE)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [(BlurKind.GAUSS, "GaussianBlur"), (BlurKind.MEDIAN, "MedianBlur"), (BlurKind.BLUR, "Blur")],
)
def test_every_blur_kind_maps_to_its_transform(kind: BlurKind, expected: str) -> None:
    """The legacy config value ``"guass"`` hit an unbound local; the enum member now names the
    class, so a member without a class is a test failure instead of a runtime one."""
    transforms = photometric_transforms(make_config(blur_aug=True, blur_k=(kind,)))
    assert type(transforms[0]).__name__ == expected


# -------------------------------------------------------------------- what each flag switches on


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # The sizing transform (``Resize``, or ``RandomResizedCrop`` with ``scaling``) is always
        # first; the rest of the set is what the flags below switch on.
        ({"flip_aug": True, "flip_index": (1,)}, {"Resize", "HorizontalFlip"}),
        ({"flip_aug": True, "flip_index": (0,)}, {"Resize", "VerticalFlip"}),
        ({"flip_aug": True, "flip_index": (-1,)}, {"Resize", "HorizontalFlip", "VerticalFlip"}),
        ({"rotation": True}, {"Resize", "RandomRotate90"}),
        ({"rotation_not_90": True, "thetha": (5.0, -5.0)}, {"Resize", "Rotate"}),
        ({"channels_shuffling": True}, {"Resize", "ChannelShuffle"}),
        ({"scaling": True, "scales": (0.6,)}, {"RandomResizedCrop"}),
        ({}, {"Resize"}),
    ],
)
def test_geometric_transforms_are_gated_on_the_config_fields(
    overrides: dict[str, object], expected: set[str]
) -> None:
    """``DataModule``'s inline pipeline applied flips and rotations to *every* run, ignoring the
    config; each transform now appears when -- and only when -- its field asks for it."""
    assert classes(geometric_transforms(plain(**overrides))) == expected


@pytest.mark.parametrize(
    ("codes", "expected"),
    [((1,), {"HorizontalFlip"}), ((0,), {"VerticalFlip"}), ((-1,), {"HorizontalFlip", "VerticalFlip"})],
)
def test_flip_codes_map_the_three_cv2_variants(codes: tuple[int, ...], expected: set[str]) -> None:
    """``cv2.flip`` code ``-1`` is *both* axes, which albumentations needs two transforms for."""
    assert classes(_flips_for(codes, DEFAULT_PROBABILITY)) == expected


def test_an_unknown_flip_code_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="not one of 1"):
        _flips_for((2,), DEFAULT_PROBABILITY)


def test_eval_geometry_is_a_resize_and_nothing_else() -> None:
    """Both legacy modules agreed on the validation pipeline, so it must stay free of any
    stochastic transform even when ``augmentation`` is on."""
    assert classes(geometric_transforms(make_config(**ALL_FLAGS), training=False)) == {"Resize"}


# --------------------------------------------------------------------- the photometric half


def test_photometric_limits_come_from_the_config() -> None:
    """``brightness=1.8`` and ``degrade_scales=(0.2, 0.5)`` used to be read by a parallel
    ``AugmentationConfig`` that training never consulted; they now reach the transforms."""
    config = make_config(blur_aug=True, blur_k=("gauss",), brightening=True, brightness=(1.8,),
                         degrading=True, degrade_scales=(0.2, 0.5))
    transforms = {(type(t).__name__): t for t in photometric_transforms(config)}
    assert transforms["RandomBrightnessContrast"].brightness_limit == (0.0, 0.8)
    assert transforms["RandomBrightnessContrast"].p == PHOTOMETRIC_PROBABILITY
    assert transforms["GaussNoise"].std_range == (0.2, 0.5)


def test_several_blur_kinds_become_one_oneof() -> None:
    """``augmentation.py`` applied *both* configured blurs in sequence; one ``OneOf`` is what the
    field means ("pick one of these blur kinds"), and duplicates collapse."""
    single = photometric_transforms(make_config(blur_aug=True, blur_k=("gauss", "gauss")))
    assert "OneOf" not in classes(single), "a duplicated kind should not build a OneOf"
    several = photometric_transforms(make_config(blur_aug=True, blur_k=("gauss", "median")))
    one_of = next(t for t in several if isinstance(t, A.OneOf))
    assert classes(list(one_of.transforms)) == {"GaussianBlur", "MedianBlur"}


def test_enhancement_adds_brightness_contrast_without_the_flag() -> None:
    """``get_enhancement_augmentations`` had its own recipe; the task still refines the set."""
    config = make_config(task=TaskType.ENHANCEMENT, n_classes=3, brightening=False)
    assert "RandomBrightnessContrast" in classes(photometric_transforms(config))


def test_every_stochastic_transform_uses_the_legacy_probability() -> None:
    """0.5 everywhere, 0.2 for brightness/contrast -- the two numbers both legacy modules shared."""
    config = make_config(**ALL_FLAGS)
    for transform in geometric_transforms(config)[1:]:  # [0] is the deterministic Resize/crop
        assert transform.p == DEFAULT_PROBABILITY, type(transform).__name__
    for transform in photometric_transforms(config):
        expected = (
            PHOTOMETRIC_PROBABILITY
            if isinstance(transform, A.RandomBrightnessContrast)
            else DEFAULT_PROBABILITY
        )
        assert transform.p == expected, type(transform).__name__


# --------------------------------------------------------------------------- composing the whole


def test_normalizing_transforms_apply_the_imagenet_statistics(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Swapping mean and std (or normalising in BGR after a cv2 read) changes every pixel."""
    image, _ = pair
    mean = np.asarray(IMAGENET_MEAN, "float32")
    std = np.asarray(IMAGENET_STD, "float32")
    out = A.Compose(normalizing_transforms(make_config()))(image=image)["image"]
    expected = ((image.astype("float32") / 255.0) - mean) / std
    np.testing.assert_allclose(out.numpy(), expected.transpose(2, 0, 1), rtol=1e-4, atol=1e-3)


def test_augment_pair_stays_numpy_uint8_at_the_configured_size(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """The offline path used to normalise the image and then write *that* back to disk, so the
    generated files were unviewable and unusable as inputs to the next run."""
    image, mask = pair
    out_image, out_mask = augment_pair(image, mask, make_config(**ALL_FLAGS), seed=7)
    assert out_image.shape == (SIZE, SIZE, 3) and out_image.dtype == np.uint8
    assert out_mask.shape == (SIZE, SIZE) and out_mask.dtype == np.uint8
    assert set(out_mask.tolist()[0]) <= {0, 1, 2}, "spatial transforms invented class ids"


def test_a_seed_makes_the_augmentation_reproducible(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """``A.set_seed`` is gone; the per-compose ``seed`` is the supported equivalent, and it is
    what makes two runs from one config comparable."""
    image, mask = pair
    config = plain(flip_aug=True, flip_index=(-1,), scaling=True, scales=(0.6,), seed=7)
    first = augment_pair(image, mask, config)
    again = augment_pair(image, mask, config)
    other = augment_pair(image, mask, config, seed=11)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0]), "the seed was ignored"


@pytest.mark.parametrize(
    ("training", "expected"),
    [
        (
            True,
            {
                "RandomResizedCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90", "Rotate",
                "ChannelShuffle", "OneOf", "RandomBrightnessContrast", "GaussNoise", "Normalize",
                "ToTensorV2",
            },
        ),
        (False, {"Resize", "Normalize", "ToTensorV2"}),
    ],
)
def test_build_transform_composes_the_requested_pipeline(training: bool, expected: set[str]) -> None:
    """One entry point, two pipelines -- and the only difference from ``augmentation=false`` is
    the geometry and photometry, never the tensor tail."""
    built = build_transform(make_config(**ALL_FLAGS), training=training)
    assert classes(built.transforms) == expected


def test_augmentation_false_makes_training_the_eval_pipeline() -> None:
    """``augmentation=false`` has to switch *everything* off, not just the sub-flags -- the bug
    ``DataModule`` shipped with was a training pipeline that always augmented."""
    config = make_config(augmentation=False, flip_aug=True, blur_aug=True, rotation=True)
    assert classes(training_transform(config).transforms) == classes(validation_transform(config).transforms)
