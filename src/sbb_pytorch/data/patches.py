"""Offline augmentation and patch extraction -- the ``provide_patches`` family.

``utils.py`` exposed this as one function of **43 positional arguments**
(``provide_patches(imgs_list_train, segs_list_train, dir_img, dir_seg, ...)``, called from
``train.py`` with a 30-line argument list) plus five near-identical patch writers
(``get_patches``, ``get_patches_num_scale``, ``get_patches_num_scale_new``, and the inline
resize-and-write inside ``dataset.preprocess_and_augment_data``). The arguments were not wrong
so much as unnameable: a caller could not tell ``scales`` from ``degrade_scales`` from
``scaling_brightness``, and two of them (``shifting``, ``scaling_brightness``) had no config
field at all, while ``dataset.py`` reimplemented the same matrix with *different* file names
(``{name}_flip_{code}.png`` instead of ``img_{n}.png``).

Now: one :class:`PatchConfig` holding every field, one :func:`variants` function that decides
what to produce, one :func:`write_patches` that writes them. Every legacy field is kept --
including the two the config never carried -- and both naming schemes survive, because run
directories already on disk use them.

Fixed while merging the copies:

* ``img_name = im.split('.')[0]`` truncated ``page.001.png`` to ``page``, so all the
  ``page.00N`` variants of one document resolved to a single label and overwrote each other.
  The stem-based :func:`~sbb_pytorch.data.dataset.label_path_for` is used instead.
* The two ``do_degrading`` copies disagreed -- ``utils.py`` downscaled and upscaled, while
  ``dataset.py`` added noise and a blur. ``utils.py``'s is the one ``provide_patches`` called,
  so :func:`~sbb_pytorch.storage.images.downscale_upscale` stands in for it; ``add_noise``
  covers the other.
* ``return_binary_image_with_given_rgb_background_and_given_foreground_rgb`` indexed a
  ``(H, W, 3)`` foreground array with ``rgb_foreground[0]`` -- one *row* -- and assigned it to a
  boolean-masked slice, which only works when the mask selects exactly one pixel. It now takes
  the per-pixel array and applies it where the binarised page is background.
* Padding built ``float64``/``int16`` canvases and a bare ``except: pass`` around the
  brightening variants swallowed every error, including "file unreadable". Padding returns
  ``uint8`` and failures are logged with the path.
* ``random.choice`` for the background/foreground picks ran on global RNG state, so a rerun of
  the same config produced different data. :class:`PatchConfig.seed` makes it reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sbb_pytorch.config.models import SBBConfig, TaskType
from sbb_pytorch.data.dataset import SegmentationDataset, label_path_for, list_image_pairs
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.logging import get_logger
from sbb_pytorch.storage.images import (
    BlurKind,
    ShiftDirection,
    blur,
    brighten,
    downscale_upscale,
    imwrite,
    list_images,
    otsu_binarize,
    read_image,
    read_mask,
    resize,
    resize_mask,
    rotate90,
    rotate_not_90,
    shift_image_and_label,
    shuffle_channels,
)

__all__ = [
    "BORDER_WIDTH",
    "PatchConfig",
    "Variants",
    "composited_variants",
    "dataset_for",
    "generate",
    "output_name",
    "pad_to",
    "pad_with_border",
    "patch_slices",
    "patches",
    "preprocess_and_augment_data",
    "variants",
    "with_red_elements",
    "write_patches",
]

logger = get_logger("sbb.data.patches")

#: Frame width the legacy ``do_padding_white`` / ``do_padding_black`` / ``do_padding_label``
#: added on every side -- a hard-coded 4 pixels, never configurable, never documented.
BORDER_WIDTH = 4


class PatchConfig(BaseModel):
    """Every argument ``provide_patches`` took, named -- and buildable from an ``SBBConfig``.

    The legacy call had 43 positional arguments; :meth:`from_sbb` copies the ones
    :class:`~sbb_pytorch.config.models.SBBConfig` already owns (input size, ``task``, and the
    dozen ``*_aug`` / ``*_scales`` / ``*_index`` knobs) so the two cannot drift apart, and the
    rest keep the values ``train.py`` passed.

    Two legacy arguments have no counterpart in ``SBBConfig`` and are recorded here as the gap
    they are: ``shifting`` and ``scaling_brightness``. ``SBBConfig`` also has no
    ``padding_white`` / ``padding_black``, so those default to ``False`` here -- asking for one
    through ``PatchConfig`` still works, which is what an existing run directory needs.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ------------------------------------------------------------- what to read
    dir_img: str | None = None
    dir_seg: str | None = None
    #: Pre-binarised pages, needed by ``adding_rgb_*`` and ``add_red_textlines``.
    dir_img_bin: str | None = None
    #: Explicit file lists, for callers that already enumerated them (``train.py`` did).
    imgs_list: tuple[str, ...] = ()
    segs_list: tuple[str, ...] = ()

    # ------------------------------------------------------------- where to write
    dir_output: str = "runs/default"
    #: ``provide_patches`` wrote into ``dir_flow_train_imgs`` / ``dir_flow_train_labels``;
    #: :attr:`images_output` derives them from ``dir_output`` unless these are set.
    dir_flow_train_imgs: str | None = None
    dir_flow_train_labels: str | None = None

    # ------------------------------------------------------------- sizes
    input_height: int = Field(default=448, gt=0)
    input_width: int = Field(default=448, gt=0)

    # ------------------------------------------------------------- mode
    task: TaskType = TaskType.SEGMENTATION
    patches: bool = False
    augmentation: bool = False

    # ------------------------------------------------------------- augmentation switches
    flip_aug: bool = False
    blur_aug: bool = False
    scaling: bool = False
    rotation: bool = False
    rotation_not_90: bool = False
    binarization: bool = False
    brightening: bool = False
    degrading: bool = False
    channels_shuffling: bool = False
    shifting: bool = False
    padding_white: bool = False
    padding_black: bool = False
    adding_rgb_background: bool = False
    adding_rgb_foreground: bool = False
    add_red_textlines: bool = False
    scaling_bluring: bool = False
    scaling_binarization: bool = False
    scaling_flip: bool = False
    scaling_brightness: bool = False

    # ------------------------------------------------------------- augmentation parameters
    blur_k: tuple[BlurKind, ...] = (BlurKind.BLUR, BlurKind.GAUSS, BlurKind.MEDIAN)
    scales: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    brightness: tuple[float, ...] = (1.3, 1.5, 1.7, 2.0)
    degrade_scales: tuple[float, ...] = (0.2, 0.4)
    flip_index: tuple[int, ...] = (0, 1, -1)
    shuffle_indexes: tuple[tuple[int, ...], ...] = ((0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0))
    thetha: tuple[float, ...] = (5.0, -5.0)
    number_of_backgrounds_per_image: int = Field(default=2, ge=1)

    # ------------------------------------------------------------- compositing sources
    dir_rgb_backgrounds: str | None = None
    dir_rgb_foregrounds: str | None = None
    #: Pre-resolved file names; otherwise listed from the directories above.
    list_all_possible_background_images: tuple[str, ...] = ()
    list_all_possible_foreground_rgbs: tuple[str, ...] = ()

    #: Seed for the background/foreground picks -- the legacy code used the global ``random``
    #: module, so re-running a config produced different data.
    seed: int | None = None

    @model_validator(mode="after")
    def _compositing_needs_a_source(self) -> PatchConfig:
        """``adding_rgb_background`` without a background folder used to crash mid-loop."""
        needs_backgrounds = self.adding_rgb_background or self.adding_rgb_foreground
        if needs_backgrounds and not (
            self.dir_rgb_backgrounds or self.list_all_possible_background_images
        ):
            raise ValueError(
                "adding_rgb_background/adding_rgb_foreground needs dir_rgb_backgrounds or an "
                "explicit list_all_possible_background_images"
            )
        if self.adding_rgb_foreground and not (
            self.dir_rgb_foregrounds or self.list_all_possible_foreground_rgbs
        ):
            raise ValueError(
                "adding_rgb_foreground needs dir_rgb_foregrounds or an explicit "
                "list_all_possible_foreground_rgbs"
            )
        if self.add_red_textlines and not self.dir_img_bin:
            raise ValueError("add_red_textlines needs dir_img_bin (the binarised image)")
        return self

    @classmethod
    def _split_folders(cls, config: SBBConfig) -> tuple[str, str]:
        """``(images_folder, labels_folder)`` for the split ``config`` points at.

        ``provide_patches`` wanted the two folders outright (``dir_img`` + ``dir_seg``), while an
        SBB config names a *split folder* -- ``<root>/train`` holding ``images/`` and ``labels/``,
        which is also what :class:`~sbb_pytorch.data.dataset.SegmentationDataset` reads. So the
        projection takes the subfolders when they are there. A ``dir_train`` with no ``labels/``
        subdirectory is taken to be the images folder itself and ``dir_eval`` the labels, the
        collapsed form of the legacy four-path call; before this rule existed, patch mode paired
        the training images with the *eval* images and silently generated nothing.
        """
        train = Path(config.dir_train)
        if not str(train):
            return "", config.dir_eval
        if (train / "labels").is_dir():
            images = train / "images" if (train / "images").is_dir() else train
            return str(images), str(train / "labels")
        return str(train), config.dir_eval

    @classmethod
    def from_sbb(cls, config: SBBConfig, **overrides: Any) -> PatchConfig:
        """Project an :class:`~sbb_pytorch.config.models.SBBConfig` onto a ``PatchConfig``.

        Replaces the ``train.py`` habit of pulling ~25 values out of a config dict one bracket
        at a time, where a wrong key name surfaced as a ``KeyError`` after training started.
        ``dir_img``/``dir_seg`` come from :meth:`_split_folders`, which is the only place that
        knows how an SBB directory is spelled.
        """
        images_folder, labels_folder = cls._split_folders(config)
        data: dict[str, Any] = {
            "dir_img": images_folder or None,
            "dir_seg": labels_folder or None,
            "dir_img_bin": config.dir_img_bin or None,
            "dir_output": config.dir_output,
            "dir_rgb_backgrounds": config.dir_rgb_backgrounds or None,
            "dir_rgb_foregrounds": config.dir_rgb_foregrounds or None,
            "input_height": config.input_height,
            "input_width": config.input_width,
            "task": config.task,
            "patches": config.patches,
            "augmentation": config.augmentation,
            "flip_aug": config.flip_aug,
            "blur_aug": config.blur_aug,
            "scaling": config.scaling,
            "rotation": config.rotation,
            "rotation_not_90": config.rotation_not_90,
            "binarization": config.binarization,
            "brightening": config.brightening,
            "degrading": config.degrading,
            "channels_shuffling": config.channels_shuffling,
            "adding_rgb_background": config.adding_rgb_background,
            "adding_rgb_foreground": config.adding_rgb_foreground,
            "add_red_textlines": config.add_red_textlines,
            "scaling_bluring": config.scaling_bluring,
            "scaling_binarization": config.scaling_binarization,
            "scaling_flip": config.scaling_flip,
            "blur_k": tuple(config.blur_k),
            "scales": tuple(config.scales),
            "brightness": tuple(config.brightness),
            "degrade_scales": tuple(config.degrade_scales),
            "flip_index": tuple(config.flip_index),
            "shuffle_indexes": tuple(config.shuffle_indexes),
            "thetha": tuple(config.thetha),
            "number_of_backgrounds_per_image": config.number_of_backgrounds_per_image,
            "seed": config.seed,
            **overrides,
        }
        return cls(**data)

    # ------------------------------------------------------------------ paths

    @property
    def images_output(self) -> Path:
        """Where generated images go (``<dir_output>/train/images``, as ``train.py`` expected)."""
        return Path(self.dir_flow_train_imgs or Path(self.dir_output) / "train" / "images")

    @property
    def labels_output(self) -> Path:
        """Where generated labels go (``<dir_output>/train/labels``)."""
        return Path(self.dir_flow_train_labels or Path(self.dir_output) / "train" / "labels")

    def background_files(self, rng: random.Random) -> list[str]:
        """Background image names to choose from -- given, or listed from ``dir_rgb_backgrounds``."""
        if self.list_all_possible_background_images:
            return list(self.list_all_possible_background_images)
        if not self.dir_rgb_backgrounds:
            raise ConfigError("no background images configured")
        files = [path.name for path in list_images(self.dir_rgb_backgrounds)]
        if not files:
            raise ConfigError(f"no background images found in {self.dir_rgb_backgrounds}")
        rng.shuffle(files)
        return files

    def foreground_files(self, rng: random.Random) -> list[str]:
        """The ``.npy`` foreground names, listed the way ``provide_patches`` did (any extension)."""
        if self.list_all_possible_foreground_rgbs:
            return list(self.list_all_possible_foreground_rgbs)
        if not self.dir_rgb_foregrounds:
            raise ConfigError("no foreground RGBs configured")
        files = sorted(path.name for path in Path(self.dir_rgb_foregrounds).glob("*"))
        if not files:
            raise ConfigError(f"no foreground RGBs found in {self.dir_rgb_foregrounds}")
        rng.shuffle(files)
        return files


@dataclass(frozen=True, slots=True)
class Variants:
    """One generated sample: what it is called, its image, its label, and where it came from.

    ``name`` follows the ``provide_patches`` scheme (``img_0``, ``img_1``, ... -- a counter, not
    the source file name) because the loader and every existing run directory key on it;
    ``source`` carries the origin so a generated file stays traceable.
    """

    name: str
    image: np.ndarray
    label: np.ndarray
    source: str = ""


# ------------------------------------------------------------------ padding


def pad_to(
    image: np.ndarray, label: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centre ``image``/``label`` on a canvas of at least ``height x width``.

    The legacy ``do_padding`` built that canvas as ``float`` (``np.ones(...) * 255`` for the
    image, zeros for the label), which survives ``cv2.imwrite`` only because the cast back to
    uint8 is exact. Both arrays come back as uint8.
    """
    image = np.asarray(image)
    label = np.asarray(label)
    canvas_h, canvas_w = max(height, image.shape[0]), max(width, image.shape[1])
    top, left = (canvas_h - image.shape[0]) // 2, (canvas_w - image.shape[1]) // 2

    canvas_image = np.full((canvas_h, canvas_w) + image.shape[2:], 255, dtype=np.uint8)
    canvas_label = np.zeros((canvas_h, canvas_w) + label.shape[2:], dtype=np.uint8)
    canvas_image[top : top + image.shape[0], left : left + image.shape[1]] = image
    canvas_label[top : top + label.shape[0], left : left + label.shape[1]] = label
    return canvas_image, canvas_label


def pad_with_border(
    image: np.ndarray, *, border: int = BORDER_WIDTH, fill: int = 255
) -> np.ndarray:
    """Add a ``border``-wide frame of ``fill`` around ``image``.

    One function replaces three -- ``do_padding_white`` (``fill=255``), ``do_padding_black``
    (``fill=0``) and ``do_padding_label`` (``fill=0`` on a label) -- which were the same eight
    lines with a different constant and returned ``float``/``int16`` where everything else
    wants uint8.
    """
    array = np.asarray(image)
    if border < 0:
        raise ValueError(f"border must be >= 0, got {border}")
    shape = (array.shape[0] + 2 * border, array.shape[1] + 2 * border) + (
        (array.shape[2],) if array.ndim == 3 else ()
    )
    padded = np.full(shape, fill, dtype=np.uint8)
    padded[border : border + array.shape[0], border : border + array.shape[1]] = array
    return padded


# ------------------------------------------------------------------ patches


def patch_slices(
    height: int, width: int, patch_height: int, patch_width: int
) -> list[tuple[int, int, int, int]]:
    """``(y0, y1, x0, x1)`` for every patch, on the legacy ceil-and-clamp grid.

    ``get_patches`` rounded the grid *up* and then clamped the last index back
    (``index_x_u = img_w; index_x_d = img_w - width``), so the final row/column of patches
    overlaps its neighbour instead of ending in a thin sliver. The overlap is kept: removing it
    would move every patch in an existing run directory.
    """
    if patch_height < 1 or patch_width < 1:
        raise ValueError(f"patch size must be positive, got {patch_height}x{patch_width}")
    if height < patch_height or width < patch_width:
        raise ValueError(
            f"{height}x{width} is smaller than the {patch_height}x{patch_width} patch; pad it "
            f"first (write_patches does)"
        )
    rows, cols = -(-height // patch_height), -(-width // patch_width)  # ceil division
    result: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * patch_height, (row + 1) * patch_height
            x0, x1 = col * patch_width, (col + 1) * patch_width
            if y1 > height:
                y1, y0 = height, height - patch_height
            if x1 > width:
                x1, x0 = width, width - patch_width
            result.append((y0, y1, x0, x1))
    return result


def patches(
    image: np.ndarray, label: np.ndarray, patch_height: int, patch_width: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every ``(image_patch, label_patch)`` pair, padding first when the image is too small."""
    image, label = np.asarray(image), np.asarray(label)
    if image.shape[0] < patch_height or image.shape[1] < patch_width:
        image, label = pad_to(image, label, patch_height, patch_width)
    return [
        (image[y0:y1, x0:x1], label[y0:y1, x0:x1])
        for y0, y1, x0, x1 in patch_slices(
            image.shape[0], image.shape[1], patch_height, patch_width
        )
    ]


def write_patches(
    image: np.ndarray,
    label: np.ndarray,
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    patch_height: int,
    patch_width: int,
    start_index: int = 0,
    scaler: float = 1.0,
) -> int:
    """Write every patch as ``img_{n}.png`` and return the next free index.

    Merges ``get_patches`` and ``get_patches_num_scale_new``: with ``scaler != 1`` the pair is
    rescaled first while the patches keep their configured size -- exactly what
    ``get_patches_num_scale_new`` did (its ``height_scale = int(height * 1)`` line makes the
    patch size independent of ``scaler``; the older ``get_patches_num_scale``, which scaled the
    patch itself, is the variant ``provide_patches`` never called).
    """
    image, label = np.asarray(image), np.asarray(label)
    if scaler != 1.0:
        if not 0.0 < scaler <= 10.0:
            raise ValueError(f"scaler must be in (0, 10], got {scaler}")
        image = resize(image, int(image.shape[0] * scaler), int(image.shape[1] * scaler))
        label = resize_mask(label, int(label.shape[0] * scaler), int(label.shape[1] * scaler))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape}")
    if Path(images_dir).resolve() == Path(labels_dir).resolve():
        # Every patch writes ``img_{n}.png`` into both folders, so one folder means the label
        # lands on the image: the same collision ``dataset.py``'s writer had.
        raise ValueError(
            f"images_dir and labels_dir must be different folders, both got {Path(images_dir)}"
        )

    index = start_index
    for image_patch, label_patch in patches(image, label, patch_height, patch_width):
        name = f"img_{index}.png"
        imwrite(Path(images_dir) / name, image_patch)
        imwrite(Path(labels_dir) / name, label_patch)
        index += 1
    return index


# ------------------------------------------------------------------ compositing


def with_red_elements(image: np.ndarray, binarised: np.ndarray) -> np.ndarray:
    """``return_image_with_red_elements``: paint the binarised text red over ``image``.

    The pixels written are ``(B, G, R) = (0, 0, 255)`` -- red in BGR order, so the name was
    right and the *axis* was not: it tests ``binarised[:, :, 0]`` (channel 0 of a BGR image) to
    decide what counts as text, the same convention its two siblings use. Kept for parity.
    """
    image, binarised = np.asarray(image), np.asarray(binarised)
    out = np.array(image, copy=True)
    text = binarised[:, :, 0] == 0
    out[..., 0] = np.where(text, 0, out[..., 0])
    out[..., 1] = np.where(text, 0, out[..., 1])
    out[..., 2] = np.where(text, 255, out[..., 2])
    return out


def _overlay(binarised: np.ndarray, background: np.ndarray) -> np.ndarray:
    """``return_binary_image_with_given_rgb_background``: the background shows through text pixels."""
    binarised, background = np.asarray(binarised), np.asarray(background)
    out = np.array(binarised, copy=True)
    for channel in range(min(binarised.shape[2], background.shape[2])):
        # Per-channel mask (``binarised[:, :, channel] != 0``), exactly as the legacy version did
        # -- its siblings use channel 0 for all three, which is the more defensible reading, but
        # changing all of them at once would change three outputs in one step.
        mask = binarised[:, :, channel] != 0
        out[:, :, channel] = np.where(mask, background[:, :, channel], out[:, :, channel])
    return out


def _overlay_with_foreground(
    binarised: np.ndarray, background: np.ndarray, foreground: np.ndarray
) -> np.ndarray:
    """``..._and_given_foreground_rgb``, with the foreground applied per pixel.

    The legacy version wrote ``rgb_foreground[0]`` -- *one row* of the ``(H, W, 3)`` array --
    into a boolean-masked channel, which only works when the mask selects exactly one pixel and
    raised ``ValueError`` for a normal page. The per-pixel colour is used now, and the
    ``H x W`` shape is checked instead of assumed.
    """
    binarised, background = np.asarray(binarised), np.asarray(background)
    foreground = np.asarray(foreground)
    expected = (binarised.shape[0], binarised.shape[1])
    if foreground.shape[:2] != expected:
        raise ConfigError(
            f"foreground has shape {foreground.shape}, expected ({expected[0]}, "
            f"{expected[1]}, 3) to match the binarised page"
        )
    out = _overlay(binarised, background)
    background_mask = binarised[:, :, 0] == 0
    for channel in range(min(3, foreground.shape[2])):
        out[:, :, channel] = np.where(
            background_mask, foreground[:, :, channel], out[:, :, channel]
        )
    return out


def composited_variants(
    image: np.ndarray,
    label: np.ndarray,
    binarised: np.ndarray,
    config: PatchConfig,
    *,
    rng: random.Random | None = None,
) -> list[Variants]:
    """The three compositing variants: background, background+foreground, red text lines.

    All three need ``dir_img_bin`` (the pre-binarised page) and the legacy code read it
    unconditionally, so ``binarization=false`` with ``adding_rgb_background=true`` crashed on a
    ``None`` image; without that folder the page is binarised here instead.
    """
    rng = rng if rng is not None else random.Random(config.seed)
    image, label, binarised = np.asarray(image), np.asarray(label), np.asarray(binarised)
    out: list[Variants] = []

    wants_backgrounds = config.adding_rgb_background or config.adding_rgb_foreground
    backgrounds = config.background_files(rng) if wants_backgrounds else []
    foregrounds = config.foreground_files(rng) if config.adding_rgb_foreground else []
    count = min(len(backgrounds), config.number_of_backgrounds_per_image) if backgrounds else 0

    for index in range(count):
        background = read_image(_source_file(config.dir_rgb_backgrounds, backgrounds[index]))
        if config.adding_rgb_background:
            out.append(
                Variants(
                    f"bg_{index}",
                    _overlay(binarised, resize(background, binarised.shape[0], binarised.shape[1])),
                    label,
                    backgrounds[index],
                )
            )
        if config.adding_rgb_foreground:
            foreground = np.load(_source_file(config.dir_rgb_foregrounds, foregrounds[index]))
            out.append(
                Variants(
                    f"fg_{index}",
                    _overlay_with_foreground(
                        binarised,
                        resize(background, binarised.shape[0], binarised.shape[1]),
                        foreground,
                    ),
                    label,
                    f"{backgrounds[index]}+{foregrounds[index]}",
                )
            )
    if config.add_red_textlines:
        out.append(Variants("red", with_red_elements(image, binarised), label, "add_red_textlines"))
    return out


def _source_file(directory: str | None, name: str) -> Path:
    """``<directory>/<name>`` with the ``None`` case already rejected by the validators."""
    if not directory:
        raise ConfigError(f"{name} has no directory configured")
    return Path(directory) / name


# ------------------------------------------------------------------ generation


def variants(
    image: np.ndarray,
    label: np.ndarray,
    config: PatchConfig,
    *,
    binarised: np.ndarray | None = None,
    rng: random.Random | None = None,
) -> list[Variants]:
    """Everything ``provide_patches`` produced for one source pair, as an in-memory list.

    With ``augmentation=False`` this is the base pair alone. With it on: one entry per enabled
    flag and parameter -- flips per ``flip_index``, blurs per ``blur_k``, brightness per
    ``brightness``, degradations per ``degrade_scales``, rotations per ``thetha``, channel
    shuffles per ``shuffle_indexes``, scaled copies per ``scales``, all six shifts, plus the four
    ``scaling_x`` combinations and the compositing variants.

    Image-only variants (blur, brightness, degrade, binarize, shuffle) keep the *original*
    label -- correct, since blurring a photo does not change its label -- while the geometric
    ones (flip, rotation, shift, scaling) transform the label alongside it.
    """
    image, label = np.asarray(image), np.asarray(label)
    out = [Variants("base", image, label, "base")]
    if not config.augmentation:
        return out

    for code in config.flip_index if config.flip_aug else ():
        out.append(
            Variants(f"flip_{code}", cv2.flip(image, code), cv2.flip(label, code), f"flip {code}")
        )
    for kind in config.blur_k if config.blur_aug else ():
        out.append(Variants(f"blur_{kind.value}", blur(image, kind), label, f"blur {kind.value}"))
    for factor in config.brightness if config.brightening else ():
        out.append(Variants(f"bright_{factor}", brighten(image, factor), label, f"bright {factor}"))
    if config.binarization:
        out.append(Variants("bin", otsu_binarize(image), label, "otsu"))
    for scale in config.degrade_scales if config.degrading else ():
        out.append(
            Variants(f"degrade_{scale}", downscale_upscale(image, scale), label, f"degrade {scale}")
        )
    if config.rotation:
        out.append(Variants("rot90", rotate90(image), rotate90(label), "rotate 90"))
    for angle in config.thetha if config.rotation_not_90 else ():
        rotated_image, rotated_label = rotate_not_90(image, label, angle_deg=angle)
        out.append(Variants(f"rot_{angle}", rotated_image, rotated_label, f"rotate {angle}"))
    for order in config.shuffle_indexes if config.channels_shuffling else ():
        out.append(
            Variants(f"shuffle_{''.join(map(str, order))}", shuffle_channels(image, order), label)
        )
    for scale in config.scales if config.scaling else ():
        scaled_image, scaled_label = _scaled_pair(image, label, scale)
        out.append(Variants(f"scale_{scale}", scaled_image, scaled_label, f"scale {scale}"))
    for direction in ShiftDirection if config.shifting else ():
        shifted_image, shifted_label = shift_image_and_label(image, label, direction)
        out.append(Variants(f"shift_{direction.value}", shifted_image, shifted_label))
    if config.padding_white:
        out.append(Variants("pad_white", pad_with_border(image), pad_with_border(label, fill=0)))
    if config.padding_black:
        out.append(
            Variants("pad_black", pad_with_border(image, fill=0), pad_with_border(label, fill=0))
        )

    # The four `scaling_x` combinations are separate outputs from plain `scaling` above: the
    # legacy code emitted e.g. scale_0.6_blur_gauss in addition to scale_0.6, never instead.
    if config.scaling_bluring:
        for scale in config.scales:
            for kind in config.blur_k:
                scaled_image, _ = _scaled_pair(image, label, scale)
                out.append(
                    Variants(
                        f"scale_{scale}_blur_{kind.value}",
                        blur(scaled_image, kind),
                        label,
                        f"scale {scale} + blur {kind.value}",
                    )
                )
    if config.scaling_brightness:
        for scale in config.scales:
            for factor in config.brightness:
                scaled_image, _ = _scaled_pair(image, label, scale)
                out.append(
                    Variants(
                        f"scale_{scale}_bright_{factor}",
                        brighten(scaled_image, factor),
                        label,
                        f"scale {scale} + bright {factor}",
                    )
                )
    if config.scaling_binarization:
        for scale in config.scales:
            scaled_image, _ = _scaled_pair(image, label, scale)
            out.append(
                Variants(
                    f"scale_{scale}_bin",
                    otsu_binarize(scaled_image),
                    label,
                    f"scale {scale} + otsu",
                )
            )
    if config.scaling_flip:
        for scale in config.scales:
            for code in config.flip_index:
                scaled_image, scaled_label = _scaled_pair(image, label, scale)
                out.append(
                    Variants(
                        f"scale_{scale}_flip_{code}",
                        cv2.flip(scaled_image, code),
                        cv2.flip(scaled_label, code),
                        f"scale {scale} + flip {code}",
                    )
                )

    if config.adding_rgb_background or config.adding_rgb_foreground or config.add_red_textlines:
        out.extend(
            composited_variants(
                image,
                label,
                binarised if binarised is not None else otsu_binarize(image),
                config,
                rng=rng,
            )
        )
    return out


def output_name(stem: str, variant_name: str) -> str:
    """The file name ``dataset.py``'s writer used: ``<stem>.png`` and ``<stem>_<variant>.png``.

    ``utils.provide_patches`` named its outputs ``img_{n}.png`` instead -- a counter with no
    link back to the source page, which is why ``dataset.py``'s scheme wins in non-patch mode
    (see :func:`generate`) and the counter is kept for patch mode, where a page legitimately
    produces many files.
    """
    return f"{stem}.png" if variant_name == "base" else f"{stem}_{variant_name}.png"


def _scaled_pair(
    image: np.ndarray, label: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both members of a pair by ``scale`` (``scale_image_for_no_patch``)."""
    if not 0.0 < scale <= 10.0:
        raise ValueError(f"scale must be in (0, 10], got {scale}")
    return (
        resize(image, int(image.shape[0] * scale), int(image.shape[1] * scale)),
        resize_mask(label, int(label.shape[0] * scale), int(label.shape[1] * scale)),
    )


def generate(config: PatchConfig, *, dry_run: bool = False) -> int:
    """Generate the training set ``config`` describes; return the number of files written.

    ``patches=True`` writes patch-sized files (``img_0.png``, ...) for every variant;
    ``patches=False`` writes each variant resized to ``input_height x input_width`` -- the two
    modes ``provide_patches`` chose between with a boolean buried in argument 31.

    ``dry_run=True`` validates and counts without writing, which is how ``sbb train --dry-run``
    answers "how much data would this produce" before touching the disk.
    """
    pairs = _resolve_pairs(config)
    if not dry_run:
        config.images_output.mkdir(parents=True, exist_ok=True)
        config.labels_output.mkdir(parents=True, exist_ok=True)

    written = index = 0
    for image_path, label_path in pairs:
        image, label = read_image(image_path), read_mask(label_path)
        binarised = (
            read_image(_source_file(config.dir_img_bin, f"{image_path.stem}.png"))
            if config.dir_img_bin
            else None
        )
        for variant in variants(image, label, config, binarised=binarised):
            if config.patches:
                before = index
                index = write_patches(
                    variant.image,
                    variant.label,
                    images_dir=config.images_output,
                    labels_dir=config.labels_output,
                    patch_height=config.input_height,
                    patch_width=config.input_width,
                    start_index=index,
                )
                written += 2 * (index - before)  # one image and one label file per patch
            else:
                sized_image = resize(variant.image, config.input_height, config.input_width)
                sized_label = resize_mask(variant.label, config.input_height, config.input_width)
                if dry_run:
                    assert sized_image.shape == (config.input_height, config.input_width, 3)
                    assert sized_label.shape == (config.input_height, config.input_width)
                else:
                    name = output_name(image_path.stem, variant.name)
                    imwrite(config.images_output / name, sized_image)
                    imwrite(config.labels_output / name, sized_label)
                written += 2
                index += 1
    logger.info("generated %d files under %s", written, config.images_output.parent)
    return written


def preprocess_and_augment_data(
    input_dir: str | Path,
    output_dir: str | Path,
    config: SBBConfig | PatchConfig,
    task: str | TaskType | None = None,
) -> int:
    """``dataset.preprocess_and_augment_data`` -- same ``train``/``eval`` layout, one call.

    The legacy version wrote ``train/{images,labels}`` and ``eval/{images,labels}``, resized
    photos with the nearest-neighbour ``resize_image``, duplicated the label for image-only
    augmentations, and printed its progress ("Found N images", "Warning: Could not read ...")
    instead of logging it. The output layout is unchanged; the naming follows
    :func:`variants` (``{name}.png`` per variant) rather than ``img_{n}.png``, which is what
    ``dataset.py``'s copy did and what the loader downstream expects.
    """
    sbb_config = config if isinstance(config, SBBConfig) else None
    task_name = (
        TaskType(task)
        if task is not None
        else (config.task if sbb_config is None else sbb_config.task)
    )
    root = Path(input_dir)
    total = 0
    for split in ("train", "eval"):
        if not (root / split / "images").is_dir() or not (root / split / "labels").is_dir():
            logger.warning("skipping %s: %s/%s missing", split, root, split)
            continue
        total += generate(_single_split_config(config, root, output_dir, split, task_name))
    return total


def _single_split_config(
    config: SBBConfig | PatchConfig,
    input_root: Path,
    output_dir: str | Path,
    split: str,
    task: TaskType,
) -> PatchConfig:
    """A ``PatchConfig`` reading ``<input_root>/<split>/{images,labels}`` and writing the same
    layout under ``output_dir``.

    ``dir_img``/``dir_seg`` are set here rather than inherited: ``from_sbb`` maps them onto
    ``dir_train``/``dir_eval`` (the two-folder convention ``provide_patches`` took), which for
    the eval split would read the *training* images again and, with an empty ``dir_eval``, fail
    outright. The split's own folders are what ``preprocess_and_augment_data`` just verified.
    """
    base = config if isinstance(config, PatchConfig) else PatchConfig.from_sbb(config, task=task)
    source = input_root / split
    return base.model_copy(
        update={
            "dir_img": str(source / "images"),
            "dir_seg": str(source / "labels"),
            "dir_flow_train_imgs": str(Path(output_dir) / split / "images"),
            "dir_flow_train_labels": str(Path(output_dir) / split / "labels"),
            "task": task,
            "patches": False,
        }
    )


def _resolve_pairs(config: PatchConfig) -> list[tuple[Path, Path]]:
    """``(image, label)`` pairs from the explicit lists, or by listing ``dir_img``/``dir_seg``."""
    if config.imgs_list:
        images = [Path(name) for name in config.imgs_list]
        if config.segs_list:
            labels = [Path(name) for name in config.segs_list]
            if len(labels) != len(images):
                raise ConfigError(
                    f"{len(images)} images but {len(labels)} labels; the legacy zip() silently "
                    f"trained on the shorter of the two"
                )
            return list(zip(images, labels, strict=True))
        if not config.dir_seg:
            raise ConfigError("imgs_list without segs_list needs dir_seg")
        return [
            (path, label_path_for(path, Path(config.dir_seg), task=config.task)) for path in images
        ]
    if not config.dir_img or not config.dir_seg:
        raise ConfigError("generate() needs imgs_list/segs_list or dir_img/dir_seg")
    return list_image_pairs(config.dir_img, config.dir_seg, task=config.task)


def dataset_for(
    config: PatchConfig,
    *,
    n_classes: int = 1,
    split: str = "train",
    sbb_config: SBBConfig | None = None,
) -> SegmentationDataset:
    """A :class:`~sbb_pytorch.data.dataset.SegmentationDataset` over what :func:`generate` wrote.

    ``PatchConfig`` carries no ``n_classes`` of its own (the class count belongs to the model,
    not to the patch writer), so it is passed here; ``split`` picks the generated split.

    Args:
        config: The patch configuration whose output folder is read.
        n_classes: Class count, used when ``sbb_config`` is not given.
        split: Which generated split to read, e.g. ``"train"`` or ``"eval"``.
        sbb_config: The run's own config, when the caller has one. Given, the dataset takes its
            class mapping, sizes and pipeline from it -- ``PatchConfig`` has no ``page2label`` of
            its own, so without it a 0/255 label folder cannot be remapped.
    """
    # The folder is the one holding ``images`` -- parenting ``<dir_output>/<split>`` instead of
    # ``<dir_output>/<split>/images`` gave ``<dir_output>`` and found no images at all.
    root = Path(config.dir_flow_train_imgs or Path(config.dir_output) / split / "images").parent
    if sbb_config is None:
        return SegmentationDataset(str(root), n_classes=n_classes)
    return SegmentationDataset(str(root), config=sbb_config)
