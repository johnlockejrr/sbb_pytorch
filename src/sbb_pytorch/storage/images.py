"""Image I/O and the pixel-level helpers that used to be copy-pasted across the repo.

``resize_image`` existed three times, and eight more helpers (``otsu_copy``,
``do_brightening``, ``do_degrading``, ``shift_image_and_label``, ``return_shuffled_channels``,
``rotation_not_90_func``, ``scale_image_for_no_patch``, ``bluring``) existed twice with
divergent behaviour. The two copies disagreed often enough to matter -- ``dataset.py`` fixed a
negative-slice bug in ``scale_image_for_no_patch`` and nobody deleted the broken original --
so this module is now the only place any of them live.

Channel convention
------------------
The canonical in-memory format for a **photo** is ``(H, W, 3) uint8`` in **RGB** order.
``cv2.imread`` yields BGR, so :func:`read_image` flips explicitly; the pre-refactor code
trained on RGB (``dataset.py`` loaded through PIL) but inferred with BGR
(``inference_pytorch.py`` called ``cv2.imread`` and then applied ImageNet's RGB statistics),
which silently mismatched the two paths. A **mask** is ``(H, W) uint8`` holding class ids;
:func:`read_mask` reads the first channel of a label PNG without any colour flip, because for
SBB label images channel 0 *is* the class id and channels 1-2 are a colour mapping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from sbb_pytorch.exceptions import ImageReadError

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SUFFIXES",
    "BlurKind",
    "ShiftDirection",
    "add_noise",
    "blur",
    "brighten",
    "downscale_upscale",
    "imread_bgr",
    "imread_rgb",
    "imwrite",
    "list_images",
    "normalize",
    "one_hot_mask",
    "otsu_binarize",
    "read_image",
    "read_mask",
    "remap_mask",
    "resize",
    "resize_mask",
    "rotate90",
    "rotate_not_90",
    "rotated_rect_max_area",
    "shift_image_and_label",
    "shuffle_channels",
    "to_uint8",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Extensions accepted for input imagery. Matching is case-insensitive, which the old
#: ``name.split('.')`` based code got wrong (it treated ``page.001.png`` as extension ``001``).
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


class BlurKind(str, Enum):
    """Blur variants. Replaces the free-form ``blur_k`` strings, where a shipped config
    spelled the value ``"guass"`` and hit an unbound-local crash."""

    GAUSS = "gauss"
    MEDIAN = "median"
    BLUR = "blur"


class ShiftDirection(str, Enum):
    """The six shift variants the legacy augmentation exposed as six ``elif`` branches."""

    XPOS = "xpos"
    XMIN = "xmin"
    YPOS = "ypos"
    YMIN = "ymin"
    XYPOS = "xypos"
    XYMIN = "xymin"


# --------------------------------------------------------------------------- I/O


def imread_bgr(path: str | Path) -> np.ndarray:
    """Read an image as ``(H, W, 3) uint8`` in **BGR** order (native OpenCV order).

    Raises:
        ImageReadError: if ``cv2.imread`` returns ``None`` -- an unreadable file, a missing
            file, or a non-ASCII path on a platform that cannot resolve one. The legacy code
            passed ``None`` straight into ``cv2.resize`` (``TypeError`` far from the cause) or
            swallowed it in a bare ``except:`` and trained on an all-white image instead.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(str(path))
    return image


def imread_rgb(path: str | Path) -> np.ndarray:
    """Read an image as ``(H, W, 3) uint8`` in **RGB** order. See module docstring."""
    return cv2.cvtColor(imread_bgr(path), cv2.COLOR_BGR2RGB)


def read_image(path: str | Path) -> np.ndarray:
    """Alias for :func:`imread_rgb` -- the canonical way to load a photo."""
    return imread_rgb(path)


def read_mask(path: str | Path) -> np.ndarray:
    """Read an SBB label image as ``(H, W) uint8`` class ids.

    SBB label PNGs store the class id in channel 0 and a colour mapping in channels 1-2, so
    this takes channel 0 with no colour flip. Every legacy reader did the same
    (``utils.get_one_hot``: ``seg = seg[:, :, 0]``; ``inference_pytorch.color_images``: same).

    Raises:
        ImageReadError: if the file cannot be decoded.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ImageReadError(str(path))
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    return to_uint8(raw)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as ``uint8`` without the rounding artefacts of ``astype``.

    ``astype(np.uint8)`` truncates and *wraps*: negative noise samples become ~255 and values
    above 255 wrap around, which is exactly how ``do_degrading`` corrupted images. Values in
    ``[0, 1]`` are scaled by 255 (the way ``cv2.imwrite`` silently does for float input);
    anything else is clipped to ``[0, 255]`` then rounded.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if float(finite.max()) <= 1.0 and float(finite.min()) >= 0.0:
            return np.clip(finite * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.clip(finite, 0, 255).round().astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def imwrite(
    path: str | Path, image: np.ndarray, *, channel_order: Literal["rgb", "bgr"] = "rgb"
) -> Path:
    """Write ``image`` to ``path``, creating parent directories.

    A 3-channel ``image`` is taken in the module's canonical **RGB** order and converted to the
    BGR order OpenCV writes, so ``imwrite(path, read_image(path))`` round-trips. The original
    code passed the array to ``cv2.imwrite`` unchanged, which saved every RGB-derived array with
    red and blue swapped -- the write-side twin of the read-side bug :func:`imread_rgb` fixed.
    Pass ``channel_order="bgr"`` for data already in OpenCV order (``imread_bgr`` output,
    ``cv2`` filter results, ``cv2.addWeighted`` composites).

    Raises:
        ImageReadError: if OpenCV reports a write failure (unsupported extension or an
            unwritable location).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if channel_order not in ("rgb", "bgr"):  # Literal narrows type checkers, not runtime callers
        raise ValueError(f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}")
    array = to_uint8(image)
    if array.ndim == 3 and array.shape[2] == 3 and channel_order == "rgb":
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    try:
        written = cv2.imwrite(str(path), array)
    except cv2.error as exc:
        # OpenCV >= 5 raises instead of returning False for an unknown extension, so the
        # boolean check alone is not enough.
        raise ImageReadError(str(path), reason=f"could not be written: {exc}") from exc
    if not written:
        raise ImageReadError(str(path), reason="could not be written")
    return path


def list_images(directory: str | Path) -> list[Path]:
    """Return every image file directly inside ``directory``, sorted, in one listing.

    Replaces ``find_format_of_given_filename_in_dir`` and the ``os.listdir`` calls that used to
    run *inside* per-file loops (an O(n^2) syscall pattern). Files only, case-insensitive
    extensions, and no crash when the directory holds nothing matching.

    Raises:
        FileNotFoundError: if ``directory`` does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------- geometry


def resize(
    image: np.ndarray, height: int, width: int, interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """Resize a **photo** to ``(height, width)``.

    ``cv2.resize`` takes ``(width, height)`` -- the one argument-order footgun that produced
    most of the transposed-size bugs in the old code, hence the explicit names. The default
    interpolation is ``INTER_AREA`` for downscaling and ``INTER_LINEAR``-equivalent quality when
    upscaling; ``INTER_NEAREST`` (the old unconditional choice, correct for *masks*) turns a
    photograph into visible blocks. Use :func:`resize_mask` for label images.
    """
    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a **mask** to ``(height, width)`` with nearest-neighbour interpolation.

    Only ``INTER_NEAREST`` is valid here: bilinear interpolation on a class-id mask invents
    values between classes, which the training loader then reads as out-of-range labels.
    """
    return cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)


def rotated_rect_max_area(width: int, height: int, angle_deg: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle ``(w, h)`` inside a ``width x height`` rect rotated by
    ``angle_deg``. Used to crop away the empty corners ``cv2.warpAffine`` leaves behind."""
    if width <= 0 or height <= 0:
        return 0.0, 0.0

    angle = math.radians(angle_deg)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # Half-constrained: two crop corners touch the longer side.
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # Fully constrained: the crop touches all four sides.
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (
            (width * cos_a - height * sin_a) / cos_2a,
            (height * cos_a - width * sin_a) / cos_2a,
        )

    return float(wr), float(hr)


# ------------------------------------------------------------------- transforms


def brighten(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness by ``factor`` (``>1`` brighter, ``<1`` darker), uint8 in and out.

    The legacy version took a *file path*, round-tripped through ``PIL`` + ``ImageEnhance`` and
    flipped BGR/RGB by hand -- forcing a second decode of an image the caller had usually
    already loaded. ``cv2.convertScaleAbs`` does the same job on the array.
    """
    return cv2.convertScaleAbs(image, alpha=float(factor), beta=0)


def blur(image: np.ndarray, kind: BlurKind | str, kernel: int = 5) -> np.ndarray:
    """Apply one of the three blur variants.

    Args:
        image: ``(H, W, C) uint8``.
        kind: A :class:`BlurKind`; a plain string is validated, so ``"guass"`` raises here
            instead of producing an unbound-local error deep in a generator.
        kernel: Odd kernel size. The legacy signature typed the value as a string
            (``blur_k: List[str]``) and then passed it where ``cv2`` required an int.

    Raises:
        ValueError: on an unknown kind or an even kernel size.
    """
    kind = BlurKind(kind)
    kernel = int(kernel)
    if kernel % 2 == 0:
        raise ValueError(f"blur kernel must be odd, got {kernel}")

    if kind is BlurKind.GAUSS:
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if kind is BlurKind.MEDIAN:
        return cv2.medianBlur(image, kernel)
    return cv2.blur(image, (kernel, kernel))


def add_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise with standard deviation ``sigma`` (in 0-255 units), clamped.

    Replaces the legacy ``np.random.normal(0, scale, ...).astype(np.uint8)``, which wrapped
    negative samples to ~255 and speckled the image with white dots.
    """
    noise = np.random.normal(0.0, float(sigma), size=image.shape)
    return to_uint8(np.asarray(image, dtype=np.float32) + noise)


def downscale_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale by ``scale`` and return to the original size -- a resolution-loss degradation.

    Works for ``scale > 1`` as well (a slight upscale); the legacy
    ``utils.scale_image_for_no_patch`` computed ``h_start`` from a *negative* difference in that
    case and raised ``ValueError: could not broadcast``. That variant also conflated this
    operation with noise; :func:`add_noise` is the noise half, exposed separately now.

    Raises:
        ValueError: if ``scale`` is not positive.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    height, width = image.shape[:2]
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    return (
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if (
            new_h,
            new_w,
        )
        == (height, width)
        else cv2.resize(
            cv2.resize(image, (new_w, new_h), cv2.INTER_AREA), (width, height), cv2.INTER_AREA
        )
    )


def rotate90(image: np.ndarray, times: int = 1) -> np.ndarray:
    """Rotate by ``90 * times`` degrees.

    ``np.rot90`` rather than the legacy ``rotation_90``, which transposed each channel -- a
    *reflection*, not a rotation, so the augmentation fed the model mirrored pages.
    """
    return np.ascontiguousarray(np.rot90(image, k=int(times), axes=(0, 1)))


def rotate_not_90(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    angle_deg: float = 0.0,
    crop_to_max_area: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rotate an image (and its mask) by a non-90-degree angle.

    The mask is always warped with ``INTER_NEAREST``. The legacy code used ``imutils.rotate``,
    which warps with bilinear interpolation -- so a mask of 0s and 1s came back containing 0.37
    and 0.8, i.e. class ids that do not exist.

    Args:
        image: ``(H, W, 3) uint8``.
        mask: ``(H, W) uint8`` class ids, or ``None`` for image-only rotation.
        angle_deg: Rotation in degrees, counter-clockwise.
        crop_to_max_area: Crop both outputs to the largest artefact-free rectangle.

    Returns:
        ``(rotated_image, rotated_mask)``; ``rotated_mask`` is ``None`` when ``mask`` is.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle_deg), 1.0)

    rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    rotated_mask = (
        None
        if mask is None
        else cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    )

    if crop_to_max_area and abs(angle_deg) > 1e-6:
        wr, hr = rotated_rect_max_area(width, height, angle_deg)
        y1 = height // 2 - int(hr / 2)
        y2 = y1 + int(hr)
        x1 = width // 2 - int(wr / 2)
        x2 = x1 + int(wr)
        if wr > 0 and hr > 0:
            rotated = rotated[y1:y2, x1:x2]
            if rotated_mask is not None:
                rotated_mask = rotated_mask[y1:y2, x1:x2]

    return rotated, rotated_mask


def shift_image_and_label(
    image: np.ndarray,
    mask: np.ndarray,
    direction: ShiftDirection | str,
    fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate an image and its mask together in one of six directions.

    Args:
        image: ``(H, W, 3) uint8``.
        mask: ``(H, W) uint8``.
        direction: A :class:`ShiftDirection`; the legacy code had six ``elif`` branches with no
            ``else``, so any other string returned through unbound names.
        fraction: Shift distance as a fraction of the axis (legacy used ``1/4``).

    Returns:
        ``(shifted_image, shifted_mask)``. Areas exposed by the shift are filled with black for
        the image and class 0 for the mask.
    """
    direction = ShiftDirection(direction)
    height, width = image.shape[:2]
    dx = {"xpos": 1, "xmin": -1, "xypos": 1, "xymin": -1, "ypos": 0, "ymin": 0}[direction.value]
    dy = {"ypos": 1, "ymin": -1, "xypos": 1, "xymin": -1, "xpos": 0, "xmin": 0}[direction.value]

    matrix = np.float32([[1, 0, dx * width * fraction], [0, 1, dy * height * fraction]])
    shifted = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
    shifted_mask = cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return shifted, shifted_mask


def shuffle_channels(image: np.ndarray, order: Sequence[int] = (1, 0, 2)) -> np.ndarray:
    """Reorder the three channels, e.g. ``order=(1, 0, 2)`` to swap red and green.

    Raises:
        ValueError: if ``order`` is not a permutation of ``0, 1, 2``.
    """
    order = tuple(int(i) for i in order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"channel order must be a permutation of (0, 1, 2), got {order}")
    return np.ascontiguousarray(image[..., list(order)])


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Binarise a photo with Otsu's method, returned as ``(H, W, 3) uint8``.

    Converts to grey first and thresholds **once**. The legacy ``otsu_copy`` ran
    ``cv2.threshold`` on all three channels and then wrote the *blue* channel's result into all
    three -- two thresholds computed and discarded, and not a binarisation of the image.
    """
    grey = cv2.cvtColor(to_uint8(image), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------- label utils


def remap_mask(mask: np.ndarray, mapping: Mapping[int, int], n_classes: int) -> np.ndarray:
    """Map stored label values to class ids and validate them against ``n_classes``.

    A label PNG saved with the 0/255 convention yields ``255`` where the class id is ``1``, and
    an anti-aliased PNG yields values like ``128`` that match no class at all. Both cases used
    to pass silently: ``get_one_hot`` produced an all-zero row (a pixel belonging to no class,
    counted as a false negative for every class) and ``IoU`` scored it accordingly.

    Args:
        mask: ``(H, W)`` array of stored label values.
        mapping: ``{stored_value: class_id}``, e.g. ``{0: 0, 255: 1}`` or the SBB
            ``{1: 1, 2: 2, ...}`` scheme from the page-to-label config.
        n_classes: Upper bound for validation.

    Returns:
        ``(H, W) uint8`` class ids.

    Raises:
        ValueError: if the mapping targets a class id outside ``[0, n_classes)``, or if ``mask``
            holds a stored value that ``mapping`` does not cover.
    """
    source_values = np.unique(mask)
    missing = [int(v) for v in source_values if int(v) not in mapping]
    if missing:
        raise ValueError(
            f"no class id configured for label values {missing} in {n_classes}-class setup"
        )

    remapped = np.full(mask.shape, fill_value=255, dtype=np.uint8)
    for stored, class_id in mapping.items():
        if not 0 <= int(class_id) < n_classes:
            raise ValueError(
                f"class id {class_id} for stored value {stored} is outside [0, {n_classes})"
            )
        remapped[mask == int(stored)] = int(class_id)
    return remapped


def one_hot_mask(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode a ``(H, W)`` class-id mask as ``(H, W, n_classes)``.

    A single ``np.eye`` gather rather than the legacy per-class ``(seg == j)`` loop, which ran
    ``n_classes`` full-array passes.
    """
    ids = np.asarray(mask, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= n_classes):
        raise ValueError(
            f"mask values out of range for n_classes={n_classes}: "
            f"[{int(ids.min())}, {int(ids.max())}]"
        )
    return np.eye(n_classes, dtype=np.uint8)[ids]


def normalize(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Apply ImageNet normalisation to an ``(H, W, 3) uint8`` RGB image, returning float32.

    ``(image/255 - mean) / std``. Kept here so inference and training normalise identically --
    see the channel-convention note in the module docstring for why that mattered.
    """
    source = np.asarray(image)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if source.dtype == np.uint8:
        scaled = source.astype(np.float32) / 255.0
    elif np.issubdtype(source.dtype, np.floating):
        finite = np.nan_to_num(source.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if finite.size == 0:
            return finite
        low, high = float(finite.min()), float(finite.max())
        if 0.0 <= low <= 1.0 and high <= 1.0:
            scaled = finite  # a float image already expressed in [0, 1]
        elif low >= 0.0 and high <= 255.0:
            scaled = finite / 255.0  # a float image still in 0-255
        else:
            # Outside any 0-255 interpretation, so it is already normalised -- dividing again
            # is the bug this branch exists to prevent.
            return finite
    else:
        scaled = to_uint8(source).astype(np.float32) / 255.0

    return (scaled - mean_arr) / std_arr
