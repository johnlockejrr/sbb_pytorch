"""Rasterising parsed regions onto page images.

The legacy module spread this concern over five functions —
``visualize_image_from_contours_layout``, ``visualize_image_from_contours``,
``visualize_model_output``, ``make_image_from_bb`` and the ~450 lines of inline
``fillPoly`` calls inside ``get_images_of_ground_truth`` — each of which kept its own
copy of the colour table and its own blend maths.

Everything here derives from one primitive, :func:`label_image`: an ``(H, W)`` image
whose pixel value *is* the class id.  The two output formats the tool offers are
derived views of it, which is why they can no longer disagree:

``2d``
    the label map repeated across three channels.  The legacy CLI documented this
    as "2D image array is 3d but only information of one channel would be enough",
    and the legacy code produced exactly that by calling
    ``cv2.fillPoly(img, pts, color=class_id)`` — a class id used as an RGB colour.
``3d``
    the label map passed through a palette (:func:`label_map_to_image`).

:data:`DEFAULT_LABEL_PALETTE` is the legacy ``labels_rgb_color`` table, verbatim.
Despite the legacy name it was handed to OpenCV and therefore behaved as **BGR**;
keeping the same order keeps the rendered colours identical to the old output.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from ..config.models import OutputType
from .geometry import BoundingBox, contour_area, mask_of
from .parser import Page, Region, RegionKind

__all__ = [
    "ARTIFICIAL_CLASS_COLOR",
    "BINARY_BACKGROUND_COLOR",
    "BINARY_FOREGROUND_COLOR",
    "DEFAULT_LABEL_PALETTE",
    "DEFAULT_OVERLAY_ALPHA",
    "PRINTSPACE_CLASS_COLOR",
    "LabelledContour",
    "OutputType",
    "Palette",
    "crop_to",
    "image_for_page",
    "label_color",
    "label_image",
    "label_map_to_image",
    "make_image_from_bb",
    "overlay",
    "paint_contours",
    "read_image",
    "render_page",
    "to_label_map",
    "to_uint8_bgr",
    "write_image",
]

#: Canonical label palette, verbatim from the legacy ``labels_rgb_color`` list.
#: Stored as **BGR** — the legacy tuple was named ``*_rgb_*`` but every consumer
#: passed it to OpenCV, which is BGR; reordering it here would shift every colour.
DEFAULT_LABEL_PALETTE: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, 0),
    (255, 0, 0),
    (255, 125, 0),
    (255, 0, 125),
    (125, 255, 125),
    (125, 125, 0),
    (0, 125, 255),
    (0, 125, 0),
    (125, 125, 125),
    (255, 0, 255),
    (125, 0, 125),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 125, 125),
    (0, 125, 125),
    (0, 255, 125),
    (255, 125, 255),
    (125, 255, 0),
)

#: Colour of the synthetic boundary class (legacy ``artificial_class_rgb_color``).
ARTIFICIAL_CLASS_COLOR: Final[tuple[int, int, int]] = (255, 255, 0)

#: Colour painted outside the print space (legacy ``printspace_class_rgb_color``).
PRINTSPACE_CLASS_COLOR: Final[tuple[int, int, int]] = (125, 125, 255)

#: Foreground/background of a binarisation prediction.  The legacy
#: ``visualize_model_output`` computed this as ``prediction * -1 + 1``, which only
#: inverts a {0, 1} prediction and silently corrupted anything else.
BINARY_FOREGROUND_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)
BINARY_BACKGROUND_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)

#: Blend weight of the photograph in every legacy visualiser.  The legacy calls were
#: ``cv2.addWeighted(img, 0.5, layout, 0.1, 0)`` — weights summing to 0.6, so the
#: composite came out at 60 % brightness.  :func:`overlay` normalises them to 1 while
#: keeping the 5:1 ratio the author asked for.
DEFAULT_OVERLAY_ALPHA: Final = 0.5 / 0.6

#: A colour table: either ``label -> colour`` or a sequence indexed by label.
Palette = Mapping[int, tuple[int, int, int]] | tuple[tuple[int, int, int], ...]

#: One ``(contour, class id)`` pair — the unit :func:`label_image` consumes.
LabelledContour = tuple[np.ndarray, int]


#: The ``2d``/``3d`` enum lives in the config schema now (``config.models.OutputType``, aliased
#: there as ``TypeOutput``); this module's copy was the second spelling of it.
def label_color(label: int, palette: Palette = DEFAULT_LABEL_PALETTE) -> tuple[int, int, int]:
    """Colour for *label*, wrapping around the palette.

    Two legacy defects: ``visualize_model_output`` kept a 16-entry dict and raised
    ``KeyError`` for any class id above 15 (and its ``'9'`` duplicated its ``'6'``),
    while the layout renderer indexed a 19-entry list and raised ``IndexError`` past
    18.  Wrapping renders both instead of aborting the page.  An explicit
    ``label -> colour`` mapping still raises, because silently inventing a colour for a
    configured label would hide a config typo.
    """
    if isinstance(palette, Mapping):
        try:
            return dict(palette)[int(label)]
        except KeyError as exc:
            raise KeyError(f"no palette entry for label {label}") from exc
    colors = tuple(palette)
    return colors[int(label) % len(colors)]


def _hw(shape: Sequence[int] | int) -> tuple[int, int]:
    """``(height, width)`` from a shape tuple of rank 1, 2 or 3 -- or a bare square edge.

    Accepting rank 1 and 3 as well as the plain ``(h, w)`` removes the ``shape[:2]`` /
    ``shape[0], shape[1]`` variants the legacy call sites each wrote their own way. A bare ``int``
    means ``square``, which is the same reading :func:`geometry.mask_of` gives it.
    """
    if isinstance(shape, int):
        return shape, shape
    if len(shape) == 1:
        return int(shape[0]), int(shape[0])
    return int(shape[0]), int(shape[1])


def label_image(
    entries: Iterable[LabelledContour],
    shape: Sequence[int] | int,
    *,
    background: int = 0,
) -> np.ndarray:
    """Rasterise ``(contour, class id)`` pairs into an ``(H, W)`` ``uint8`` label map.

    Later entries overwrite earlier ones, so callers paint the background first and the
    small classes last — the ordering the legacy code relied on when it drew the
    print-space background, then the regions, then the boundary class.

    Raises
    ------
    ValueError
        A class id — including *background* — does not fit the ``uint8`` label map.  The
        legacy code produced this silently by storing label ids in an image typed
        ``int8``/``uint8`` and writing ``cv2.fillPoly(..., color=class_id)``, so a class id
        above 255 (or the ``int8`` reading-order maps, above 127) wrapped around.
        ``background`` used to skip the check and reach ``np.full`` with an out-of-range
        value, which answered ``OverflowError`` instead of naming the mistyped id.
    """
    height, width = _hw(shape)
    labels = list(entries)
    if not 0 <= int(background) <= 255:
        raise ValueError(f"background id {int(background)} does not fit a uint8 label map")
    if any(not 0 <= int(label) <= 255 for _, label in labels):
        bad = next(int(label) for _, label in labels if not 0 <= int(label) <= 255)
        raise ValueError(f"class id {bad} does not fit a uint8 label map")
    canvas = np.full((height, width), int(background), dtype=np.uint8)
    for contour, label in labels:
        cv2.fillPoly(canvas, [_as_contour(contour, height, width)], int(label))
    return canvas


def _as_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Contour clipped into a ``height x width`` canvas, as ``(N, 1, 2)``."""
    array = np.asarray(contour).reshape(-1, 2).copy()
    if array.size == 0:
        return np.zeros((1, 1, 2), dtype=np.int32)
    array[:, 0] = np.clip(array[:, 0], 0, max(width - 1, 0))
    array[:, 1] = np.clip(array[:, 1], 0, max(height - 1, 0))
    return np.ascontiguousarray(array.reshape(-1, 1, 2))


def label_map_to_image(
    label_map: np.ndarray,
    palette: Palette = DEFAULT_LABEL_PALETTE,
) -> np.ndarray:
    """Colourise a class-index image, for the ``3d`` output format.

    Replaces the ``for unq_class in unique_classes: layout_only[..., ch][...] = ...``
    loop, which built a float array, downcast it through ``int32`` and handed the
    ``int32`` result to ``cv2.imwrite`` — producing a 32-bit PNG instead of the 8-bit
    one the training pipeline expected.
    """
    array = label_map if label_map.ndim == 2 else label_map[..., 0]
    if isinstance(palette, Mapping):
        image = np.zeros((*array.shape, 3), dtype=np.uint8)
        for label, color in palette.items():
            for channel in range(3):
                image[..., channel][array == label] = color[channel]
        return image
    colors = np.asarray(palette, dtype=np.uint8)
    return colors[np.clip(np.asarray(array, dtype=np.intp), 0, len(colors) - 1) % len(colors)]


def to_label_map(label_map: np.ndarray) -> np.ndarray:
    """The ``2d`` form: the label map repeated across three channels.

    The legacy code reached this by ``fillPoly``-ing a class id as an RGB colour, which
    costs three times the memory to store one number per pixel.  Writing a single
    channel is the correct representation; the three-channel copy is kept because the
    existing training data and every downstream consumer read a 3-channel PNG.

    The shape is upstream's own, not an artefact of the port: ``gt_gen_utils.py:289`` fills
    a ``np.zeros((y_len, x_len, 3))`` even on the ``2d`` branch, and the legacy CLI said so
    outright -- ``--type_output``'s help in ``generate_gt_for_training.py:45`` reads "2D
    image array is 3d but only information of one channel would be enough since all
    channels have the same values."  A ``(H, W)`` label would be the deviation.
    """
    array = label_map if label_map.ndim == 2 else label_map[..., 0]
    return np.repeat(array[..., None], 3, axis=2)


def paint_contours(
    contours: Sequence[np.ndarray],
    shape: Sequence[int],
    color: tuple[int, int, int],
    *,
    outline_color: tuple[int, int, int] | None = None,
    outline_thickness: int = 1,
) -> np.ndarray:
    """BGR image of *shape* with *contours* filled in *color*.

    Merges the legacy pattern of a ``fillPoly`` immediately followed by a
    ``drawContours`` of the same contours, which the layout visualiser applied to
    paragraphs only: the fill and the one-pixel outline are separate masks here, so the
    outline does not eat into the fill and the fill-only callers share the code path.
    The stroked ring *is* the outline — ``filled`` keeps only the pixels it does not cover,
    which is what makes ``outline_color`` visible at the default thickness of 1 (the first
    version subtracted the fill from the ring, and a 1-pixel stroke coincides exactly with
    the boundary ``mask_of`` has already filled, so nothing was left to colour).
    """
    height, width = _hw(shape)
    contour_list = list(contours)
    if not contour_list:
        return np.zeros((height, width, 3), dtype=np.uint8)

    filled = mask_of(contour_list, (height, width), fill_value=1).astype(bool)
    edge = np.zeros((height, width), dtype=bool)
    if outline_color is not None:
        stroked = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(
            stroked,
            [_as_contour(c, height, width) for c in contour_list],
            -1,
            1,
            thickness=outline_thickness,
        )
        edge = stroked.astype(bool)
        filled &= ~edge  # the two masks must not overlap: the ring wins, the fill keeps the rest

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        outline_value = outline_color[channel] if outline_color is not None else 0
        image[..., channel] = np.where(filled, color[channel], np.where(edge, outline_value, 0))
    return image


def overlay(
    base: np.ndarray,
    top: np.ndarray,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend *top* over *base* with normalised weights and matching sizes.

    Fixes two legacy defects: the ``0.5 / 0.1`` weight pair (summing to 0.6, hence the
    dimmed output — see :data:`DEFAULT_OVERLAY_ALPHA`), and the calls that passed an
    ``int32`` operand and so wrote a 32-bit PNG.  *top* is resized to *base* when the
    two disagree, which the legacy ``visualize_model_output`` did by hand via
    ``resize_image`` and the other visualisers omitted entirely — raising ``cv2.error``
    on any prediction that missed the requested output size.
    """
    base8 = to_uint8_bgr(base)
    top8 = to_uint8_bgr(top)
    if top8.shape[:2] != base8.shape[:2]:
        top8 = cv2.resize(top8, (base8.shape[1], base8.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(base8, float(alpha), top8, 1.0 - float(alpha), 0)


def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    """Normalise *image* to ``uint8`` 3-channel BGR.

    Handles the float-in-[0, 1] predictions the models emit, single-channel masks, and
    already-encoded images of another depth.  This is the second of the four divergent
    ``astype(uint8)``/``cvtColor`` pairs in the legacy code, kept as the one public
    version; ``sbb_pytorch.storage.images`` re-exports it for non-GT callers.
    """
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("empty image")
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 4:
        array = array[0]
    if not np.issubdtype(array.dtype, np.uint8):
        floating = np.issubdtype(array.dtype, np.floating)
        in_unit_range = floating and float(array.min()) >= 0.0 and float(array.max()) <= 1.0
        scale = 255.0 if in_unit_range else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return array


def read_image(path: str | Path) -> np.ndarray | None:
    """``cv2.imread`` with the ``None`` the legacy code never checked for.

    Every legacy visualiser fed the result straight into ``cv2.cvtColor``, so one
    unreadable file aborted the whole run with ``cv2.error`` instead of skipping a page.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Write *image* to *path* as an 8-bit PNG, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    written = cv2.imwrite(str(target), array if array.dtype == np.uint8 else to_uint8_bgr(array))
    if not written:
        raise OSError(f"cv2.imwrite failed for {target}")
    return target


def image_for_page(page: Page, directory: str | Path) -> np.ndarray | None:
    """The raster image of *page*, located in *directory* by filename stem."""
    from .parser import image_for  # Local import: keeps the parser/render pair acyclic.

    found = image_for(page, directory)
    return None if found is None else read_image(found)


def render_page(
    page: Page,
    *,
    size: tuple[int, int] | None = None,
    output_type: OutputType | str = OutputType.THREE_D,
    palette: Palette = DEFAULT_LABEL_PALETTE,
    labels: Mapping[RegionKind, int] | None = None,
) -> np.ndarray:
    """Render *page*'s regions, as ``3d`` RGB or ``2d`` class-id image.

    The single replacement for the inline rasterisation block of
    ``get_images_of_ground_truth``, which branched over ``config_params`` keys eleven
    times to reach the same ``fillPoly`` call.

    Parameters
    ----------
    page:
        Parsed page whose regions are drawn.
    size:
        ``(height, width)`` of the output; defaults to the page's declared size.
    output_type:
        ``3d`` colours the label map through *palette*; ``2d`` replicates the class id
        across the three channels (the legacy 2d format).
    palette:
        ``label -> colour`` mapping, or a sequence indexed by label id.
    labels:
        ``RegionKind -> class id``.  When omitted, ids are derived from each region's
        ``@ID`` digits, which suits visualisation; generation always passes the
        configured mapping.
    """
    target = size if size is not None else (page.height, page.width)
    pairs: list[tuple[Region, int]] = [
        (region, _label_of(region, labels)) for region in page.regions
    ]
    # Largest region first, so small classes drawn later are not buried by big ones —
    # the painter's order the legacy code got by accident through dict iteration.
    entries = sorted(
        ((region.contour, label) for region, label in pairs),
        key=lambda entry: contour_area(entry[0]),
        reverse=True,
    )
    label_map = label_image(entries, target, background=0)
    if OutputType.parse(output_type) is OutputType.TWO_D:
        return to_label_map(label_map)
    return label_map_to_image(label_map, palette)


def _label_of(region: Region, labels: Mapping[RegionKind, int] | None) -> int:
    """Class id of *region*: the configured one, else the digits of its ``@ID``.

    The legacy tool encoded the class in the filename and in ``@ID`` suffixes
    (``paragraph_2`` → class 2); with no configured mapping and no numeric suffix the
    region hashes to a stable id instead of defaulting to 0 and vanishing into the
    background.  "Stable" needs the hash to be the same in every process, so it is
    ``zlib.crc32`` of the identity rather than the builtin ``hash()``, which CPython
    randomises per run and made the same page render with different class ids on
    consecutive invocations.
    """
    if labels is not None and region.kind in labels:
        return int(labels[region.kind])
    digits = "".join(char for char in region.region_id if char.isdigit())
    if digits:
        return int(digits)
    identity = f"{region.kind.value}:{region.type_name}".encode()
    return zlib.crc32(identity) % len(DEFAULT_LABEL_PALETTE)


def make_image_from_bb(
    width: int,
    height: int,
    boxes: Sequence[Sequence[int]],
    *,
    value: int = 1,
    dtype: type[np.generic] = np.float64,
) -> np.ndarray:
    """Binary image holding *value* inside every ``[x, y, w, h]`` box in *boxes*.

    Port of ``make_image_from_bb``.  The legacy version sized the canvas from the
    caller's ``width_l``/``height_l`` and sliced each box unclamped, so a box extending
    past the canvas produced a silently empty slice and one starting beyond it produced
    a ``ValueError``-free no-op; the boxes are clamped here.
    """
    canvas = np.zeros((int(height), int(width)), dtype=dtype)
    for box in boxes:
        x, y, w, h = (int(v) for v in tuple(box)[:4])
        x, y = max(0, x), max(0, y)
        if x >= width or y >= height:
            continue
        canvas[y : min(y + max(h, 0), height), x : min(x + max(w, 0), width)] = value
    return canvas


def crop_to(image: np.ndarray, bbox: BoundingBox | None) -> np.ndarray:
    """Crop *image* to a ``(x, y, w, h)`` box, returning it unchanged when absent.

    The legacy print-space crop was written three times as
    ``img_poly[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]`` — with no guard for the empty
    or zero-height slice that a degenerate print space produces, which
    ``cv2.imwrite`` then rejected.  The box is clamped to the image the way
    :func:`make_image_from_bb` clamps its boxes, so a box that hangs over an edge crops
    the intersection instead of handing back a too-small slice, and a box that starts
    outside the image — where numpy answers a zero-length array — returns the image.
    """
    if bbox is None:
        return image
    x, y, w, h = (int(value) for value in tuple(bbox)[:4])
    height, width = image.shape[:2]
    x, y = max(0, x), max(0, y)
    if x >= width or y >= height:
        return image
    w, h = min(w, width - x), min(h, height - y)
    if w <= 0 or h <= 0:
        return image
    return image[y : y + h, x : x + w, ...]
