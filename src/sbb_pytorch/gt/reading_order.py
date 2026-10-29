"""Machine-based reading order: the region-pair training samples.

Ports the ``machine_based_reading_order`` command of
``generate_gt_for_training.py`` (and the ``find_new_features_of_contours`` /
``overlay_layout_on_image`` helpers it leaned on).

The task the samples describe is a pairwise ordering problem: given two text
regions of one page, predict which one comes first.  For a page with *n* usable
regions the tool emits ``n * (n - 1)`` samples — one per **ordered** pair.  Both
``(i, j)`` and ``(j, i)`` are kept deliberately: the modal image differs (channel 0
carries the label map of *i*, channel 2 that of *j*) and so does the label, so the
reverse pair is a genuine sample rather than a duplicate.

The modal image packs three views into three channels:

======  =======================================================
channel content
======  =======================================================
0       label map of region *i* (plus image regions and headers)
1       the page layout raster (``img_poly``)
2       label map of region *j* (plus image regions and headers)
======  =======================================================

Label alphabet — note it is *not* the layout use case's alphabet; the numbers are
hard-coded here exactly as the legacy code hardcoded them, so a page labelled for
reading order cannot be relabelled by a layout config:

======  =============================
0       background
1       the queried text region
2       image region
3       header, and the header-separator band beneath it
======  =============================
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from .geometry import (
    DEFAULT_MIN_AREA_FRACTION,
    ContourFeatures,
    FilteredContours,
    contour_features,
    filter_contours_by_area,
    mask_of,
    scale_contour,
)
from .parser import Page, Region, RegionKind
from .render import DEFAULT_LABEL_PALETTE, overlay, render_page, to_uint8_bgr, write_image

__all__ = [
    "BACKGROUND_LABEL",
    "DEFAULT_MAX_AREA_FRACTION",
    "HEADER_LABEL",
    "HEADER_SEPARATOR_BAND_WIDTH",
    "IMAGE_REGION_LABEL",
    "QUERY_LABEL",
    "ReadingOrderPair",
    "generate_pairs",
    "header_separator_mask",
    "ordered_regions",
    "ordered_text_regions",
    "region_label_map",
    "visualize_reading_order",
    "write_pair",
]

#: Width, in pixels, of the separator band drawn under a header.  The legacy code
#: hardcoded the ``+12`` slice; naming it makes the two use cases share one constant.
HEADER_SEPARATOR_BAND_WIDTH: Final = 12

#: Largest region size (page fraction) admitted as a text region.  The legacy tool
#: hardcoded ``max_area = 1`` — i.e. no upper bound — while still routing it through
#: the area filter; it is configurable here, defaulting to the same effective 1.0.
DEFAULT_MAX_AREA_FRACTION: Final = 1.0

BACKGROUND_LABEL: Final = 0
#: Label of whichever region the pair asks about first (channel 0 of the modal image).
QUERY_LABEL: Final = 1
#: Label written over image regions, hardcoded in the legacy tool rather than taken
#: from the page-to-label config (where ``imageregion`` is 4 in the shipped preset).
IMAGE_REGION_LABEL: Final = 2
#: Label written over headers and the separator band beneath them.
HEADER_LABEL: Final = 3


@dataclass(frozen=True, slots=True)
class ReadingOrderPair:
    """One pairwise training sample.

    Attributes
    ----------
    name:
        Output stem, ``<page-stem>_<index>`` — the counter runs across every pair of
        the page, shared by the normal and the ``_missed`` outputs, exactly as the
        legacy ``indexer`` did.
    class_type:
        ``1`` when the first region precedes the second in reading order, else ``0``.
    modal_image:
        ``(H, W, 3)`` ``uint8`` image — see the module docstring for the channels.
        The legacy array was ``int8``, which is the second bug fixed here: channel 1
        holds palette values up to 255 and the label values go to 3, so everything
        above 127 wrapped to a negative value on disk.
    missed:
        ``True`` when this region pair only exists because ``min_area`` was lowered
        below the value an earlier run used (the legacy ``_missed`` suffix).
    """

    name: str
    class_type: int
    modal_image: np.ndarray
    missed: bool = False

    @property
    def suffix(self) -> str:
        """``"_missed"`` or ``""`` — the filename part after the counter."""
        return "_missed" if self.missed else ""


def ordered_text_regions(
    page: Page,
    *,
    min_area: float = DEFAULT_MIN_AREA_FRACTION,
    max_area: float = DEFAULT_MAX_AREA_FRACTION,
    min_area_early: float | None = None,
) -> FilteredContours:
    """Text regions of *page* that are usable as reading-order samples.

    Replaces the ``id_paragraph + id_header`` / ``co_text_paragraph + co_text_header``
    concatenation and the ``filter_contours_area_of_image`` call that followed it.
    The reading order addresses paragraph *and* header regions together, so both are
    collected here via :meth:`Page.contours_for`.
    """
    regions = [region for region in page.regions if region.kind is RegionKind.TEXT]
    contours = [region.contour for region in regions]
    order_index = [
        page.region_index.get(region.region_id, position) for position, region in enumerate(regions)
    ]
    return filter_contours_by_area(
        contours,
        (page.height, page.width),
        min_area=min_area,
        max_area=max_area,
        min_area_early=min_area_early,
        order_index=order_index,
    )


def ordered_regions(page: Page, *, limit: int | None = None) -> list[Region]:
    """Every region of *page*, in reading order.

    Ordering is ``RegionRefIndexed/@index`` via :attr:`Page.region_index`, falling
    back to document order when the file carries no indices, then to a top-to-bottom
    / left-to-right sweep of the centroids.  The legacy code produced two parallel
    lists (``index_element_to_be_updated`` / ``pr_list`` / ``pos_list``) that its
    only consumer re-sorted with ``list.index(...)`` per element; the sort happens
    once here instead.
    """
    unindexed = np.iinfo(np.int64).max
    if page.region_index:
        ranked = sorted(page.regions, key=lambda r: page.region_index.get(r.region_id, unindexed))
    else:
        ranked = sorted(page.regions, key=_geometric_key)
    return ranked[:limit] if limit is not None else ranked


def _geometric_key(region: Region) -> tuple[int, int]:
    """Sort key placing regions top-to-bottom, then left-to-right."""
    box = region.contour.reshape(-1, 2)
    return (int(box[:, 1].min()), int(box[:, 0].min()))


def header_separator_mask(
    page: Page,
    shape: tuple[int, int],
    *,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """The band under every header, marked :data:`HEADER_LABEL`.

    The legacy tool wrote ``img_header_and_sep[y_max:y_max + 12, x_min:x_max] = 1``
    for each header contour — a fixed-height strip below the header's bottom edge,
    standing in for the separator rule that separates a header from its body text.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    scale_x, scale_y = _scales(shape, target_size)
    headers = [region for region in page.regions if _is_header(page, region)]
    for feature in contour_features([_scaled(r.contour, scale_x, scale_y) for r in headers], shape):
        x, y, w, h = feature.bbox
        band_top = min(y + h, max(shape[0] - 1, 0))
        band_bottom = min(band_top + HEADER_SEPARATOR_BAND_WIDTH, shape[0])
        if w <= 0 or band_bottom <= band_top:
            continue
        mask[band_top:band_bottom, x : min(x + max(w, 1), shape[1])] = HEADER_LABEL
    return mask


def _is_header(page: Page, region: Region) -> bool:
    """True when *region* is typed as a header for this page's config."""
    return region.type_name == "header"


def _scales(shape: tuple[int, int], target_size: tuple[int, int] | None) -> tuple[float, float]:
    """``(scale_x, scale_y)`` mapping a page-sized canvas onto *target_size*."""
    if target_size is None or shape == tuple(target_size):
        return (1.0, 1.0)
    height, width = shape
    target_height, target_width = target_size
    return (target_width / max(width, 1), target_height / max(height, 1))


def _scaled(contour: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Contour scaled by the page→network factor (see :func:`~.geometry.scale_contour`)."""
    if scale_x == 1.0 and scale_y == 1.0:
        return contour
    return scale_contour(contour, scale_x=scale_x, scale_y=scale_y)


def region_label_map(
    region_contour: np.ndarray,
    page: Page,
    shape: tuple[int, int],
    *,
    header_mask: np.ndarray | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Label map in which *region_contour* is :data:`QUERY_LABEL`.

    The ``img_label`` of the legacy loop: fill the queried region with 1, then
    overwrite image regions with 2 and headers with 3.  Those two overwrites run
    *after* the fill and therefore win — reproduced here so the samples match the
    dataset already trained on.

    Note the difference from the legacy code, which built ``(y_len, x_len, N)`` and
    resized the whole stack with ``cv2.resize``: OpenCV refuses more than four
    channels, so any page with more than four text regions raised ``cv2.error``.
    Scaling the contour coordinates (:func:`~.geometry.scale_contour`) replaces that.
    """
    scale_x, scale_y = _scales(shape, target_size)
    mask = mask_of(_scaled(region_contour, scale_x, scale_y), shape, fill_value=QUERY_LABEL)
    images = [region.contour for region in page.regions if region.kind is RegionKind.IMAGE]
    if images:
        image_mask = mask_of([_scaled(c, scale_x, scale_y) for c in images], shape, fill_value=1)
        mask = np.where(image_mask.astype(bool), IMAGE_REGION_LABEL, mask)
    if header_mask is None:
        header_mask = header_separator_mask(page, shape, target_size=target_size)
    return np.where(header_mask == HEADER_LABEL, HEADER_LABEL, mask).astype(np.uint8)


def generate_pairs(
    page: Page,
    *,
    input_height: int | None = None,
    input_width: int | None = None,
    min_area: float = DEFAULT_MIN_AREA_FRACTION,
    max_area: float = DEFAULT_MAX_AREA_FRACTION,
    min_area_early: float | None = None,
    index_start: int = 0,
) -> list[ReadingOrderPair]:
    """All ``n * (n - 1)`` ordered-pair samples of one page.

    Parameters
    ----------
    page:
        Parsed page to build samples from.
    input_height, input_width:
        Network input size; the label maps are rasterised at this size directly
        rather than resized afterwards.  Defaults to the page's own size.
    min_area, max_area, min_area_early:
        Area thresholds, as fractions of the page area (see
        :func:`~.geometry.filter_contours_by_area`).
    index_start:
        First counter value, for appending to an existing dataset (the legacy
        ``index_start``/``indexer_start``).

    Returns
    -------
    list[ReadingOrderPair]
        Empty when the page has fewer than two usable text regions.
    """
    shape = (page.height, page.width)
    target = (int(input_height or page.height), int(input_width or page.width))
    filtered = ordered_text_regions(
        page,
        min_area=min_area,
        max_area=max_area,
        min_area_early=min_area_early,
    )
    if len(filtered.contours) < 2:
        return []

    header_mask = header_separator_mask(page, shape, target_size=target)
    layout_channel = _layout_channel(page, shape, target)
    label_maps = [
        region_label_map(contour, page, shape, header_mask=header_mask, target_size=target)
        for contour in filtered.contours
    ]

    pairs: list[ReadingOrderPair] = []
    misses = filtered.below_earlier_min
    for counter, (i, j) in enumerate(_ordered_pairs(len(filtered.contours))):
        missed = bool(misses[i]) if misses is not None else False
        pairs.append(
            ReadingOrderPair(
                name=f"{page.file_name}_{index_start + counter}{'_missed' if missed else ''}",
                class_type=_pair_label(filtered.order_index, i, j),
                modal_image=np.stack([label_maps[i], layout_channel, label_maps[j]], axis=-1),
                missed=missed,
            )
        )
    return pairs


def _ordered_pairs(count: int) -> Iterator[tuple[int, int]]:
    """Every ordered pair ``(i, j)`` with ``i != j`` — ``count * (count - 1)`` of them.

    ``(i, j)`` and ``(j, i)`` are both emitted: the modal image and the label differ
    between them (see the module docstring), so the reverse pair is a real sample.
    """
    return ((i, j) for i in range(count) for j in range(count) if i != j)


def _pair_label(order_index: list[int], i: int, j: int) -> int:
    """``1`` when region *i* precedes *j* in reading order, otherwise ``0``.

    The legacy rule was ``index_i - index_j < 0``; without any recorded index the
    legacy code still labelled every pair ``0``, which is reproduced here.
    """
    if not order_index or i >= len(order_index) or j >= len(order_index):
        return 0
    return int(order_index[i] < order_index[j])


def _layout_channel(page: Page, shape: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    """Channel 1 of the modal image: the page layout raster, as one label plane.

    The legacy code used ``img_poly[:, :, 0]`` — the blue channel of the RGB render —
    which is a colour value, not a class id, and is the reason the modal image had to
    be widened from ``int8``.  When the page has no rendered layout the region labels
    themselves stand in, so the channel is never blank.
    """
    rendered = render_page(page, palette=DEFAULT_LABEL_PALETTE, size=shape)
    if target == shape:
        return rendered[..., 0]
    # Rasterise at page size, then resample: cv2.resize on a label map keeps the class
    # ids intact, unlike resizing the stacked label volume the legacy code built.
    return cv2.resize(
        to_uint8_bgr(rendered)[..., 0],
        (target[1], target[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def write_pair(
    pair: ReadingOrderPair,
    images_dir: str | Path,
    classes_dir: str | Path,
) -> tuple[Path, Path]:
    """Write *pair*'s modal image and its label, returning both paths.

    Reproduces the legacy output contract: ``<name>.png`` into the image directory and
    ``<name>.npy`` holding the scalar class into the classes directory.  The class file
    is written as a 0-d array so a plain ``np.load`` returns the same value the legacy
    ``np.save(..., class_type)`` produced.
    """
    image_path = write_image(Path(images_dir) / f"{pair.name}.png", pair.modal_image)
    class_path = Path(classes_dir) / f"{pair.name}.npy"
    class_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(class_path, np.asarray(pair.class_type, dtype=np.int8))
    return image_path, class_path


def visualize_reading_order(
    image: np.ndarray,
    regions: Sequence[Region],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw the reading order onto *image* as a chain of arrows.

    Port of ``overlay_layout_on_image``'s arrow-drawing half.  *regions* are joined in
    the order given — pass them straight from :func:`ordered_regions`.  Two fixes: the
    legacy blend used ``cv2.addWeighted(img, 0.5, layout, 0.1, 0)``, whose weights sum
    to 0.6 and dimmed the result, and it drew onto a single-channel canvas that it then
    mixed with a three-channel photo.  :func:`~.render.overlay` now owns the blend
    maths and normalises both operands.
    """
    region_list = list(regions)
    if not region_list:
        raise ValueError("visualize_reading_order(): no regions given")
    shape = (int(image.shape[0]), int(image.shape[1]))
    return overlay(image, _arrows(region_list, shape, color, thickness), alpha)


def _arrows(
    regions: Sequence[Region],
    shape: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """The arrow chain through *regions*' centroids, on a black canvas."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    points = [ContourFeatures.from_contour(region.contour, shape).centroid for region in regions]
    for (x1, y1), (x2, y2) in pairwise(points):
        cv2.arrowedLine(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=thickness,
            tipLength=0.03,
        )
    return canvas
