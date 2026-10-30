"""Contour arithmetic shared by the parser, the renderer and the generator.

Consolidates the geometry helpers that the legacy code repeated across
``gt_gen_utils.py`` and ``utils.py``:

=========================================================  ==========================
legacy                                                     here
=========================================================  ==========================
``get_contours_of_image`` / ``get_contours`` (x2)           :func:`contours_of_image`
``filter_contours_area_of_image`` (x2, one frozen index)   :func:`filter_contours_by_area`
``filter_contours_area_of_image_tables``                   :func:`filter_contours_by_area`
``return_parent_contours``                                 :func:`return_parent_contours`
``get_contours_of_region`` (``np.repeat`` BGR round-trip)  :func:`contours_of_region`
``update_region_contours`` (bare ``except:``)              :func:`update_region_contours`
``bounding_box`` (undefined ``scale_w``/``scale_h``)       :func:`bounding_box`
``find_new_features_of_contours``                          :class:`ContourFeatures`
``rotatedRectWithMaxArea`` (``utils.py``)                  :func:`rotated_rect_with_max_area`
``largest_contour`` / ``get_largest_contour_bbox``         :func:`largest_contour_bbox`
``resize_image`` (x3 copies)                               ``sbb_pytorch.storage.images``
=========================================================  ==========================

Areas are measured with :func:`cv2.contourArea` throughout.  The legacy code
mixed ``cv2.contourArea`` and ``shapely.geometry.Polygon(...).area``; the two
disagree on collinear and self-intersecting polygons, which made the two copies
of the area filter disagree about which contours to keep.  ``shapely`` is no
longer a dependency of this package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np

__all__ = [
    "DEFAULT_KERNEL",
    "DEFAULT_MIN_AREA_FRACTION",
    "DEFAULT_MIN_AREA_PIXELS",
    "BoundingBox",
    "ContourFeatures",
    "FilteredContours",
    "bounding_box",
    "contour_area",
    "contour_bbox",
    "contour_centroid",
    "contour_features",
    "contours_of_image",
    "contours_of_region",
    "filter_contours_by_area",
    "largest_contour",
    "largest_contour_bbox",
    "mask_of",
    "return_parent_contours",
    "rotated_rect_with_max_area",
    "scale_contour",
    "update_region_contours",
]

#: ``(x, y, w, h)`` — the OpenCV bounding-box convention used across the package.
BoundingBox = tuple[int, int, int, int]

#: Structuring element used by every legacy erosion/dilation (``KERNEL`` in
#: ``gt_gen_utils.py``: ``np.ones((5, 5), np.uint8)``).
DEFAULT_KERNEL: Final[np.ndarray] = np.ones((5, 5), dtype=np.uint8)

#: Smallest region size, as a *fraction of the page area*, that the label tool kept.
#: The legacy default was ``0.0002`` (``return_contours_of_interested_region``), and
#: ``filter_contours_area_of_image`` compared areas against
#: ``min_area * np.prod(image.shape[:2])`` — a fraction, never an absolute pixel count.
DEFAULT_MIN_AREA_FRACTION: Final = 0.0002

#: Floor in absolute pixels, so a sub-pixel contour on a very large page is still
#: dropped (the legacy fraction alone permits a 1-pixel region on a 5000x5000 page).
DEFAULT_MIN_AREA_PIXELS: Final = 5.0


def contour_area(contour: np.ndarray) -> float:
    """Area of *contour* in pixels, via :func:`cv2.contourArea`.

    Accepts ``(N, 2)``, ``(N, 1, 2)`` or ``(K, N, 2)`` of any numeric dtype; the dtype and
    shape normalisation is :func:`_as_cv_contour` alone.  Its earlier whitelist listed
    ``float64`` as already-measurable, and since ``cv2.contourArea`` accepts ``int32`` and
    ``float32`` only, that one dtype -- the one almost every numpy operation hands back,
    from ``contour * 2.0`` to a JSON round-trip -- raised ``cv2.error``.  The legacy
    ``find_new_features_of_contours`` wrapped this in ``try/except`` to work around
    exactly that dtype/shape variety.
    """
    array = np.asarray(contour)
    if array.size == 0:
        return 0.0
    if array.ndim == 2 and array.shape[0] < 3:
        return 0.0  # A polygon cannot have fewer than three points.
    return float(cv2.contourArea(_as_cv_contour(array)))


def contour_bbox(contour: np.ndarray, image_shape: Sequence[int] | None = None) -> BoundingBox:
    """``(x, y, w, h)`` bounding box of *contour*.

    When *image_shape* is given the box is clipped to the image, which the legacy
    ``bounding_box`` did by hand for some callers only — the unclipped variant
    produced boxes outside the canvas whenever a contour carried a stray vertex.
    """
    array = np.asarray(contour)
    if array.size == 0:
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(_as_cv_contour(array))
    if image_shape is not None and len(image_shape) >= 2:
        height, width = int(image_shape[0]), int(image_shape[1])
        x, y = max(0, x), max(0, y)
        w, h = max(0, min(w, width - x)), max(0, min(h, height - y))
    return (int(x), int(y), int(w), int(h))


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    """Centroid ``(cx, cy)`` of *contour*, falling back to the box centre.

    The legacy helper guarded the division with ``+ 1e-32`` rather than testing
    ``m00 == 0``, which yields a near-zero quotient instead of a meaningful point
    for degenerate (collinear or duplicated-vertex) contours; ``nan``/``inf`` then
    propagated into the reading-order sort.  The box centre is used instead.
    """
    array = _as_cv_contour(np.asarray(contour))
    moments = cv2.moments(array)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(array)
        return (x + w / 2.0, y + h / 2.0)
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _as_cv_contour(array: np.ndarray) -> np.ndarray:
    """Return *array* as the ``(N, 1, 2)`` ``int32`` array every OpenCV build accepts.

    Two normalisations in one place, because every OpenCV entry point demands both and
    every legacy call site did only one of them:

    * *shape* -- ``(N, 2)`` works on most builds, ``(N, 1, 2)`` on all of them, so the
      ``np.expand_dims`` and the ``if len(shape)==3: c = c[0]`` unwrapping repeated at six
      legacy call sites collapse into one reshape; a ``(K, N, 2)`` batch becomes
      ``(K*N, 1, 2)``, which is what the legacy ``fillPoly`` calls were handed anyway.
    * *dtype* -- ``cv2.contourArea`` takes ``int32``/``float32`` and ``cv2.fillPoly``
      ``int32`` only, so a ``float64``, ``uint8`` or ``int64`` contour has to be cast
      before either call, not merely at one of them.  Truncating matches the parser's
      ``int(float(x))`` and OpenCV's own internal narrowing.
    """
    array = np.asarray(array)
    if array.dtype != np.int32:
        array = array.astype(np.int32)
    if array.ndim == 3 and array.shape[1] == 1:
        return array
    if array.ndim in (1, 3):
        array = array.reshape(-1, 2)
    return np.ascontiguousarray(array.reshape(-1, 1, 2))


def mask_of(
    contours: Iterable[np.ndarray] | np.ndarray,
    shape: Sequence[int] | int,
    *,
    fill_value: int = 255,
) -> np.ndarray:
    """Rasterise *contours* onto a blank ``uint8`` canvas of *shape*.

    Replaces the repeated ``np.zeros(...); cv2.fillPoly(...)`` pair, and the
    ``np.repeat(grayscale[..., None], 3, axis=-1)`` round-trip the legacy
    ``get_contours_of_region`` used to turn a single-channel mask into a 3-channel
    image before the very next line converted it back to grayscale.

    Parameters
    ----------
    contours:
        One contour, a list of contours, or a stacked ``(K, N, 2)`` batch.  Each is
        normalised through :func:`_as_cv_contour` first, so a ``uint8`` or ``float64``
        contour rasterises instead of raising, and a point-less one paints nothing.
    shape:
        ``(height)``, ``(height, width)`` or ``(height, width, channels)``, or the bare
        ``int`` height the one-element form abbreviates.
    fill_value:
        Value painted inside the contours.  On a multi-channel canvas it is repeated
        across the channels -- ``cv2.fillPoly`` with a scalar colour would otherwise
        write channel 0 and leave the rest of the canvas black.
    """
    target = (shape, shape) if isinstance(shape, int | np.integer) else tuple(shape)
    if len(target) == 1:
        target = (target[0], target[0])
    canvas = np.zeros(target, dtype=np.uint8)
    channels = target[2] if len(target) == 3 else 1
    colour = int(fill_value) if channels == 1 else (int(fill_value),) * channels
    for contour in _iter_contours(contours):
        if contour.shape[0] == 0:
            continue  # fillPoly rejects a point-less contour; there is nothing to paint.
        cv2.fillPoly(canvas, [contour], colour)
    return canvas


def _iter_contours(contours: Iterable[np.ndarray] | np.ndarray) -> Iterable[np.ndarray]:
    """Yield each contour in *contours*, unwrapping a bare single contour.

    A contour is ``(N, 2)``; a *list* of contours is a list, or a stacked
    ``(K, N, 2)`` array.  The legacy code got this wrong by handing a bare
    ``(N, 2)`` array to a call site expecting a list, rasterising one vertex.
    """
    if isinstance(contours, np.ndarray):
        array = np.asarray(contours)
        if array.ndim == 3 and array.shape[1] != 1:
            for contour in array:  # ``(K, N, 2)`` batch: every row normalised, not just one.
                yield _as_cv_contour(contour)
            return
        yield _as_cv_contour(array)
        return
    for contour in contours:
        yield _as_cv_contour(np.asarray(contour))


def contours_of_image(
    image: np.ndarray,
    *,
    min_area: float = DEFAULT_MIN_AREA_PIXELS,
    mode: int = cv2.RETR_LIST,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> list[np.ndarray]:
    """All contours of a raster *image*, with sub-pixel noise removed.

    The legacy ``get_contours_of_image`` ran without an area floor, so JPEG artefacts
    produced hundreds of one-pixel contours downstream.  *min_area* here is in
    absolute pixels; use :func:`filter_contours_by_area` for the page-relative rule.
    """
    if image is None or np.asarray(image).size == 0:
        return []
    found, _ = cv2.findContours(np.ascontiguousarray(image), mode, method)
    return [contour for contour in found if contour_area(contour) >= min_area]


def contours_of_region(
    region_image: np.ndarray,
    *,
    min_area: float = DEFAULT_MIN_AREA_PIXELS,
) -> list[np.ndarray]:
    """Contours inside a single rendered region image.

    Replaces ``get_contours_of_region``, which built a 3-channel image from a
    grayscale mask via ``np.repeat``, then ``cv2.cvtColor``-ed it back to grayscale,
    then ``cv2.threshold(imgray, 0, 255, 0)`` — an Otsu-free round-trip that
    allocated three times the pixels to produce the mask it already had.
    """
    array = np.asarray(region_image)
    if array.size == 0:
        return []
    gray = array if array.ndim == 2 else array[..., 0]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return contours_of_image(binary, min_area=min_area)


@dataclass(frozen=True, slots=True)
class FilteredContours:
    """Result of :func:`filter_contours_by_area`.

    Attributes
    ----------
    contours:
        The kept contours, in the input order.
    order_index:
        The matching entries of the input ``order_index``, when one was supplied —
        the reading order of each kept region.
    below_earlier_min:
        ``1`` for each kept contour that the *earlier*, larger ``min_area`` would
        have rejected; ``None`` when no earlier threshold was given.  This is how
        the tool marked the regions that a smaller ``min_area`` newly admitted, so a
        re-run could label only the new samples (the ``_missed`` suffix).
    """

    contours: list[np.ndarray]
    order_index: list[int] = field(default_factory=list)
    below_earlier_min: list[int] | None = None


def filter_contours_by_area(
    contours: Iterable[np.ndarray],
    page_shape: Sequence[int] | np.ndarray,
    *,
    min_area: float = DEFAULT_MIN_AREA_FRACTION,
    max_area: float = 1.0,
    min_area_early: float | None = None,
    order_index: Sequence[int] | None = None,
) -> FilteredContours:
    """Keep contours sized within *min_area* .. *max_area* **fractions of the page**.

    Merges ``filter_contours_area_of_image`` and its ``_tables`` twin, which differed
    only in that one tracked ``order_index`` and ``min_area_early`` and the other did
    not.  The two bugs fixed here:

    * the ``_tables`` variant advanced its ``jv`` counter only on the branch that
      *kept* a contour, so a rejected contour froze the counter and every later
      comparison reused the wrong index; ``enumerate`` replaces the manual counter;
    * the pair disagreed on how to measure area (shapely vs OpenCV), so the same set
      of contours filtered two different ways.  OpenCV's measure is used here.

    Parameters
    ----------
    contours:
        Contours to filter; ``(N, 2)``/``(N, 1, 2)`` per contour.
    page_shape:
        Page shape (or any array) whose area the fractions are relative to.
    min_area, max_area:
        Fraction of the page area, e.g. ``0.0002`` = 0.02 % of the page.
    min_area_early:
        The ``min_area`` used by an earlier dataset generation; see
        :attr:`FilteredContours.below_earlier_min`.
    order_index:
        Reading-order index aligned with *contours*.
    """
    contour_list = list(contours)
    # An ndarray gives its own shape; anything else *is* the shape tuple. max() keeps a
    # degenerate page from dividing the threshold down to nothing.
    dims = page_shape.shape[:2] if isinstance(page_shape, np.ndarray) else tuple(page_shape)[:2]
    page_area = max(float(np.prod(dims)), 1.0) if len(dims) == 2 else 1.0

    kept: list[np.ndarray] = []
    kept_index: list[int] = []
    below: list[int] | None = [] if min_area_early is not None else None

    for position, contour in enumerate(contour_list):
        array = _as_cv_contour(np.asarray(contour))
        if array.shape[0] < 3:  # A polygon cannot have fewer than three points.
            continue
        area = contour_area(array)
        if not (area >= min_area * page_area and area <= max_area * page_area):
            continue
        kept.append(array)
        if order_index is not None and position < len(order_index):
            kept_index.append(int(order_index[position]))
        if below is not None:
            below.append(1 if area < min_area_early * page_area else 0)

    return FilteredContours(contours=kept, order_index=kept_index, below_earlier_min=below)


def return_parent_contours(
    contours: Sequence[np.ndarray],
    hierarchy: np.ndarray | None = None,
) -> list[np.ndarray]:
    """The outermost contours of *contours*.

    Two legacy variants existed: one required a *hierarchy* (so it could not be used
    on a hand-built contour list at all) and one recomputed it from an image.  Here
    the hierarchy is optional and, when absent, the parent set is approximated by
    containment — which is the behaviour the callers actually relied on.
    """
    contour_list = list(contours)
    if not contour_list:
        return []
    if hierarchy is not None:
        first_child = np.asarray(hierarchy).reshape(-1, 4)[:, 3]
        roots = [c for c, child in zip(contour_list, first_child, strict=False) if child == -1]
        return roots or contour_list[:1]

    ranked = sorted(contour_list, key=contour_area, reverse=True)
    parents: list[np.ndarray] = []
    for contour in ranked:
        vertex = _as_cv_contour(np.asarray(contour))[0][0]
        nested = any(
            cv2.pointPolygonTest(parent, tuple(int(v) for v in vertex), False) >= 0
            for parent in parents
        )
        if nested:
            continue
        parents.append(contour)
    return parents


def update_region_contours(
    contours: Sequence[np.ndarray],
    image_shape: Sequence[int],
    *,
    erosion_rate: int,
    dilation_rate: int,
    kernel: np.ndarray | None = None,
    erosion_early: int | None = None,
    dilation_early: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Shrink each region and return the ring that its dilation adds back.

    Faithful port of ``update_region_contours``: rasterise the region, optionally
    pre-erode/dilate (the ``*_early`` knobs used when regenerating a dataset with
    different rates), erode by *erosion_rate*, take the resulting outline, then add
    ``dilate(eroded) - eroded`` to the shared boundary mask.

    Note the asymmetry inherited from the original: the artificial class covers the ring
    *outside* the eroded contour, so the strip the erosion removed becomes background,
    not boundary.  The eroded contour — not the original — is what the caller renders as
    the region, which is the whole point of the class.

    The bare ``except:`` around ``con_eroded[0]`` is replaced by an explicit
    ``if found:`` — the old one silently fell back to the un-eroded contour on *any*
    error (bad dtype, empty mask), which hid malformed regions rather than reporting them.
    """
    structuring = DEFAULT_KERNEL if kernel is None else kernel
    eroded_contours: list[np.ndarray] = []
    boundary = np.zeros(tuple(image_shape)[:2], dtype=np.uint8)

    for contour in contours:
        working = mask_of(contour, image_shape, fill_value=1)
        if dilation_early:
            working = cv2.dilate(working, structuring, iterations=int(dilation_early))
        if erosion_early:
            working = cv2.erode(working, structuring, iterations=int(erosion_early))
        if erosion_rate > 0:
            working = cv2.erode(working, structuring, iterations=int(erosion_rate))

        found = contours_of_image(working, min_area=DEFAULT_MIN_AREA_PIXELS)
        eroded_contours.append(found[0] if found else _as_cv_contour(np.asarray(contour)))

        # The ring is whatever the dilation adds to the eroded mask (legacy
        # ``boundary = img_boundary_in_dilated - img_boundary_in``); with
        # ``dilation_rate=0`` the dilation is a copy and the ring stays empty.
        dilated = cv2.dilate(working, structuring, iterations=int(dilation_rate))
        boundary = np.maximum(boundary, (dilated > working).astype(np.uint8))

    return eroded_contours, boundary


@dataclass(frozen=True, slots=True)
class ContourFeatures:
    """Geometry of one contour, computed once and reused.

    Replaces the seven parallel lists ``find_new_features_of_contours`` returned
    (``cx, cy, x_min, x_max, y_min, y_max, y_corr_x_min_from_argmin``) — whose
    ``areas_main`` member was computed and never read by any caller — with a single
    record.  Its ``try/except`` around the ``(N,1,2)`` vs ``(N,2)`` indexing
    disappears with :func:`_as_cv_contour`.
    """

    contour: np.ndarray
    area: float
    bbox: BoundingBox
    centroid: tuple[float, float]
    aspect_ratio: float
    extent: float
    y_at_x_min: int

    @classmethod
    def from_contour(
        cls, contour: np.ndarray, image_shape: Sequence[int] | None = None
    ) -> ContourFeatures:
        """Compute every feature of *contour* in one pass."""
        array = _as_cv_contour(np.asarray(contour))
        points = array.reshape(-1, 2)  # ``(N, 1, 2)`` and ``(N, 2)`` views alike.
        bbox = contour_bbox(array, image_shape)
        x, y, w, h = bbox
        area = contour_area(array)
        cx, cy = contour_centroid(array)
        return cls(
            contour=array,
            area=area,
            bbox=bbox,
            centroid=(cx, cy),
            aspect_ratio=(w / h) if h else 0.0,
            extent=(area / float(w * h)) if w and h else 0.0,
            y_at_x_min=int(points[np.argmin(points[:, 0]), 1]) if points.size else y,
        )


def contour_features(
    contours: Iterable[np.ndarray],
    image_shape: Sequence[int] | None = None,
) -> list[ContourFeatures]:
    """Features for every contour in *contours* (see :class:`ContourFeatures`)."""
    return [ContourFeatures.from_contour(contour, image_shape) for contour in _iter_many(contours)]


def _iter_many(contours: Iterable[np.ndarray] | np.ndarray) -> Iterable[np.ndarray]:
    """Iterate *contours* as individual contours, accepting a bare array or a list."""
    return _iter_contours(contours)


def bounding_box(
    contour: np.ndarray,
    color: int | tuple[int, ...] | None = None,
    corr_order_index: int | None = None,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> BoundingBox:
    """Bounding box of *contour*, optionally scaled about its own top-left vertex.

    Signature and semantics of the legacy ``bounding_box(cnt, color, corr_order_index)``,
    whose body referenced ``scale_w`` and ``scale_h`` — names that do not exist in
    that module, so **every** call raised ``NameError``.  The scaling now uses the
    explicit *scale_x*/*scale_y* pair, so an anisotropically resized page yields the
    box of the resized contour rather than the unscaled one.
    """
    del color, corr_order_index  # Kept for call compatibility; a box does not depend on them.
    array = np.asarray(contour)
    if array.size == 0:
        return (0, 0, 0, 0)
    if scale_x != 1.0 or scale_y != 1.0:
        array = scale_contour(array, scale_x=scale_x, scale_y=scale_y)
    return contour_bbox(array)


def scale_contour(
    contour: np.ndarray,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    origin: tuple[float, float] | None = None,
) -> np.ndarray:
    """Scale *contour* about *origin* (default: its own top-left vertex).

    The legacy scaling helpers used the image origin, which is right for a whole page
    and wrong for a region crop; *origin* makes both expressible.
    """
    array = np.asarray(contour).reshape(-1, 2).astype(np.float64)
    source_min = (array[:, 0].min(), array[:, 1].min())
    anchor = source_min if origin is None else origin
    scaled = np.stack([array[:, 0] * scale_x, array[:, 1] * scale_y], axis=1)
    return np.rint(scaled - np.asarray(source_min) + np.asarray(anchor)).astype(np.int32)


def largest_contour(contours: Sequence[np.ndarray]) -> np.ndarray | None:
    """The largest contour in *contours* by area, or ``None``."""
    if not len(contours):
        return None
    return max(contours, key=contour_area)


def largest_contour_bbox(
    contours: Sequence[np.ndarray],
    shape: Sequence[int] | None = None,
) -> BoundingBox | None:
    """Bounding box of the single largest contour in *contours*.

    Replaces ``get_largest_contour_bbox``, which rasterised every contour with
    ``fillPoly``, re-ran ``findContours`` over the merged mask and took the largest
    result.  Merging is retained because two overlapping contours become one
    connected component — the behaviour the print-space extraction depends on.
    """
    contour_list = list(contours)
    if not contour_list:
        return None
    target = tuple(shape) if shape is not None else _tight_shape(contour_list)
    merged = mask_of(contour_list, target)
    found = contours_of_image(merged)
    if not found:
        return None
    return contour_bbox(max(found, key=contour_area))


def _tight_shape(contours: Sequence[np.ndarray]) -> tuple[int, int]:
    """Smallest ``(height, width)`` canvas containing every contour in *contours*."""
    stacked = np.concatenate([np.asarray(contour).reshape(-1, 2) for contour in contours])
    x, y, w, h = cv2.boundingRect(_as_cv_contour(stacked.astype(np.int32)))
    return (int(y + h), int(x + w))


def _max_inscribed_size(width: int, height: int, alpha: float) -> tuple[float, float]:
    """``(width, height)`` of the largest axis-aligned rectangle that fits inside a
    ``width x height`` rectangle rotated by *alpha* radians
    (:func:`utils.rotatedRectWithMaxArea`, ``utils.py:318``).

    The legacy helper answers with a *size*, never with corners, and takes radians -- the
    wrapper below converts.  Two cases cover every angle, because the answer for
    ``angle``, ``-angle`` and ``180 - angle`` is identical and only the first quadrant has
    to be measured: the *half-constrained* one, where two corners of the inner rectangle
    touch the long side and the other two rest on the mid-line parallel to it, and the
    *fully constrained* one, where the inner rectangle touches all four sides.  The
    ``abs(sin_a - cos_a) < 1e-10`` test is the 45-degree corner of the first case, the one
    angle at which the second case's ``cos(2a)`` divisor vanishes.
    """
    width_is_longer = width >= height
    longer, shorter = (width, height) if width_is_longer else (height, width)
    sin_alpha, cos_alpha = abs(float(np.sin(alpha))), abs(float(np.cos(alpha)))

    half_constrained = shorter <= 2.0 * sin_alpha * cos_alpha * longer or (
        abs(sin_alpha - cos_alpha) < 1e-10
    )
    if half_constrained:
        half = 0.5 * shorter
        return (
            (half / sin_alpha, half / cos_alpha)
            if width_is_longer
            else (half / cos_alpha, half / sin_alpha)
        )

    cos_2a = cos_alpha * cos_alpha - sin_alpha * sin_alpha
    return (
        (width * cos_alpha - height * sin_alpha) / cos_2a,
        (height * cos_alpha - width * sin_alpha) / cos_2a,
    )


def rotated_rect_with_max_area(width: int, height: int, angle_degrees: float) -> np.ndarray:
    """Four ``(x, y)`` corners of the largest axis-aligned rectangle that fits inside
    a ``width x height`` rectangle rotated by *angle_degrees*.

    Port of ``rotatedRectWithMaxArea`` (``utils.py:318``), used to render a deskewed
    page's usable area without black corners.  The size comes from
    :func:`_max_inscribed_size` and the placement from its caller
    :func:`utils.rotate_max_area` (``utils.py:346``), which crops that size out of the
    rotated image centred on ``w // 2, h // 2``; the corners below are that crop, as an
    int32 ``(4, 2)`` ring running top-left, top-right, bottom-right, bottom-left.

    The first port collapsed the trigonometry to identities that only survived the
    right-angle cases -- every other angle came back as the ``(-1, -1)`` sentinel or a
    quad with a repeated vertex, so the helper could describe an axis-aligned page and
    nothing else.  Corner coordinates may now leave the ``width x height`` canvas, exactly
    as the legacy slice did: a ``100 x 50`` page turned 90 degrees is a ``50 x 100`` shape
    whose top edge sits at ``y = -25``.  A page with no area still has no rectangle to
    inscribe, and keeps the ``(-1, -1)`` corners.
    """
    if width <= 0 or height <= 0:
        return np.full((4, 2), -1, dtype=np.int32)

    size_x, size_y = _max_inscribed_size(width, height, float(np.deg2rad(angle_degrees)))
    left, top = width / 2.0 - size_x / 2.0, height / 2.0 - size_y / 2.0
    corners = np.array(
        [
            [left, top],
            [left + size_x, top],
            [left + size_x, top + size_y],
            [left, top + size_y],
        ],
        dtype=np.float64,
    )
    return np.rint(corners).astype(np.int32)
