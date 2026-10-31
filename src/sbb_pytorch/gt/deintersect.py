"""Separate overlapping polygons -- the shapely port from ``oxygraphos-ocr``.

Ported from ``dfine-det/dfine_det/serialization/deintersect.py``, which is byte-identical in the
three detection siblings of that workspace; ``surya-seg/surya_seg/lib/preprocess_polygons.py`` is
the same three functions under the names used here.  The problem it solves is the one a pixelwise
model cannot learn around: two labelled zones that overlap share the pixels between them, and a
label image paints one over the other, so the boundary the annotation drew is not the boundary the
model is supervised on.  On ``BL_Or_1443_300`` the class-2 zone is one web of rings around the
text lines, and every class-1 fragment inside a ring is painted over by it.

This package's version speaks the contour type the rest of ``gt`` uses -- :data:`parser.Contour`,
normalised by :func:`~sbb_pytorch.gt.geometry._as_cv_contour` -- instead of upstream's
``list[list[tuple[float, float]]]``, so a trimmed contour goes straight back into a
:class:`~sbb_pytorch.gt.parser.Region` or a :func:`~sbb_pytorch.gt.render.label_image` call.  The
``changed`` flags upstream also returns are dropped: a caller who needs to know whether a polygon
moved can diff the contour it gets back, and one that a neighbour consumed comes back empty rather
than being reported twice.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Final

import numpy as np

from ..exceptions import GeometryError
from .geometry import _as_cv_contour, _iter_contours
from .parser import Contour

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = [
    "DEFAULT_GAP",
    "DEFAULT_MAX_PASSES",
    "DEFAULT_MIN_INTERSECTION_AREA",
    "DEFAULT_SIMPLIFY_TOLERANCE",
    "deintersect_polygons",
    "polygon_from_points",
    "polish_polygon",
    "separate_intersecting_pair",
    "shapely_available",
]

#: The width a trimmed pair is left apart, in pixels (upstream's ``gap``, ``deintersect.py:74``).
#: The intersection is buffered by half of this on each side before it is subtracted from both,
#: so the two zones end up this far apart rather than edge to edge.
DEFAULT_GAP: Final = 1.5

#: Smallest area, in px², an overlap may have before it is worth trimming (``min_intersection_area``,
#: ``deintersect.py:75``).  Below it the two zones are merely touching, and shaving a sliver off
#: both would move a boundary the annotation drew on purpose.
DEFAULT_MIN_INTERSECTION_AREA: Final = 1.0

#: The shapely ``simplify`` tolerance applied to a polygon *after* it was trimmed
#: (``simplify_tolerance``, ``deintersect.py:76``).  Not the Douglas-Peucker epsilon of
#: :mod:`sbb_pytorch.gt.simplify`: this one runs on the shapely geometry the subtraction produced,
#: whose vertex count is a product of the boolean operation rather than of the original outline.
DEFAULT_SIMPLIFY_TOLERANCE: Final = 0.5

#: How many pairwise sweeps to make (``max_passes``, ``deintersect.py:77``), with a sweep that
#: changed nothing ending them early.  Two is upstream's answer: trimming one pair changes what
#: its neighbours overlap, so one more sweep usually settles the rest, and a full fixpoint loop on
#: a page of a few thousand polygons costs more than the change it finds.
DEFAULT_MAX_PASSES: Final = 2


def shapely_available() -> bool:
    """Whether ``shapely`` can be imported -- what de-intersecting any polygon needs.

    The question :func:`sbb_pytorch.storage.checkpoint.safetensors_available` asks about
    ``safetensors`` (itself the pattern of
    :func:`sbb_pytorch.training.trainer.tensorboard_available`), asked the same way:
    ``importlib.util.find_spec`` at call time, so a caller can be tested by monkeypatching this one
    function instead of breaking an import.  It is a declared dependency of the package, so
    ``False`` means an incomplete install -- and an explicit request to de-intersect is *refused*
    (:class:`~sbb_pytorch.exceptions.GeometryError`) rather than silently skipped, exactly as
    ``Checkpoint.save`` refuses an explicit ``.safetensors``.
    """
    return importlib.util.find_spec("shapely") is not None


@lru_cache(maxsize=1)
def _backend() -> ModuleType:
    """The cached ``shapely.geometry`` module: ``Polygon``, ``MultiPolygon``, ``base``.

    Cached because importing it costs more than a dict lookup, and imported through
    :func:`shapely_available` so a machine without the package can still *import this module* --
    only asking it for work raises.  Every public function below goes through this one, which is
    what lets a test fake the whole dependency by patching two functions.
    """
    if not shapely_available():
        raise GeometryError(
            "de-intersecting polygons needs the shapely package, which this environment cannot "
            "import; it is declared in pyproject.toml as ``shapely>=2.0`` (install it into the "
            'virtualenv with `uv pip install --python <venv python> "shapely>=2.0"`)'
        )
    from shapely import geometry

    return geometry


def _largest_polygon(geometry: BaseGeometry) -> BaseGeometry:
    """The biggest polygon inside *geometry* -- a polygon itself, or the biggest of a multipolygon.

    Upstream's ``_largest_polygon`` (``deintersect.py:11``), widened from ``Polygon`` to any
    ``BaseGeometry``: ``buffer(0)`` on a self-intersecting ring hands back a ``MultiPolygon``, and
    the output loop used to read ``.exterior`` on whatever came back -- an ``AttributeError`` on
    exactly the repaired polygons the repair exists to save.
    """
    backend = _backend()
    if isinstance(geometry, backend.Polygon):
        return geometry
    parts = [part for part in getattr(geometry, "geoms", ()) if isinstance(part, backend.Polygon)]
    return max(parts, key=lambda part: part.area) if parts else backend.Polygon()


def polygon_from_points(points: Contour | Iterable[tuple[int, int]]) -> BaseGeometry:
    """The polygon *points* outline, repaired with ``buffer(0)`` when the ring is invalid.

    Takes any contour shape the package uses (``(N, 2)``, ``(N, 1, 2)``, a list of pairs) through
    :func:`~sbb_pytorch.gt.geometry._as_cv_contour`, so the ``float64`` a numpy round-trip leaves
    behind is a polygon rather than a ``cv2.error``.  Fewer than three vertices is the empty
    polygon -- upstream's rule at ``deintersect.py:21``, and the reason a two-point "polygon" is
    refused instead of being closed into a degenerate sliver.
    """
    backend = _backend()
    ring = _as_cv_contour(np.asarray(points)).reshape(-1, 2)
    if len(ring) < 3:
        return backend.Polygon()
    polygon = backend.Polygon([(float(x), float(y)) for x, y in ring])
    return polygon if polygon.is_valid else polygon.buffer(0)


def polish_polygon(
    polygon: BaseGeometry, simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE
) -> BaseGeometry:
    """``buffer(0)`` then ``simplify`` a trimmed polygon, keeping its largest piece.

    Upstream's ``_polish_polygon`` (``deintersect.py:27``) including its blanket ``except``: a
    shapely operation that raises on one pair must not lose the pair, so the polygon it was
    handed comes back unchanged.  The simplify step is what upstream uses to keep a boolean
    difference from inheriting a vertex per pixel of the neighbour it cut against.
    """
    if polygon.is_empty:
        return polygon
    try:
        polished = polygon.buffer(0).simplify(float(simplify_tolerance), preserve_topology=True)
    except Exception:  # upstream's bare except: a shapely failure keeps the input, not the run.
        return polygon
    return _largest_polygon(polished)


def separate_intersecting_pair(
    first: BaseGeometry,
    second: BaseGeometry,
    *,
    gap: float = DEFAULT_GAP,
    min_intersection_area: float = DEFAULT_MIN_INTERSECTION_AREA,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
) -> tuple[BaseGeometry, BaseGeometry, bool]:
    """The two polygons with their shared area subtracted from **both**, and whether that happened.

    The third value is ``False`` -- and both polygons come back untouched -- when they do not
    intersect, when the overlap is smaller than *min_intersection_area*, or when trimming would
    have eaten one of them whole (upstream's ``area < 1.0`` reject at ``deintersect.py:65``: a
    line whose neighbour swallowed it is better left as it was than shaved to a thread).
    """
    if first.is_empty or second.is_empty or not first.intersects(second):
        return first, second, False

    intersection = first.intersection(second)
    if intersection.is_empty or intersection.area < float(min_intersection_area):
        return first, second, False

    # Buffered by half the gap on each side, so the two end up `gap` apart instead of touching
    # along the cut -- which is what stops the next pass finding the same overlap again.
    shrink = intersection.buffer(float(gap) / 2.0)
    trimmed_first = polish_polygon(_largest_polygon(first.difference(shrink)), simplify_tolerance)
    trimmed_second = polish_polygon(_largest_polygon(second.difference(shrink)), simplify_tolerance)

    if trimmed_first.is_empty or trimmed_second.is_empty:
        return first, second, False
    if trimmed_first.area < 1.0 or trimmed_second.area < 1.0:
        return first, second, False
    return trimmed_first, trimmed_second, True


def deintersect_polygons(
    contours: Contour | Iterable[Contour],
    *,
    gap: float = DEFAULT_GAP,
    min_intersection_area: float = DEFAULT_MIN_INTERSECTION_AREA,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> list[Contour]:
    """Every polygon in *contours*, trimmed of the area its neighbours share.

    One contour per contour given, in the same order, so a caller can hand a page's regions to it
    and put the answers back where they came from.  A polygon a neighbour consumed comes back as
    the empty ``(0, 1, 2)`` contour rather than being dropped from the list -- upstream appends an
    empty point list in its place (``deintersect.py:116``), and the caller building regions from
    the result is the one that knows whether an empty region is written, skipped or merged.

    The pairwise sweep is upstream's: ``max_passes`` sweeps at most, a sweep that changed nothing
    ends them, and a broad-phase ``buffer(gap + 2.0)`` of the first polygon screens the second out
    before the (expensive) exact intersection is computed.
    """
    given = list(_iter_contours(contours))
    if not given:
        return []

    working = [polygon_from_points(contour) for contour in given]
    passes = max(1, int(max_passes))
    for _ in range(passes):
        changed = 0
        for index, polygon in enumerate(working):
            if polygon.is_empty:
                continue
            screen = polygon.buffer(float(gap) + 2.0)
            for other in range(index + 1, len(working)):
                if working[other].is_empty or not screen.intersects(working[other]):
                    continue
                trimmed, trimmed_other, did = separate_intersecting_pair(
                    polygon,
                    working[other],
                    gap=gap,
                    min_intersection_area=min_intersection_area,
                    simplify_tolerance=simplify_tolerance,
                )
                if not did:
                    continue
                working[index] = polygon = trimmed
                working[other] = trimmed_other
                screen = polygon.buffer(float(gap) + 2.0)
                changed += 1
        if not changed:
            break

    results: list[Contour] = []
    for polygon in working:
        biggest = _largest_polygon(polygon)
        if biggest.is_empty:
            results.append(np.empty((0, 1, 2), dtype=np.int32))
            continue
        ring = list(biggest.exterior.coords)
        # shapely closes every ring it returns; this package's contours are open vertex lists.
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        results.append(_as_cv_contour(np.asarray(ring, dtype=np.int32)))
    return results
