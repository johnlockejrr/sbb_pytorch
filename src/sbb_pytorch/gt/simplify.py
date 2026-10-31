"""Polygon simplification -- the Douglas-Peucker pair the sibling projects share.

Ported from ``oxygraphos-ocr``'s ``surya-seg/surya_seg/lib/simplify_geometry.py``, itself the
refactor of ``regnetx-seg/regnetx_seg/lib/simplify_geometry.py`` -- the older split API
(``simplify_open_polyline_xy`` / ``simplify_closed_polygon_xy``) beside a ``Segmentation``-aware
wrapper this package has no equivalent of.  The two list-taking functions here are that port
unchanged; :func:`simplify_contour` and :func:`simplify_contours` are the shell that speaks *this*
package's polygon type, and :class:`PageGeometry` is the pair of switches that decides what a
page's geometry looks like before it is rasterised or written.

Why a page needs it at all: a PageXML ``@points`` list is a *rasterised* outline -- every pixel of
a straight edge is a vertex of its own -- and ``cv2.findContours`` with ``CHAIN_APPROX_SIMPLE``
(the method :func:`~sbb_pytorch.gt.geometry.contours_of_image` uses) is a storage trick rather
than a simplification, dropping repeats only inside the straight runs a contour happens to have.
``sbb infer`` on a page of ``BL_Or_1443_300``'s shape therefore hands the writer a contour of
thousands of vertices per predicted region, and the PageXML that comes out is the same fact from
the file-size side.

The epsilon is rfdetr-seg's rule rather than surya-seg's fixed 1.5 px: ``max(base, 0.002 *
perimeter)``, so a full-resolution page gets a tolerance scaled to its own size where a 60x60 crop
keeps a small one.  An epsilon of ``0`` means *off* -- every vertex kept -- which is what every
output this package has produced so far looked like, and what the config default still is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Final

import cv2
import numpy as np

from ..config.models import SBBConfig
from .deintersect import deintersect_polygons
from .geometry import _as_cv_contour, _iter_contours
from .parser import Page, Region

__all__ = [
    "DEFAULT_SIMPLIFY_EPSILON_BASE",
    "DEFAULT_SIMPLIFY_EPSILON_FRACTION",
    "PageGeometry",
    "collapse_axis_aligned_redundant",
    "simplify_contour",
    "simplify_contours",
    "simplify_polygon",
    "simplify_polyline",
]

#: The Douglas-Peucker tolerance, in pixels, that the ``simplify_*`` functions default to --
#: rfdetr-seg's ``base`` (``rfdetr-seg/rfdetr_seg/engine/infer.py:67``), the one value among the
#: siblings that is not page-size dependent: surya-seg fixes 1.5 px, dfine-seg passes the caller's
#: tolerance through untouched, docufcn-seg hardcodes 4.0 and regnetx-seg floors at 0.5.
DEFAULT_SIMPLIFY_EPSILON_BASE: Final = 2.0

#: The part of a contour's *perimeter* the tolerance grows to on a big outline, same source
#: (``infer.py:72``: ``max(base, 0.002 * arcLength)``).  A fixed pixel tolerance that is right on a
#: 60x60 crop rounds a full-page outline away entirely, and one that is right on a page leaves a
#: crop's corners jagged; 0.2 % of the perimeter is the compromise the sibling that measures it
#: settled on.  Applied by :func:`simplify_contour` alone -- the two list ports take their epsilon
#: exactly, so the ported tests test the port and not this policy.
DEFAULT_SIMPLIFY_EPSILON_FRACTION: Final = 0.002


def _dedupe_consecutive(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """*points* as ``int`` pairs, with each run of identical neighbours collapsed to one.

    The first step of both ports (``surya-seg/.../simplify_geometry.py:13``).  A rasterised ring
    repeats its first vertex to close itself and repeats corners where two pixel runs meet, and
    ``cv2.approxPolyDP`` counts those repeats as shape.
    """
    out: list[tuple[int, int]] = []
    for point in points:
        pair = (int(point[0]), int(point[1]))
        if not out or pair != out[-1]:
            out.append(pair)
    return out


def collapse_axis_aligned_redundant(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop the middle vertex of every straight horizontal or vertical run.

    Both siblings share this function byte for byte (``surya-seg/.../simplify_geometry.py:23``).
    It is the answer to the specific shape of a PageXML ``@points`` list, which is a rasterised
    outline: a 900-pixel straight edge arrives as 900 vertices, and ``approxPolyDP`` with a small
    epsilon keeps most of them because each is a *documented* corner.  Dropping the middles first
    leaves the two ends of the run, which is what the annotation means.
    """
    pts = _dedupe_consecutive(points)
    if len(pts) < 3:
        return pts
    out: list[tuple[int, int]] = [pts[0]]
    for index in range(1, len(pts) - 1):
        x0, y0 = out[-1]
        x1, y1 = pts[index]
        x2, y2 = pts[index + 1]
        if y0 == y1 == y2 or x0 == x1 == x2:
            continue  # The middle of a straight run: not a corner, whatever the file says.
        out.append(pts[index])
    out.append(pts[-1])
    return out


def simplify_polygon(
    polygon: Iterable[tuple[int, int]],
    epsilon: float,
    *,
    min_points: int = 3,
) -> list[tuple[int, int]]:
    """*polygon* as a simplified **closed** ring: axis collapse, then ``cv2.approxPolyDP``.

    surya-seg's ``simplify_polygon`` (``surya-seg/.../simplify_geometry.py:40``) unchanged, and
    regnetx-seg's ``simplify_closed_polygon_xy`` before it.  Three behaviours are the port's and
    worth knowing before reading the body: ``epsilon <= 0`` returns the vertices as given (the
    off switch), a duplicated closing vertex is stripped before simplifying and never put back,
    and a result that fell below *min_points* is **thrown away** in favour of the input --
    simplifying must never be able to delete a shape, only shorten its vertex list.
    """
    if not (pts := _dedupe_consecutive(polygon)):
        return []
    if epsilon <= 0:
        return list(pts)
    ring = pts[:-1] if len(pts) >= 2 and pts[0] == pts[-1] else pts
    if len(ring) < min_points:
        return list(pts)
    ring = collapse_axis_aligned_redundant(ring)
    if len(ring) < min_points:
        return list(pts)
    approx = cv2.approxPolyDP(
        np.asarray(ring, dtype=np.float32).reshape(-1, 1, 2), float(epsilon), closed=True
    )
    out = _dedupe_consecutive((pair[0], pair[1]) for pair in approx.reshape(-1, 2))
    return out if len(out) >= min_points else list(pts)


def simplify_polyline(
    points: Iterable[tuple[int, int]],
    epsilon: float,
    *,
    min_points: int = 2,
) -> list[tuple[int, int]]:
    """*points* as a simplified **open** polyline -- a baseline, not a region outline.

    regnetx-seg's ``simplify_open_polyline_xy`` (``regnetx-seg/.../simplify_geometry.py:54``);
    surya-seg folded it away when it merged the two entry points, which is why the closed-only port
    above cannot answer for a ``TextLine/@points`` baseline.  Same three behaviours as the closed
    port (off on ``epsilon <= 0``, revert on degeneracy, and a polyline of two points or fewer is
    returned collapsed rather than simplified, because the two ends *are* the line).
    """
    if not (pts := _dedupe_consecutive(points)):
        return []
    if epsilon <= 0:
        return list(pts)
    collapsed = collapse_axis_aligned_redundant(pts)
    if len(collapsed) <= min_points:
        return collapsed
    approx = cv2.approxPolyDP(
        np.asarray(collapsed, dtype=np.float32).reshape(-1, 1, 2), float(epsilon), closed=False
    )
    out = _dedupe_consecutive((pair[0], pair[1]) for pair in approx.reshape(-1, 2))
    if len(out) < min_points:
        return [collapsed[0], collapsed[-1]]
    return out


def simplify_contour(
    contour: np.ndarray,
    epsilon: float = DEFAULT_SIMPLIFY_EPSILON_BASE,
    *,
    closed: bool = True,
) -> np.ndarray:
    """*contour* simplified, in the shape every other contour in this package has.

    The adapter between the two ports above (which speak ``list[tuple[int, int]]``) and
    :data:`sbb_pytorch.gt.parser.Contour` (an ``(N, 2)``/``(N, 1, 2)`` ``int32`` array, normalised
    by :func:`~sbb_pytorch.gt.geometry._as_cv_contour`, which is what ``contour_area``,
    ``mask_of`` and ``writer.points_attribute`` all accept).  ``closed`` picks the port: ``True``
    for a region outline, ``False`` for a baseline.

    *epsilon* is the **base** tolerance, and the effective one is
    ``max(epsilon, DEFAULT_SIMPLIFY_EPSILON_FRACTION * perimeter)`` -- rfdetr-seg's rule, applied
    here and only here so the ports keep their exact-epsilon meaning.  ``epsilon == 0`` and a
    contour with fewer than three vertices both come back as the contour itself, normalised:
    simplification off is a *no*, not a deletion.
    """
    array = np.asarray(contour)
    if array.size == 0:
        return _as_cv_contour(array)
    points = [(int(x), int(y)) for x, y in array.reshape(-1, 2)]
    if epsilon <= 0 or len(points) < 3:
        return _as_cv_contour(array)
    normalised = _as_cv_contour(array)
    tolerance = max(
        float(epsilon), DEFAULT_SIMPLIFY_EPSILON_FRACTION * float(cv2.arcLength(normalised, closed))
    )
    simplified = (
        simplify_polygon(points, tolerance) if closed else simplify_polyline(points, tolerance)
    )
    return _as_cv_contour(np.asarray(simplified, dtype=np.int32)) if simplified else normalised


def simplify_contours(
    contours: Iterable[np.ndarray] | np.ndarray,
    epsilon: float = DEFAULT_SIMPLIFY_EPSILON_BASE,
    *,
    closed: bool = True,
) -> list[np.ndarray]:
    """Every contour in *contours* simplified -- the batch form of :func:`simplify_contour`.

    Takes one contour, a list of them or a stacked ``(K, N, 2)`` batch through
    :func:`~sbb_pytorch.gt.geometry._iter_contours`, which is how the rest of ``gt`` accepts a
    contour argument, so ``contours_of_image`` output and a generator's per-class list go through
    the same call.  The list it returns is the same length as the input, in the same order.
    """
    return [
        simplify_contour(contour, epsilon, closed=closed) for contour in _iter_contours(contours)
    ]


@dataclass(frozen=True, slots=True)
class PageGeometry:
    """How a page's polygons are prepared before anything reads, rasterises or writes them.

    The three switches in one place, because three of the five ``gt`` entry points need them
    together -- :func:`~sbb_pytorch.gt.render.render_page`,
    :func:`~sbb_pytorch.gt.generator.build_label_map` and the two mask-to-region builders in the
    CLI -- and each of them reads them off a *different* config object (an
    :class:`~sbb_pytorch.config.models.SBBConfig` for ``sbb infer``, a
    :class:`~sbb_pytorch.gt.generator.GroundTruthConfig` for ``sbb gt generate``, and nothing at
    all for ``sbb gt visualize``, which takes no config).  Both defaults mean *off*: every page
    this package has produced looked like ``PageGeometry()``.

    ``deintersect`` needs ``shapely`` (see :mod:`sbb_pytorch.gt.deintersect`); asking for it
    without the package raises rather than doing nothing.
    """

    #: Douglas-Peucker base tolerance in pixels; ``0`` keeps every vertex the page or model gave.
    simplify_epsilon: float = 0.0
    #: Trim the area two overlapping regions share off both, instead of letting one paint over it.
    deintersect: bool = False
    #: How far apart the trimmed neighbours are left, in pixels.
    intersection_gap: float = 1.5

    @classmethod
    def from_sbb(cls, config: SBBConfig) -> PageGeometry:
        """The geometry an :class:`~sbb_pytorch.config.models.SBBConfig` asks for."""
        return cls(
            simplify_epsilon=config.simplify_epsilon,
            deintersect=config.deintersect,
            intersection_gap=config.intersection_gap,
        )

    @property
    def is_off(self) -> bool:
        """Whether preparing a page under these settings would change nothing about it."""
        return self.simplify_epsilon <= 0 and not self.deintersect

    def prepare(self, page: Page) -> Page:
        """*page* with every region's geometry prepared under these settings.

        Nothing is mutated: the page, every region in it and every region nested inside one are
        rebuilt with :func:`dataclasses.replace`, so a caller that still holds the parsed page --
        the generator holds it while it writes the label *and* copies the source image beside it --
        keeps seeing the object it was given, and ``Region.parent`` links point at the prepared
        copies rather than the originals they were parsed into.
        """
        if self.is_off:
            return page
        copies = {id(region): replace(region, children=[], parent=None) for region in page.regions}
        for region in page.regions:
            prepared = copies[id(region)]
            prepared.contour = simplify_contour(region.contour, self.simplify_epsilon, closed=True)
            if region.baseline is not None:
                prepared.baseline = simplify_contour(
                    region.baseline, self.simplify_epsilon, closed=False
                )
        if self.deintersect:
            self._separate(page.regions, copies)
        for region in page.regions:
            copies[id(region)].children = [
                copies[id(child)] for child in region.children if id(child) in copies
            ]
        return replace(
            page,
            regions=[copies[id(region)] for region in page.regions],
            printspace_contour=(
                None
                if page.printspace_contour is None
                else simplify_contour(page.printspace_contour, self.simplify_epsilon, closed=True)
            ),
        )

    def _separate(self, regions: list[Region], copies: dict[int, Region]) -> None:
        """Trim shared area inside each region's children, then across the page's own regions.

        Two passes, in this order, because the first one changes what the second finds: the lines
        *inside* one region are separated from each other first (the case upstream wrote this for,
        ``deintersect_line_polygons`` trimming the lines of one ``TextRegion``), and only then are
        the page-level regions trimmed against each other -- a page whose ring web overlaps a
        fragment would otherwise be trimmed against the *un*trimmed fragment.

        Runs on the prepared copies, so it trims the simplified geometry the writer will write
        rather than the raw outline the parser read.
        """
        for region in regions:
            children = [copies[id(child)] for child in region.children if id(child) in copies]
            if len(children) > 1:
                self._apply(children)
        roots = [
            copies[id(region)]
            for region in regions
            if region.parent is None and id(region) in copies
        ]
        if len(roots) > 1:
            self._apply(roots)

    def _apply(self, group: list[Region]) -> None:
        """De-intersect one group of regions and put each contour back on the region it came from."""
        trimmed = deintersect_polygons(
            [region.contour for region in group], gap=self.intersection_gap
        )
        for region, contour in zip(group, trimmed, strict=True):
            if contour.shape[0] >= 3:  # A region a neighbour swallowed keeps its own outline.
                region.contour = contour
