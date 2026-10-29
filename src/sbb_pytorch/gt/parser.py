"""ALTO/PageXML parsing.

This module is the single place in the package that knows the page-description
schema.  It replaces the five near-identical XML readers of the legacy
``gt_gen_utils`` module (``read_xml``, ``get_textline_contours_for_visualization``,
``get_layout_contours_for_visualization`` and the two inline blocks inside
``get_images_of_ground_truth``), which together accounted for roughly 1000
duplicated lines.

Two coordinate encodings exist in the wild and both are supported, exactly as
the legacy code supported them:

``<Coords points="x1,y1 x2,y2 ..."/>``
    Preferred whenever the element carries attributes.
``<Point x="..." y="..."/>`` children
    Used when ``Coords`` is present but attribute-less (or absent).

The legacy code derived the namespace prefix with ``alltags[0].split('}')[0] + '}'``,
which yields the nonsense prefix ``"PcGts}"`` for a namespace-less document and
then leaves ``imageHeight``/``imageWidth`` unbound.  :func:`_namespace_prefix`
derives it from the ``{uri}tag`` convention instead and falls back to no prefix.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np

from .geometry import contour_bbox

__all__ = [
    "DEFAULT_XML_ENCODING",
    "GRAPHIC_REGION_TYPES",
    "LEGACY_INACTIVE_TEXT_TYPES",
    "TEXT_REGION_TYPES",
    "Contour",
    "Page",
    "PageXMLError",
    "Region",
    "RegionKind",
    "discover_xml_files",
    "normalise_text_type",
    "parse_pagexml",
]

#: Encoding the legacy tool used for every PageXML file.
DEFAULT_XML_ENCODING: Final = "iso-8859-5"

#: A polygon as consumed by OpenCV: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
Contour = np.ndarray


class PageXMLError(ValueError):
    """Raised when a file cannot be interpreted as an ALTO/PageXML document."""


class RegionKind(str, Enum):
    """Region kinds, keyed by the lower-cased XML local name."""

    TEXT = "textregion"
    GRAPHIC = "graphicregion"
    IMAGE = "imageregion"
    SEPARATOR = "separatorregion"
    TABLE = "tableregion"
    NOISE = "noiseregion"
    TEXTLINE = "textline"
    WORD = "word"
    GLYPH = "glyph"
    PRINTSPACE = "printspace"
    BORDER = "border"


#: Canonical PageXML ``TextRegion/@type`` values.
TEXT_REGION_TYPES: Final[tuple[str, ...]] = (
    "paragraph",
    "header",
    "heading",
    "footnote",
    "footnote-continued",
    "drop-capital",
    "signature-mark",
    "catch-word",
    "page-number",
    "marginalia",
)

#: Canonical PageXML ``GraphicRegion/@type`` values.
GRAPHIC_REGION_TYPES: Final[tuple[str, ...]] = (
    "decoration",
    "handwritten-annotation",
    "stamp",
    "signature",
)

#: ``read_xml`` has its ``catch-word`` and ``page-number`` branches commented out,
#: so regions of those types were routed to ``paragraph``.  The layout visualiser
#: does collect them under their own type.  The distinction is preserved here
#: rather than silently unified.
LEGACY_INACTIVE_TEXT_TYPES: Final[frozenset[str]] = frozenset({"catch-word", "page-number"})

_REGION_TAGS: Final[dict[str, RegionKind]] = {
    # The legacy code matched the pairs ``'}TextRegion'`` / ``'}Textregion'`` (and the
    # same for every other kind) because both spellings occur in real files; lower-casing
    # the local name in :func:`_local_name` covers both with one key per kind.
    "textregion": RegionKind.TEXT,
    "graphicregion": RegionKind.GRAPHIC,
    "imageregion": RegionKind.IMAGE,
    "separatorregion": RegionKind.SEPARATOR,
    "tableregion": RegionKind.TABLE,
    "noiseregion": RegionKind.NOISE,
    "textline": RegionKind.TEXTLINE,
    "word": RegionKind.WORD,
    "glyph": RegionKind.GLYPH,
    "printspace": RegionKind.PRINTSPACE,
    "border": RegionKind.BORDER,
}


def _local_name(tag: str) -> str:
    """Return the lower-cased local name of an ElementTree tag string.

    ``"{http://...}TextRegion"`` -> ``"textregion"``; ``"TextRegion"`` -> ``"textregion"``.
    """
    return tag.rsplit("}", 1)[-1].lower()


def _namespace_prefix(root: ET.Element) -> str:
    """Return the ``"{uri}"`` prefix of *root*, or ``""`` when the document has none."""
    return root.tag.split("}", 1)[0] + "}" if "}" in root.tag else ""


def _parse_coords_points(raw: str) -> Contour | None:
    """Parse a ``Coords/@points`` string into an ``(N, 2)`` int32 contour."""
    pairs = [chunk for chunk in raw.replace(",", " ").split() if chunk]
    if len(pairs) < 2:
        return None
    if len(pairs) % 2:
        pairs = pairs[:-1]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def _contour_of(element: ET.Element, prefix: str) -> Contour | None:
    """Extract the outline of one region element, honouring both coordinate encodings.

    Mirrors the legacy traversal order: a ``Coords`` element with attributes wins
    and stops the search; otherwise the ``Point`` children are accumulated.
    """
    if element.tag != f"{prefix}PcGts":
        coords = element.find(f"{prefix}Coords") if element.tag != f"{prefix}Coords" else element
        if coords is not None and coords.get("points"):
            contour = _parse_coords_points(coords.get("points"))
            if contour is not None:
                return contour

    points = [
        (child.get("x"), child.get("y"))
        for child in element.iter(f"{prefix}Point")
        if child.get("x") is not None and child.get("y") is not None
    ]
    if len(points) < 3:
        return None
    return np.asarray([[int(float(x)), int(float(y))] for x, y in points], dtype=np.int32)


def _element_contour(element: ET.Element, prefix: str) -> Contour | None:
    """Contour of *element*, accepting either a ``Coords`` attribute or ``Point`` children."""
    if element.tag == f"{prefix}Coords" and element.get("points"):
        return _parse_coords_points(element.get("points"))
    direct = element.find(f"{prefix}Coords")
    if direct is not None and direct.get("points"):
        return _parse_coords_points(direct.get("points"))
    return _contour_of(element, prefix)


def normalise_text_type(raw_type: str, configured_types: list[str] | None = None) -> str:
    """Map a raw ``TextRegion/@type`` onto the label key the pipeline uses.

    Two legacy rules are encoded here, previously duplicated at five call sites:

    * a configured ``rest_as_<type>`` key renames that type, e.g.
      ``rest_as_paragraph`` -> ``paragraph``;
    * ``read_xml`` had ``catch-word``/``page-number`` disabled, routing those
      regions to ``paragraph`` (see :data:`LEGACY_INACTIVE_TEXT_TYPES`).  This is
      applied only when *configured_types* is given, which is how the caller
      signals it wants the label-generation routing rather than the faithful one.
    """
    if raw_type.startswith("rest_as_"):
        return raw_type[len("rest_as_") :]
    if configured_types is None:
        return raw_type
    declared = set(configured_types)
    # A ``rest_as_paragraph`` key also declares the bare name ``paragraph``.
    folded = declared | {
        name[len("rest_as_") :] for name in declared if name.startswith("rest_as_")
    }
    if raw_type in LEGACY_INACTIVE_TEXT_TYPES and raw_type not in folded and "paragraph" in folded:
        return "paragraph"
    return raw_type


@dataclass(slots=True)
class Region:
    """One labelled region of a page.

    Attributes
    ----------
    kind:
        The region kind this region was parsed from.
    type_name:
        ``@type`` attribute as written in the file, or ``""`` when absent.
    region_id:
        ``@ID`` attribute, or ``""`` when absent.
    contour:
        ``(N, 2)`` int32 array of ``(x, y)`` pixel coordinates.
    text:
        ``TextEquiv/Unicode`` content, or ``None``.
    """

    kind: RegionKind
    type_name: str
    region_id: str
    contour: Contour
    text: str | None = None


@dataclass(slots=True)
class Page:
    """A parsed page.

    Attributes
    ----------
    path:
        File the page was parsed from.
    file_name:
        Stem of *path* — the key the rest of the pipeline names outputs by.
    height, width:
        ``Page/@imageHeight`` and ``Page/@imageWidth`` in pixels.
    columns_width:
        ``Metadata/Comments`` ``num_col`` value, when the file carries one.
    regions:
        Every labelled region found on the page.
    printspace_contour / printspace_bbox:
        Print space outline and its ``[x, y, w, h]`` bounding box, if present.
    region_index:
        ``@ID`` -> ``RegionRefIndexed/@index``, i.e. the reading order.
    """

    path: Path
    file_name: str
    height: int
    width: int
    columns_width: int | None = None
    regions: list[Region] = field(default_factory=list)
    printspace_contour: Contour | None = None
    printspace_bbox: tuple[int, int, int, int] | None = None
    region_index: dict[str, int] = field(default_factory=dict)

    def contours_for(self, kind: RegionKind, type_name: str | None = None) -> list[Contour]:
        """Contours of every region of *kind*, optionally restricted to *type_name*."""
        return [
            region.contour
            for region in self.regions
            if region.kind is kind
            and (type_name is None or region.type_name == type_name)
        ]

    def grouped(self, kind: RegionKind) -> dict[str, list[Contour]]:
        """``type_name -> [contour, ...]`` mapping for all regions of *kind*."""
        grouped: dict[str, list[Contour]] = {}
        for region in self.regions:
            if region.kind is kind:
                grouped.setdefault(region.type_name, []).append(region.contour)
        return grouped

    def ordered_ids(self, kind: RegionKind, type_name: str | None = None) -> list[str]:
        """Region ids of *kind* in document (reading) order, empty when unparsable."""
        return [
            region.region_id
            for region in self.regions
            if region.kind is kind
            and (type_name is None or region.type_name == type_name)
            and region.region_id
        ]


def parse_pagexml(xml_file: str | Path) -> Page:
    """Parse an ALTO/PageXML file into a :class:`Page`.

    Parameters
    ----------
    xml_file:
        Path to a ``.xml`` / ``.XML`` page-description file.

    Raises
    ------
    PageXMLError
        The file is unreadable, has no ``Page`` element, or has no usable geometry.
    """
    path = Path(xml_file)
    try:
        tree = ET.parse(path, parser=ET.XMLParser(encoding=DEFAULT_XML_ENCODING))
    except (ET.ParseError, OSError, LookupError) as exc:  # LookupError: unknown codec
        raise PageXMLError(f"cannot parse {path}: {exc}") from exc

    root = tree.getroot()
    if _local_name(root.tag) != "pcgts":
        raise PageXMLError(f"{path}: expected a <PcGts> root, found <{root.tag}>")
    prefix = _namespace_prefix(root)

    page_element = root.find(f"{prefix}Page")
    if page_element is None:
        raise PageXMLError(f"{path}: no <Page> element")
    try:
        height = int(float(page_element.get("imageHeight", "")))
        width = int(float(page_element.get("imageWidth", "")))
    except ValueError as exc:
        raise PageXMLError(f"{path}: <Page> lacks imageHeight/imageWidth") from exc

    return Page(
        path=path,
        file_name=path.stem,
        height=height,
        width=width,
        columns_width=_columns_width(root, prefix),
        regions=list(_iter_regions(root, prefix)),
        **_printspace(root, prefix, height, width),
        region_index=_region_index(root, prefix),
    )


def _columns_width(root: ET.Element, prefix: str) -> int | None:
    """Read ``Metadata/Comments`` for the ``num_colNNNN`` marker the tool writes."""
    metadata = root.find(f"{prefix}Metadata")
    if metadata is None:
        return None
    for child in metadata:
        if _local_name(child.tag) != "comments" or not child.text:
            continue
        _, _, digits = child.text.partition("num_col")
        digits = "".join(char for char in digits if char.isdigit())
        if digits:
            return int(digits)
    return None


def _region_index(root: ET.Element, prefix: str) -> dict[str, int]:
    """``regionRef -> index`` from ``RegionRefIndexed`` elements.

    Built as a dict rather than the legacy pair of parallel lists, whose consumers
    did ``list.index(...)`` inside a comprehension (quadratic per page).
    """
    index: dict[str, int] = {}
    for ref in root.iter(f"{prefix}RegionRefIndexed"):
        region_ref, raw_index = ref.get("regionRef"), ref.get("index")
        if region_ref is None or raw_index is None:
            continue
        try:
            index[region_ref] = int(float(raw_index))
        except ValueError:
            continue
    return index


def _iter_regions(root: ET.Element, prefix: str) -> Iterator[Region]:
    """Yield every labelled region of the page, whatever its kind."""
    for element in root.iter():
        kind = _REGION_TAGS.get(_local_name(element.tag))
        if kind is None:
            continue
        contour = _element_contour(element, prefix)
        if contour is None:
            continue
        yield Region(
            kind=kind,
            type_name=element.get("type", ""),
            # PageXML spells this attribute `id`, ALTO spells it `ID`; both are read here.
            region_id=element.get("ID") or element.get("id") or "",
            contour=contour,
            text=_text_of(element, prefix),
        )


def _text_of(element: ET.Element, prefix: str) -> str | None:
    """``TextEquiv/Unicode`` text of *element*, if any."""
    equiv = element.find(f"{prefix}TextEquiv")
    if equiv is None:
        return None
    unicode_node = equiv.find(f"{prefix}Unicode")
    return unicode_node.text if unicode_node is not None else None


def _printspace(root: ET.Element, prefix: str, height: int, width: int) -> dict[str, object]:
    """Print space contour and bounding box (``Border`` is accepted as a fallback)."""
    contours = [
        region.contour
        for element in root.iter()
        if _local_name(element.tag) in {"printspace", "border"}
        for region in [_element_region(element, prefix)]
        if region is not None
    ]
    if not contours:
        return {"printspace_contour": None, "printspace_bbox": None}
    return {
        "printspace_contour": contours[0],
        "printspace_bbox": contour_bbox(contours[0], (height, width)),
    }


def _element_region(element: ET.Element, prefix: str) -> Region | None:
    """Wrap a bare ``PrintSpace``/``Border`` element as a :class:`Region`."""
    contour = _element_contour(element, prefix)
    if contour is None:
        return None
    return Region(
        kind=RegionKind(_local_name(element.tag)),
        type_name=element.get("type", ""),
        # Same `id`/`ID` spelling as the region reader above.
        region_id=element.get("ID") or element.get("id") or "",
        contour=contour,
    )


def discover_xml_files(directory: str | Path) -> list[Path]:
    """All XML files directly inside *directory*, sorted, extensions matched case-first.

    Replaces ``get_content_of_dir`` and the ``dir_imgs`` listings it was paired with: the
    legacy version called ``os.listdir`` once per loop iteration (once per file, in the
    ``pagexml2label`` path) and treated subdirectories as files, so a nested directory
    named ``foo.xml`` was handed to the parser.  Files only, ``.xml``/``.XML``, walked once.
    """
    base = Path(directory)
    return sorted(
        entry for entry in base.iterdir() if entry.is_file() and entry.suffix.lower() == ".xml"
    )


def image_for(page: Page, directory: str | Path) -> Path | None:
    """Locate the raster image matching *page* in *directory*.

    Any extension is accepted, matching the legacy
    ``find_format_of_given_filename_in_dir``, which raised ``IndexError`` instead
    of returning ``None``.
    """
    base = Path(directory)
    if not base.is_dir():
        return None
    for entry in sorted(base.iterdir()):
        if entry.is_file() and entry.stem == page.file_name:
            return entry
    return None
