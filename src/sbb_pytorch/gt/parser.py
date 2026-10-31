"""ALTO/PageXML parsing.

This module reads a page.  The *names* it reads -- the element per region kind, the two ``@type``
enumerations, the ``rest_as_`` legacy rule -- live in :mod:`.vocabulary`, which the writer consults
too, so a name is decided once and read and written the same way round; what is left here is the
XML itself.  Reading replaces the five near-identical readers of the legacy
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

The byte encoding is read the same way round: the legacy readers passed
``ET.XMLParser(encoding='iso-8859-5')``, and an explicit ``encoding`` *overrides* the
document's own declaration, so a UTF-8-authored page arrived as mojibake.
:func:`_parse_tree` honours the declaration -- UTF-8 by the XML spec's default when the
file omits one -- and keeps :data:`DEFAULT_XML_ENCODING` as the fallback that still
reads the legacy corpus.

PageXML is a tree and this module now reads it as one.  A ``TextRegion`` holds
``TextLine``\\ s, which hold ``Word``\\ s, which hold ``Glyph``\\ s, and the legacy readers
flattened all four levels into one list, so the answer to "which lines make up this block?"
was gone before the pipeline started.  :class:`Region` carries ``children``/``parent`` and
:func:`_iter_regions` fills them by walking the elements in order; :attr:`Page.regions` stays
the flat list every existing consumer expects, and :meth:`Page.root_regions` /
:meth:`Page.walk` give the tree back.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path
from typing import Final

import numpy as np

from ..exceptions import PageXMLError
from .geometry import contour_bbox
from .vocabulary import (
    GRAPHIC_REGION_TYPES,
    LEGACY_INACTIVE_TEXT_TYPES,
    REGION_TAGS,
    TEXT_REGION_TYPES,
    RegionKind,
    normalise_text_type,
)

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

#: Encoding the legacy tool used for every PageXML file, and the fallback here when a
#: document's own declaration cannot decode it (see :func:`_parse_tree`).
DEFAULT_XML_ENCODING: Final = "iso-8859-5"

#: A polygon as consumed by OpenCV: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
Contour = np.ndarray


#: The element-per-kind table, :data:`.vocabulary.REGION_TAGS` under its original private name --
#: kept as an alias because every reader of this module reached for it by name, and because the
#: writer used to keep a second copy of the same mapping (``writer._REGION_ELEMENTS``) that this one
#: now stands for.
_REGION_TAGS: Final[dict[str, RegionKind]] = REGION_TAGS

#: Which region kinds the schema allows *inside* each kind, from the content models of
#: ``pagecontent/2019-07-15``: ``TextRegionType`` nests regions and ``TextLine``\\ s,
#: ``TextLineType`` nests ``Word``\\ s and ``WordType`` nests ``Glyph``\\ s; every other region
#: type has no region children at all, and neither has ``PrintSpace`` nor ``Border``, which are
#: not members of the region choice but separate ``Page`` children.
#:
#: A child found somewhere else is kept -- losing annotations to a technicality is worse than
#: recording an odd nesting -- but as a page-level *orphan* rather than as that region's child,
#: so the nesting a file actually contains is never invented or reassigned.
_ALLOWED_CHILDREN: Final[dict[RegionKind, frozenset[RegionKind]]] = {
    RegionKind.TEXT: frozenset(
        {
            RegionKind.TEXT,
            RegionKind.IMAGE,
            RegionKind.GRAPHIC,
            RegionKind.TABLE,
            RegionKind.SEPARATOR,
            RegionKind.NOISE,
            RegionKind.TEXTLINE,
            RegionKind.WORD,
        }
    ),
    RegionKind.TEXTLINE: frozenset({RegionKind.WORD}),
    RegionKind.WORD: frozenset({RegionKind.GLYPH}),
    RegionKind.GLYPH: frozenset(),
    RegionKind.IMAGE: frozenset(),
    RegionKind.GRAPHIC: frozenset(),
    RegionKind.SEPARATOR: frozenset(),
    RegionKind.TABLE: frozenset(),
    RegionKind.NOISE: frozenset(),
    RegionKind.PRINTSPACE: frozenset(),
    RegionKind.BORDER: frozenset(),
}


def _local_name(tag: str) -> str:
    """Return the lower-cased local name of an ElementTree tag string.

    ``"{http://...}TextRegion"`` -> ``"textregion"``; ``"TextRegion"`` -> ``"textregion"``.
    """
    return tag.rsplit("}", 1)[-1].lower()


def _namespace_prefix(root: ET.Element) -> str:
    """Return the ``"{uri}"`` prefix of *root*, or ``""`` when the document has none."""
    return root.tag.split("}", 1)[0] + "}" if "}" in root.tag else ""


def _coordinate(value: str) -> int:
    """One coordinate attribute as an ``int``, via ``int(float(value))``.

    The legacy ``<Point>`` reader wrote ``int(float(vv.attrib['x']))``
    (``gt_gen_utils.py:197``), which truncates the fractional coordinates modern tools
    emit, while its ``@points`` twin eight lines above wrote ``int(x.split(',')[0])``
    (``gt_gen_utils.py:191``) and so *raised* on the same ``"1.5"`` -- taking the whole
    page with it.  One shared conversion removes the asymmetry in favour of the reading
    that never loses a page.
    """
    return int(float(value))


def _parse_coords_points(raw: str) -> Contour | None:
    """Parse a ``Coords/@points`` string into an ``(N, 2)`` int32 contour.

    Every value goes through :func:`_coordinate`, so a fractional coordinate is
    truncated rather than raising, exactly as the ``<Point>`` encoding has always been.
    """
    pairs = [chunk for chunk in raw.replace(",", " ").split() if chunk]
    if len(pairs) < 2:
        return None
    if len(pairs) % 2:
        pairs = pairs[:-1]  # A dangling value is dropped, never padded with a zero.
    return np.asarray([_coordinate(value) for value in pairs], dtype=np.int32).reshape(-1, 2)


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
    return np.asarray([[_coordinate(x), _coordinate(y)] for x, y in points], dtype=np.int32)


def _element_contour(element: ET.Element, prefix: str) -> Contour | None:
    """Contour of *element*, accepting either a ``Coords`` attribute or ``Point`` children."""
    if element.tag == f"{prefix}Coords" and element.get("points"):
        return _parse_coords_points(element.get("points"))
    direct = element.find(f"{prefix}Coords")
    if direct is not None and direct.get("points"):
        return _parse_coords_points(direct.get("points"))
    return _contour_of(element, prefix)


@dataclass(slots=True)
class Region:
    """One labelled region of a page, plus the regions nested inside it.

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
    custom:
        ``@custom`` as written -- the escriptorium/Aletheia dialect keeps the region's class
        there (``structure {type:paragraph;}``) on the pages whose ``@type`` the schema does
        not let them use; ``""`` when absent.
    baseline:
        ``TextLine/Baseline/@points``, ``(N, 2)`` int32 -- a line's text baseline, which
        ``Coords`` alone cannot express.  ``None`` everywhere else.
    text_conf:
        ``TextEquiv/@conf``, the confidence of the transcription whose text is in *text*.
    children:
        The regions nested inside this one -- ``TextLine``\\ s in a ``TextRegion``, ``Word``\\ s
        in a ``TextLine``, ``Glyph``\\ s in a ``Word``.  Empty on a leaf.
    parent:
        The region this one was nested under, or ``None`` for a page-level region and for a
        region the schema does not allow where it was found (see :data:`_ALLOWED_CHILDREN`).
        Excluded from ``__eq__`` and ``__repr__``: comparing it would walk back up through the
        parent's own ``children``, which holds this region, and recurse forever.
    """

    kind: RegionKind
    type_name: str
    region_id: str
    contour: Contour
    text: str | None = None
    custom: str = ""
    baseline: Contour | None = None
    text_conf: float | None = None
    children: list[Region] = field(default_factory=list)
    parent: Region | None = field(default=None, repr=False, compare=False)


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
        Every labelled region found on the page, in document order -- parents before the
        children nested in them.  This flat list is what the rest of the pipeline consumes
        (:meth:`contours_for`, :class:`~sbb_pytorch.gt.render`, the generator's per-class
        contour lookups); the nesting lives on :attr:`Region.children`, reachable through
        :meth:`root_regions` and :meth:`walk`.
    printspace_contour / printspace_bbox:
        Print space outline and its ``[x, y, w, h]`` bounding box, if present.
    region_index:
        ``@ID`` -> ``RegionRefIndexed/@index``, i.e. the reading order.
    external_ref:
        ``Metadata/@externalRef`` -- the archive path of the source image, which a rewrite
        used to drop while keeping the file name.
    creator / created / last_change:
        ``Metadata/Creator``, ``Created`` and ``LastChange``, the three elements the schema
        requires.  ``None`` when the page was not parsed from a file that had them.
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
    external_ref: str = ""
    creator: str | None = None
    created: str | None = None
    last_change: str | None = None

    def root_regions(self) -> list[Region]:
        """The page-level regions: the roots of the trees :attr:`regions` holds flat."""
        return [region for region in self.regions if region.parent is None]

    def walk(self) -> Iterator[tuple[Region, int]]:
        """Every region as ``(region, depth)``, depth-first and parents before children.

        Derived from :meth:`root_regions` and :attr:`Region.children` rather than from the flat
        :attr:`regions` list, so it also says whether the two agree: a region that is neither a
        root nor some root's descendant -- the shape ``_iter_regions`` cannot produce but a
        hand-built :class:`Page` can -- would show up in *regions* and vanish from here.
        """
        for region in self.root_regions():
            yield region, 0
            yield from _descendants(region)

    def contours_for(self, kind: RegionKind, type_name: str | None = None) -> list[Contour]:
        """Contours of every region of *kind*, optionally restricted to *type_name*."""
        return [
            region.contour
            for region in self.regions
            if region.kind is kind and (type_name is None or region.type_name == type_name)
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


def _descendants(region: Region, depth: int = 1) -> Iterator[tuple[Region, int]]:
    """Yield *region*'s descendants as ``(region, depth)``, deepest last."""
    for child in region.children:
        yield child, depth
        yield from _descendants(child, depth + 1)


#: The ``encoding`` pseudo-attribute of an XML declaration, restricted to the characters the
#: XML spec allows in an encoding name.
_XML_DECLARATION: Final = re.compile(rb"<\?xml[^>]*?encoding\s*=\s*[\"']([\w.:-]+)[\"']", re.I)

#: The XML spec's default encoding for a document that declares none.
_XML_DEFAULT_ENCODING: Final = "utf-8"


def _declared_encoding(raw: bytes) -> str:
    """The encoding to decode *raw* with: its declaration, else UTF-8.

    ``ET.XMLParser(encoding=...)`` *overrides* the declaration instead of reading it, so
    the single-encoding policy of the legacy readers (``gt_gen_utils.py:145``) decoded a
    UTF-8-authored page as iso-8859-5 and mangled its text.  Reading the declaration
    first keeps both corpora readable; :func:`_parse_tree` still tries
    :data:`DEFAULT_XML_ENCODING` afterwards.
    """
    match = _XML_DECLARATION.search(raw[:1024])  # The declaration opens the document.
    return match.group(1).decode("ascii", "replace") if match else _XML_DEFAULT_ENCODING


def _parse_tree(path: Path) -> ET.Element:
    """Parse *path*, preferring the encoding its XML declaration names.

    The declared encoding is tried first and :data:`DEFAULT_XML_ENCODING`, the encoding
    the legacy tool wrote every file with, second: a legacy page whose declaration lies
    (an iso-8859-5 file stamped ``encoding="UTF-8"``) decodes either way, while a
    correctly-stamped modern file no longer arrives as mojibake.  An unknown codec
    (``LookupError``) and a syntax error (``ParseError``) both become
    :class:`PageXMLError` rather than escaping as they used to.
    """
    raw = path.read_bytes()
    last_error: Exception = LookupError(f"no candidate encoding for {path}")
    for encoding in dict.fromkeys((_declared_encoding(raw), DEFAULT_XML_ENCODING)):
        try:
            return ET.parse(io.BytesIO(raw), parser=ET.XMLParser(encoding=encoding)).getroot()
        except (ET.ParseError, ValueError, LookupError) as exc:
            last_error = exc
    raise PageXMLError(f"cannot parse {path}: {last_error}") from last_error


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
        root = _parse_tree(path)
    except OSError as exc:  # a missing or unreadable file, as before
        raise PageXMLError(f"cannot parse {path}: {exc}") from exc

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
        **_metadata(root, prefix),
    )


def _columns_width(root: ET.Element, prefix: str) -> int | None:
    """Read ``Metadata/Comments`` for the ``num_colNNNN`` marker the tool writes.

    The value is the digit run *between the marker and the next delimiter*, which is what
    ``int(text_comments.split('num_col')[1])`` (``gt_gen_utils.py:164``) reads -- the
    comma that ends the first width stops the number there, even though the legacy line
    then raised on a two-column page's ``"0456, "``.  Collecting every remaining digit
    instead turned the tool's own ``"#num_col0456, num_col1500"`` into the width
    ``4561500``, which the resize step then tried to honour.
    """
    metadata = root.find(f"{prefix}Metadata")
    if metadata is None:
        return None
    for child in metadata:
        if _local_name(child.tag) != "comments" or not child.text:
            continue
        _, _, after_marker = child.text.partition("num_col")
        digits = "".join(takewhile(str.isdigit, after_marker))
        if digits:
            return int(digits)
    return None


def _metadata(root: ET.Element, prefix: str) -> dict[str, object]:
    """``Metadata``'s remaining fields, as the keyword arguments :class:`Page` takes for them.

    This element used to be read for one thing -- the ``num_col`` marker in ``Comments``, which
    :func:`_columns_width` parses -- and the rest of it was dropped, so rewriting a page lost
    the archive reference and the author it came from.  The three elements the schema requires
    are kept verbatim; a document without them yields ``None`` rather than an empty string, which
    is what tells the writer there is nothing to preserve and it has to state a time itself.
    """
    metadata = root.find(f"{prefix}Metadata")
    if metadata is None:
        return {"external_ref": "", "creator": None, "created": None, "last_change": None}
    elements = {_local_name(child.tag): (child.text or "").strip() for child in metadata}
    return {
        "external_ref": metadata.get("externalRef") or "",
        "creator": elements.get("creator") or None,
        "created": elements.get("created") or None,
        "last_change": elements.get("lastchange") or None,
    }


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
    """Yield every labelled region of the page, whatever its kind, each one before its children.

    The single ``root.iter()`` sweep this function used to be turned up every region in the
    document and threw away the one thing that said which line belonged to which block, so a page
    of 38 lines nested inside one block came back as 38 sibling blocks.  :func:`_build_region`
    reads the same elements and keeps the nesting; the list returned here is walked back out of
    the finished tree, so :attr:`Page.regions` and :meth:`Page.walk` cannot disagree.
    """
    return iter(_walk_element(root, prefix, None))


def _walk_element(element: ET.Element, prefix: str, parent: Region | None) -> list[Region]:
    """Every region at or below *element*, in document order, with the nesting attached.

    *parent* is the region the regions found here belong to.  An element that yields no region of
    its own -- a tag the parser does not know (``Coords``, ``TextEquiv``, the ``<Page>`` itself)
    or a known region whose geometry is missing -- contributes what is inside it, still under the
    same *parent*.  That promotion is what the old flat ``root.iter()`` sweep did by construction,
    and it is why a line inside a contour-less block outlives the block.
    """
    kind = _REGION_TAGS.get(_local_name(element.tag))
    if kind is None:  # not a region at all: Coords, TextEquiv, ReadingOrder, the Page itself
        return _regions_below(element, prefix, parent)
    region = _read_region(element, prefix, kind, _checked_parent(parent, kind))
    if region is None:  # a known region with no usable geometry: keep what is inside, not it
        return _regions_below(element, prefix, parent)
    below = _regions_below(element, prefix, region)
    # Only the regions that ended up naming this one as their parent become its children: the ones
    # :func:`_checked_parent` rejected are page-level orphans, and each is listed exactly once.
    region.children[:] = [child for child in below if child.parent is region]
    return [region, *below]


def _regions_below(element: ET.Element, prefix: str, parent: Region | None) -> list[Region]:
    """Every region under *element*'s children, in document order -- whole subtrees and all.

    Listing them flat here, in the same pass that builds them, is what lets :attr:`Page.regions`
    and :meth:`Page.walk` describe one tree rather than two accounts of it.
    """
    return [region for child in element for region in _walk_element(child, prefix, parent)]


def _checked_parent(parent: Region | None, kind: RegionKind) -> Region | None:
    """Return *parent* when the schema lets a ``kind`` region live inside it, else ``None``.

    A region found somewhere the content model forbids keeps existing -- as a page-level orphan
    rather than under a parent that cannot hold it -- so the odd nesting a file really contains
    survives a round trip instead of being quietly reassigned or dropped.
    """
    if parent is None or kind in _ALLOWED_CHILDREN.get(parent.kind, frozenset()):
        return parent
    return None


def _read_region(
    element: ET.Element, prefix: str, kind: RegionKind, parent: Region | None
) -> Region | None:
    """Build the region *element* describes, or ``None`` when it carries no usable geometry.

    *parent* is the region this one nests in, already checked by :func:`_checked_parent`;
    ``None`` marks a page-level region.  Used for every kind, ``PrintSpace`` and ``Border``
    included, so the attributes are read the same way round whatever the element is.
    """
    contour = _element_contour(element, prefix)
    if contour is None:
        return None
    text, text_conf = _text_of(element, prefix)
    return Region(
        kind=kind,
        type_name=element.get("type", ""),
        # PageXML spells this attribute `id`, ALTO spells it `ID`; both are read here.
        region_id=element.get("ID") or element.get("id") or "",
        contour=contour,
        text=text,
        custom=element.get("custom", ""),
        baseline=_baseline_of(element, prefix),
        text_conf=text_conf,
        parent=parent,
    )


def _text_of(element: ET.Element, prefix: str) -> tuple[str | None, float | None]:
    """``TextEquiv`` of *element* as ``(text, confidence)``, both ``None`` when it has none.

    The two come from one element on purpose: ``@conf`` belongs to the ``TextEquiv``, not to its
    ``Unicode`` child, and reading it from a *different* ``TextEquiv`` than the text -- a
    multi-script page has several -- would attribute one script's confidence to another's.
    """
    equiv = element.find(f"{prefix}TextEquiv")
    if equiv is None:
        return None, None
    unicode_node = equiv.find(f"{prefix}Unicode")
    raw_conf = equiv.get("conf")
    try:
        text_conf = float(raw_conf) if raw_conf is not None else None
    except ValueError:
        text_conf = None  # A malformed confidence costs the number, not the text.
    return (unicode_node.text if unicode_node is not None else None), text_conf


def _baseline_of(element: ET.Element, prefix: str) -> Contour | None:
    """``<Baseline/@points>`` of *element* -- the baseline only a ``TextLine`` can have.

    ``Coords`` describes the glyph box; the baseline it sits on is a separate element, and a
    model trained on lines loses that information entirely if the parse keeps only the box.
    Same coordinate grammar as :func:`_element_contour`, so it goes through the same parser.
    """
    baseline = element.find(f"{prefix}Baseline")
    if baseline is None or not baseline.get("points"):
        return None
    return _parse_coords_points(baseline.get("points"))


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
    return _read_region(element, prefix, RegionKind(_local_name(element.tag)), None)


def discover_xml_files(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """Every XML file under *directory*, sorted, extensions matched case-first.

    Replaces ``get_content_of_dir`` and the ``dir_imgs`` listings it was paired with: the
    legacy version called ``os.listdir`` once per loop iteration (once per file, in the
    ``pagexml2label`` path) and treated subdirectories as files, so a nested directory
    named ``foo.xml`` was handed to the parser.  Files only, ``.xml``/``.XML``, walked once.

    The walk is ``**``-deep since 0.2.0 because a dataset is often a *parent* folder of child
    datasets -- ``smp-private/`` holding one folder per export -- and the one-level listing
    answered such a folder with nothing at all: no files of its own, so
    ``sbb gt generate --xml <parent>`` said ``no ALTO/PageXML files in <parent>`` and a run that
    looked like it covered three datasets had read zero of them.  Pass ``recursive=False`` for
    the old listing, which is what ``sbb gt generate --no-recursive`` now means.

    Sorting is over the full paths, so a nested file sorts by its directory name first; which
    of two files answers the same stem (:func:`~sbb_pytorch.gt.generator.output_stem`) is
    therefore decided by folder order, not by the order the folders were created in.
    """
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    if not base.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    entries = base.rglob("*") if recursive else base.iterdir()
    return sorted(entry for entry in entries if entry.is_file() and entry.suffix.lower() == ".xml")


def image_for(page: Page, directory: str | Path, *, recursive: bool = True) -> Path | None:
    """Locate the raster image matching *page* in *directory*.

    Any extension is accepted, matching the legacy
    ``find_format_of_given_filename_in_dir``, which raised ``IndexError`` instead
    of returning ``None``.

    *directory* is searched in subfolders too (``recursive=False`` restricts it to the folder
    itself), so a page whose photo sits in the same child folder as its XML is still found when
    the pair is addressed through a parent dataset.  The first match in sorted-path order wins,
    which is the same rule as the one-level version for a flat folder.
    """
    base = Path(directory)
    if not base.is_dir():
        return None
    entries = base.rglob("*") if recursive else base.iterdir()
    for entry in sorted(entries):
        if entry.is_file() and entry.stem == page.file_name:
            return entry
    return None
