"""PageXML out -- the writer this package never had, and why ``sbb infer --xml-file`` never existed.

:mod:`.parser` reads a page and :mod:`.render` rasterises it; neither could write one back, so
every feature whose answer *is* a file had nothing to answer with. The upstream option is
``inference.py:731-735`` of the v3 repository --

    ``--xml_file``: "xml file with layout coordinates that reading order detection will be
    implemented on. The result will be written in the same xml file."

-- and beside it ``inference.py:736-740``, ``--page-xml``: "save PAGE-XML file with detected
regions and lines." v3 grew a writer for the second one only much later
(``PAGE_XML_Generator``, ``inference.py:761-844``, reached from ``ModelInference.predict`` at
``inference.py:667-672``); the first stayed a help string there and was never implemented here at
all, because writing meant editing a file by hand.

The v3 generator is the shape this module follows and, at the same time, the list of things not to
copy:

``Created`` / ``LastChange`` are stamped ``datetime.now(timezone.utc)`` (``inference.py:791-795``)
    so writing one page twice produced two different files and a generated page could never be
    compared against anything. Nothing here is stamped with a clock: :class:`~.parser.Page` has no
    field for a timestamp -- the eynollah original kept one (``utils/xml.py:32-40``:
    ``Creator='SBB_QURATOR'``, ``Created``, ``LastChange``) and the port's parser dropped all
    three -- so the writer emits what the page models and stays byte-for-byte reproducible rather
    than inventing a value it would have to forget on the next read.
    ``--save-logits`` and :func:`~.write_image` make the same trade in the same direction: the
    output is a function of the input, not of the moment it was written.
    The port's ``Page`` carries no *Creator* either (``parser.py:257-287`` models the file as
    geometry plus the metadata its readers could find), so :data:`CREATOR` names this package in the
    one place the two upstream writers disagree, instead of copying a string that belonged to a
    different tool.
``id`` values come from ``uuid.uuid4().hex[:8]`` (``inference.py:766-767``)
    so a rewrite renamed every region and destroyed the ids ``parser._region_index``
    (``parser.py:424-439``) had just read. Here the id is the parsed one; a region that had none
    stays without one instead of being issued a random name (``parser.py:454-456`` reads both the
    PageXML ``@id`` and the ALTO ``@ID`` spelling, and writes ``@id``, the current schema's).
    The caller that *needs* names -- :func:`~sbb_pytorch.inference` ``--xml-file`` -- gets ids
    derived from the class the region was predicted for, which is what ``generator.output_stem``
    does for the raster side.
It nests ``TextLine`` elements under a generated ``TextRegion`` (``inference.py:809-827``)
    while :class:`~.parser.Page` models a *flat* ``regions`` list with no parent field, whose order
    is what ``Page.ordered_ids`` (``parser.py:305-313``) and ``reading_order.ordered_regions``
    read. Nesting re-derived parents would reorder the elements the next read produced, so every
    region here becomes one element of its own kind, in list order, and the round trip is exact.
    The eynollah writer had the same flattening problem solved the other way round: it wrote real
    ``TextRegion``/``TextLine`` parents because it *had* the hierarchy
    (``writer.py:196-205``, one region per polygon plus its lines); a parsed page has lost it.

Everything the port's ``Page`` does model is written, in the order the parser reads it:
``Page/@imageFilename`` plus ``@imageWidth``/``@imageHeight`` (``parser.py:384-385``), the
``Metadata/Comments`` column marker (``parser.py:401-421``), every region with its ``@type``,
``@id``, ``Coords/@points`` and ``TextEquiv/Unicode`` (``parser.py:442-467``), the print space
(``parser.py:470-484``, which accepts ALTO's ``Border`` as the page outline), and the reading order
(``parser.py:424-439``, written as ``Page/ReadingOrder/OrderedGroup`` the way
``utils/xml.py:49-65`` does, with the same ``@id`` the tool has always used).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

import numpy as np

from .parser import Page, Region, RegionKind

__all__ = [
    "CREATOR",
    "PAGE_NAMESPACE",
    "SCHEMA_LOCATION",
    "points_attribute",
    "to_pagexml",
    "write_pagexml",
]

#: ``Metadata/Creator``. The two upstream writers disagree on it (``utils/xml.py:36``:
#: ``SBB_QURATOR``; ``inference.py:789``: ``SBB-Textline-Detection``) and the port's ``Page`` has no
#: field for it, so the writer names the package instead of copying a tool-specific string.
CREATOR: Final = "SBB-PyTorch"

#: The PAGE content schema this package writes, and the one v3 writes
#: (``inference.py:763``). ``parser._namespace_prefix`` (``parser.py:141-143``) derives the prefix
#: from the root tag instead of the first child, so any namespace -- this one, the
#: ``.../PAGE/gts/schema/v2019.07.24`` spelling the shipped fixtures use, or none at all -- parses
#: back the same way. The port's ``Page`` does not record which one a file declared, so a rewritten
#: page is normalised to the canonical URI rather than guessing at a vendor namespace.
PAGE_NAMESPACE: Final = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

#: ``xsi:schemaLocation`` value, the same pairing v3 writes (``inference.py:764,784``).
SCHEMA_LOCATION: Final = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)

_XSI_NAMESPACE: Final = "http://www.w3.org/2001/XMLSchema-instance"

#: Two spaces, the indent v3 got from ``minidom.toprettyxml(indent="  ")`` (``inference.py:832``).
_INDENT: Final = "  "

#: ``OrderedGroup/@id`` the tool has written since the eynollah days
#: (``utils/xml.py:51``: ``OrderedGroupType(id="ro357564684568544579089")``). The port reads only
#: ``RegionRefIndexed``, never this id, so one constant stands in for the uuid it used to be.
_ORDERED_GROUP_ID: Final = "ro357564684568544579089"

#: Element name per :class:`~.parser.RegionKind` -- the ``_REGION_TAGS`` table
#: (``parser.py:115-130``) read the other way round. The kind enum has exactly these members, so a
#: parsed region always has an element name; v3's generator spelled only ``PcGts``/``Page``/
#: ``TextRegion``/``TextLine``/``ImageRegion`` and let everything else go unnamed.
_REGION_ELEMENTS: Final[dict[RegionKind, str]] = {
    RegionKind.TEXT: "TextRegion",
    RegionKind.GRAPHIC: "GraphicRegion",
    RegionKind.IMAGE: "ImageRegion",
    RegionKind.SEPARATOR: "SeparatorRegion",
    RegionKind.TABLE: "TableRegion",
    RegionKind.NOISE: "NoiseRegion",
    RegionKind.TEXTLINE: "TextLine",
    RegionKind.WORD: "Word",
    RegionKind.GLYPH: "Glyph",
    RegionKind.PRINTSPACE: "PrintSpace",
    RegionKind.BORDER: "Border",
}

ET.register_namespace("", PAGE_NAMESPACE)
ET.register_namespace("xsi", _XSI_NAMESPACE)


def _element(name: str) -> ET.Element:
    """One namespaced element, ``"Page"`` -> ``<Page>`` under :data:`PAGE_NAMESPACE`."""
    return ET.Element(f"{{{PAGE_NAMESPACE}}}{name}")


def _sub_element(parent: ET.Element, name: str, **attributes: str) -> ET.Element:
    """Append a namespaced child holding *attributes*, in the keyword order they are given.

    ElementTree serialises attributes in insertion order (no ``sort_attributes`` option), which is
    what makes the output byte-stable; v3 relied on the same property
    (``inference.py:798-801`` passes ``imageFilename``, ``imageWidth``, ``imageHeight`` in that
    order, which is also the fixture order) and so this helper takes ``**kwargs``, whose order
    Python preserves.
    """
    child = ET.SubElement(parent, f"{{{PAGE_NAMESPACE}}}{name}")
    for key, value in attributes.items():
        child.set(key, value)
    return child


def points_attribute(contour: np.ndarray) -> str:
    """``Coords/@points`` for *contour*: ``"x,y x,y ..."``.

    The same string the parser's own two encodings collapse to
    (``parser.py:159-170`` parses ``"x,y x,y"`` and ``parser.py:186-193`` reads the ``<Point x y>``
    children into the same ``(N, 2)`` array), and the same one v3 built by hand
    (``inference.py:842-844``: ``" ".join(f"{x},{y}" for x, y in polygon)``). Because every stored
    coordinate is already an ``int`` -- :func:`~.parser._coordinate` truncates the fractional ones
    the modern tools emit -- the round trip is exact and no formatting choice can lose a pixel.

    Raises:
        ValueError: *contour* holds a whole number of ``(x, y)`` pairs; a lone coordinate cannot be
            a polygon, and ``parser._parse_coords_points`` would drop it anyway.
    """
    array = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if array.size == 0:
        raise ValueError("points_attribute() got an empty contour")
    return " ".join(f"{x},{y}" for x, y in array.tolist())


def _image_filename(page: Page) -> str:
    """``Page/@imageFilename``: the raster the regions belong to.

    ``Page`` records only the *stem* (``parser.py:266``, "the key the rest of the pipeline names
    outputs by") because the schema keeps the raster's name on the same element the parser reads for
    size only, so the extension is re-derived rather than copied: v3 wrote
    ``os.path.basename(self.image)`` (``inference.py:668-671``) and eynollah the path it was handed
    (``utils/xml.py:32``), while every raster this package names is ``<stem>.png``
    (:func:`sbb_pytorch.data.dataset.label_path_for`). A page parsed from ``page.jpg`` therefore
    keeps saying ``page.jpg``; one parsed from ``page.xml`` says ``page.png``.
    """
    name = page.path.name
    if name and name.rsplit(".", 1)[-1].lower() != "xml":
        return name
    return f"{page.file_name}.png"


def _metadata_element(root: ET.Element, page: Page) -> None:
    """Write ``Metadata`` -- ``Creator`` always, ``Comments`` only when the page has a column count.

    v3 wrote three children here, ``Creator`` plus the two clock stamps
    (``inference.py:787-795``); eynollah wrote ``Creator`` *as an attribute* plus
    ``Comments = "num_col %d"`` (``writer.py:176``); the port's parser reads only the child element
    ``Comments`` (``parser.py:401-421``, which explicitly rejects ``Metadata/Labels`` -- the
    ``<Labels><Label content="num_col0777"/></Metadata>`` spelling is parsed as no width at all)
    and stores nothing else, so writing the two fields that survive the round trip is the honest
    version, and the empty element the no-column page produces is what the reader agrees to again.
    """
    metadata = _sub_element(root, "Metadata")
    _sub_element(metadata, "Creator").text = CREATOR
    if page.columns_width is not None:
        # The marker's digits run up to the next delimiter, which is why a bare integer is written
        # instead of eynollah's two-column "num_col0456, num_col1500" string.
        _sub_element(metadata, "Comments").text = f"num_col{page.columns_width}"


def _region_element(parent: ET.Element, region: Region) -> ET.Element:
    """One region: element named by its kind, with ``@id``, ``@type``, ``Coords`` and ``TextEquiv``.

    Attributes are written in the order the schema and the shipped fixtures use them (``id``, then
    ``type``); each is left out when the parsed region had none, because v3's ``type=""``/``id=""``
    style would come back as a region that has a type and an id of the wrong emptiness -- see the
    module docstring on why ``@id`` is never invented here.
    """
    attributes = {"id": region.region_id} if region.region_id else {}
    if region.type_name:
        attributes["type"] = region.type_name
    element = _sub_element(parent, _REGION_ELEMENTS[region.kind], **attributes)
    _sub_element(element, "Coords", points=points_attribute(region.contour))
    if region.text:
        # ``TextEquiv/Unicode``, which is what ``parser._text_of`` reads (``parser.py:461-467``):
        # the text belongs on the ``Unicode`` child, not on ``TextEquiv`` itself, where it would be
        # parsed away. An empty ``<Unicode/>`` parses back as ``None``, so only real text is written.
        _sub_element(_sub_element(element, "TextEquiv"), "Unicode").text = region.text
    return element


def _reading_order_element(parent: ET.Element, region_index: dict[str, int]) -> None:
    """``Page/ReadingOrder/OrderedGroup`` with one ``RegionRefIndexed`` per ordered region id.

    The shape ``utils/xml.py:49-65`` writes and ``parser.py:424-439`` reads; the entries are sorted
    by their ``@index`` rather than emitted in dict order, so a page that was parsed from a
    document listing them out of order (``{"b": 7, "a": 2}``) comes back as the document said.
    """
    group = _sub_element(_sub_element(parent, "ReadingOrder"), "OrderedGroup")
    for region_id, index in sorted(region_index.items(), key=lambda item: (item[1], item[0])):
        _sub_element(group, "RegionRefIndexed", index=str(index), regionRef=region_id)


def to_pagexml(page: Page) -> str:
    """Serialise *page* back to PageXML: the exact inverse of :func:`~.parser.parse_pagexml`.

    The output is a pure function of the page -- no timestamp, no uuid, no dict order -- so
    ``to_pagexml(parse_pagexml(to_pagexml(page))) == to_pagexml(page)`` holds, which is what lets
    :command:`sbb infer --xml-file` rewrite a page it has read without churning the file on every
    run (see the module docstring for the v3 generator, which could not say that).

    Regions are written in ``page.regions`` order and the reading order after them, the position
    both the schema and the shipped fixtures use; the parser reads every region through
    ``root.iter()`` (``parser.py:442-444``), i.e. in that same document order, so the list it
    rebuilds is element-for-element the one that was written.
    """
    root = _element("PcGts")
    root.set(f"{{{_XSI_NAMESPACE}}}schemaLocation", SCHEMA_LOCATION)
    _metadata_element(root, page)

    page_element = _sub_element(
        root,
        "Page",
        imageFilename=_image_filename(page),
        imageWidth=str(page.width),
        imageHeight=str(page.height),
    )
    for region in page.regions:
        _region_element(page_element, region)
    if page.region_index:
        _reading_order_element(page_element, page.region_index)

    ET.indent(root, space=_INDENT)
    # ``encoding="unicode"`` -- the str form, so no bytes round trip here; the declaration it
    # stamps says utf-8, which is what ``write_pagexml`` then writes the file in.
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def write_pagexml(page: Page, path: str | Path) -> Path:
    """Write :func:`to_pagexml` to *path* as UTF-8 and return the file written.

    ``EynollahXmlWriter.write_pagexml`` (``writer.py:66-69``) opened a filename the constructor had
    already assembled and wrote ``to_xml(pcgts)`` into it, and v3 did the same with a bare ``open``
    (``inference.py:835-837``) -- both raised ``FileNotFoundError`` on a directory nobody had made,
    which is why the parents are created here. *path* ending in a directory-ish name (no ``.xml``
    suffix, or an existing directory) gets the eynollah naming rule (``writer.py:34``:
    ``<stem>.xml``) instead of overwriting a directory.
    """
    target = Path(path)
    if target.is_dir() or not target.suffix:
        target = target.with_name(f"{target.stem}.xml")
    text = to_pagexml(page)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
