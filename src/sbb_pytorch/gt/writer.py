"""PageXML out -- and why the file ``sbb infer`` wrote could not be opened.

:mod:`.parser` reads a page, :mod:`.vocabulary` names its elements, and this module writes one
back.  It replaces the five-line generator v3 grew late (``PAGE_XML_Generator``,
``inference.py:761-844``) and the first version of this port's writer, whose output the schema
rejected on six counts -- every one of them visible in a real file, ``/tmp/inference.xml``, where
a page annotated as *one* ``TextRegion`` holding *thirty-eight* ``TextLine``\\ s came back as 239
flat regions and ``xmllint`` answered ``Element 'TextLine': This element is not expected``
thirty-eight times.

What the schema (``pagecontent/2019-07-15``, the one :data:`PAGE_NAMESPACE` names) requires, and
what the old writer did instead:

``Metadata`` is ``Creator``, ``Created``, ``LastChange`` in that order, all three required
    The old writer wrote ``Creator`` alone, because the page it was given had no timestamps to
    write.  v3's answer was ``datetime.now(timezone.utc)`` (``inference.py:791-795``); the port
    answered with ``1970-01-01T00:00:00Z`` instead, which is a well-formed ``xs:dateTime`` that
    tells a reader the file was never stamped -- and a PAGE viewer shows it to them as a date.  The
    parser now reads all three attributes off a page that has them, so a rewrite keeps the real
    ``Creator``/``Created``/``LastChange`` and the ``@externalRef`` beside them; a page with nothing
    to preserve gets :func:`timestamp_now`, the moment it was written, and
    :option:`sbb infer --created` overrides it.
``Page``'s children run ``Border``, ``PrintSpace``, ``ReadingOrder``, *then* the regions
    The old writer emitted the regions first and the reading order last, where the schema puts the
    order before them -- and wrote ``PrintSpace``/``Border`` as members of the region choice, which
    they are not: they are sequence slots of their own, holding nothing but a ``<Coords>``, and they
    take *no attributes at all*, so the ``@id`` the old writer put on one is the error, not the cure.
``TextLine``/``Word``/``Glyph`` nest inside a region and can never be a child of ``Page``
    The old writer emitted every parsed region at page level, because ``Page`` modelled a flat list
    with no parent field and nesting "re-derived parents [that] would reorder the elements the next
    read produced".  Now that :class:`~.parser.Region` has ``children``, the writer writes the tree
    it read -- and wraps a nested-only orphan in the ancestors the schema requires
    (:func:`~.vocabulary.ancestor_chain`) instead of emitting it where it cannot go.
``@id`` is required on every region
    Written always now.  A region that had none is given ``<Element>_<n>``, numbered per element
    name in document order, which is stable enough that re-parsing the written file and writing it
    again produces the same bytes -- v3's ``uuid.uuid4().hex[:8]`` (``inference.py:766-767``)
    renamed every region on every write and destroyed the ids the parser had just read.
``@type`` is a closed enumeration, not a free string
    ``"header, heading"``, ``"<imageregion>"`` and ``"rest_as_paragraph"`` -- all three of which
    this package used to write -- are not in it, and most of the fifteen page-level elements (all of
    ``ImageRegion``, ``SeparatorRegion``, ``TableRegion``, ``NoiseRegion`` among them) accept no
    ``@type`` at all.  :func:`~.vocabulary.type_for` answers ``""`` for anything the element cannot
    say, and the name goes to ``@custom``, where any string is legal.
``<OrderedGroup>`` needs an ``@id`` and at least one indexed child
    Written with the id the tool has used since the eynollah days (:data:`_ORDERED_GROUP_ID`, which
    sat unused in this file while the writer emitted a parentless ``<OrderedGroup>``) and only when
    there is something to order.

Everything else is written as the port's ``Page`` models it: ``Page/@imageFilename`` plus
``@imageWidth``/``@imageHeight``, the ``Metadata/Comments`` column marker, each region's
``Coords/@points``, its ``<Baseline>`` when it has one, its ``TextEquiv/Unicode`` (with ``@conf``
when the parse found one) and the reading order, whose entries are sorted by ``@index`` because the
parser read them out of order.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np

from .parser import Page, Region
from .vocabulary import VOCABULARY, RegionKind, ancestor_chain, custom_for, element_for, type_for

__all__ = [
    "CREATOR",
    "PAGE_NAMESPACE",
    "SCHEMA_LOCATION",
    "points_attribute",
    "timestamp_now",
    "to_pagexml",
    "write_pagexml",
]

#: ``Metadata/Creator``. The two upstream writers disagree on it (``utils/xml.py:36``:
#: ``SBB_QURATOR``; ``inference.py:789``: ``SBB-Textline-Detection``) and a page parsed from a file
#: that says its own (``page.creator``) now wins, so this is what a page with no author is written
#: as: the package's name, not a tool-specific string copied from a different tool.
CREATOR: Final = "SBB-PyTorch"

#: The PAGE content schema this package writes, and the one v3 writes
#: (``inference.py:763``). ``parser._namespace_prefix`` (``parser.py``) derives the prefix
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
#: ``RegionRefIndexed``, never this id, so one constant stands in for the uuid it used to be -- and
#: writing it is not cosmetic: ``@id`` is required on ``OrderedGroupType``, and the writer that
#: omitted it produced an ``<OrderedGroup>`` the schema called invalid.
_ORDERED_GROUP_ID: Final = "ro357564684568544579089"

ET.register_namespace("", PAGE_NAMESPACE)
ET.register_namespace("xsi", _XSI_NAMESPACE)


def timestamp_now() -> str:
    """``Metadata/Created`` and ``LastChange`` for a page that states neither: now, in UTC.

    The schema types both as ``xsd:dateTime``, so the value has to be a real instant. v3 wrote
    ``datetime.now(timezone.utc)`` (``inference.py:791-795``); the port wrote ``1970-01-01T00:00:00Z``
    instead, which is well-formed but tells a reader the file was never stamped -- and a PAGE viewer
    shows it to them as a date. A page parsed from a file keeps its own stamps, so this answers only
    "when did *this* write happen", and :option:`sbb infer --created` overrides it.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    The schema's own pattern (``([0-9]+,[0-9]+ )+([0-9]+,[0-9]+)``) is the reason a whole number of
    pairs is required rather than suggested.

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
    """``Metadata`` in the order the schema requires -- ``Creator``, ``Created``, ``LastChange``,
    then ``Comments`` -- and ``@externalRef`` when the page was parsed from a file that had one.

    The three children are required, which is what the old writer's
    ``<Metadata><Creator/></Metadata>`` failed on, and why a page now has to answer for all three:
    the parsed page's own values when it has them (so a rewrite keeps escriptorium's author and
    stamps instead of erasing them), :data:`CREATOR` and :func:`timestamp_now` when it does not --
    the write's own instant, not the epoch the port used to stamp every unauthored page with.
    ``Comments`` stays optional and is written only when the page has a column count, whose
    digits run to the next delimiter for the reasons :func:`~.parser._columns_width` gives.
    """
    created = page.created or timestamp_now()
    metadata = _element("Metadata")
    if page.external_ref:
        # An optional attribute is omitted rather than written empty, and it belongs to Metadata
        # itself -- so it is set before the element is attached, not passed as a child.
        metadata.set("externalRef", page.external_ref)
    root.append(metadata)
    _sub_element(metadata, "Creator").text = page.creator or CREATOR
    _sub_element(metadata, "Created").text = created
    _sub_element(metadata, "LastChange").text = page.last_change or created
    if page.columns_width is not None:
        # The marker's digits run up to the next delimiter, which is why a bare integer is written
        # instead of eynollah's two-column "num_col0456, num_col1500" string.
        _sub_element(metadata, "Comments").text = f"num_col{page.columns_width}"


class _IdFactory:
    """``@id`` values a rewrite can reproduce: the parsed one, or ``<Element>_<n>``.

    v3 issued ``uuid.uuid4().hex[:8]`` for every region on every write, which renamed a page's
    regions each time and broke the ids ``parser._region_index`` had just read. A number per
    element name, counted in document order, is the same size and is stable instead: the first write
    gives an id-less ``TextRegion`` the name ``TextRegion_1``, and every write after that reads the
    name back and uses it verbatim -- so ``to_pagexml(parse(to_pagexml(page)))`` is the same text.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def __call__(self, kind: RegionKind, existing: str = "") -> str:
        """``@id`` for a region of *kind*: *existing* when it has one, else the next number.

        *existing* is empty for a generated wrapper (a ``TextLine``'s synthetic ``TextRegion`` has
        no id of its own to keep) and for the regions the old writer emitted without one.
        """
        element = element_for(kind)
        if existing:
            return existing
        self._counts[element] = self._counts.get(element, 0) + 1
        return f"{element}_{self._counts[element]}"


def _attributes(kind: RegionKind, region: Region, ids: _IdFactory) -> dict[str, str]:
    """``@id``, ``@type`` and ``@custom`` the region itself is written with, in that order.

    Both optional attributes come from :mod:`.vocabulary`, the only place that knows whether the
    element may say them at all: ``type_for`` is empty for a kind with no ``@type`` attribute or a
    name outside the closed enumeration, and ``custom_for`` carries the class name where ``@type``
    cannot (``structure {type:paragraph;}``), the element's own word where the class does not
    (``text_line``), and nothing where neither has anything to say.  A ``class_1``-style placeholder
    counts as no class at all, so the block a predicted line forms is written ``text_zone`` -- the
    word escriptorium uses for it -- and not a class name the vocabulary has never heard of.
    """
    attributes = {"id": ids(kind, region.region_id)}
    if type_name := type_for(kind, region.type_name):
        attributes["type"] = type_name
    if custom := custom_for(kind, region.type_name):
        attributes["custom"] = custom
    return attributes


def _wrapper_attributes(kind: RegionKind, ids: _IdFactory) -> dict[str, str]:
    """``@id`` and ``@custom`` for a *generated* wrapper element of *kind* (no region to copy).

    Two things a real region's attributes cannot be reused for here.  The ``@id`` has to be counted:
    ``xs:ID`` values are unique across the document, so a ``TextRegion`` wrapped around
    ``<TextLine id="l1">`` cannot also be called ``l1`` -- that is what ``xmllint`` reports as
    ``'l1' is not a valid value of the atomic type 'xs:ID'``.  And the class must not be copied
    either: a region holding a line is a ``text_zone``, not a ``text_line``, so the vocabulary's own
    default is written and ``@type`` stays empty (a name is what makes a ``@type``, and a wrapper has
    none).
    """
    attributes = {"id": ids(kind)}
    if custom := custom_for(kind):
        attributes["custom"] = custom
    return attributes


def _kind_of(region: Region) -> RegionKind:
    """*region*'s kind as a :class:`~.vocabulary.RegionKind` member.

    ``Region.kind`` is typed ``RegionKind``, and a str enum means a caller may hand over the plain
    string (``Region(kind="textregion", ...)``) and get the same member back -- which is what the
    test helpers and :mod:`~sbb_pytorch.cli` do. Normalising once here is what lets every comparison
    in this module be ``is`` and every table lookup be exact, instead of working for one spelling and
    silently missing the other: the first version of this writer compared ``region.kind is
    RegionKind.BORDER`` and so wrote no ``<Border>`` at all for a region built from the string.
    """
    return RegionKind(region.kind)


def _region_element(parent: ET.Element, region: Region, ids: _IdFactory) -> ET.Element:
    """One region: its ``@id``/``@type``/``@custom``, ``Coords``, ``Baseline``, children, text.

    The order is the schema's content model -- ``TextRegionType`` is ``Coords``, the nested regions,
    ``TextLine*``, ``TextEquiv*``, and ``TextLineType`` is ``Coords``, ``Baseline?``, ``Word*``,
    ``TextEquiv*`` -- and it is not a style choice: ``Coords`` is required on every region, the
    transcription has to follow the lines it transcribes, and a ``TextLine`` written where the schema
    wants a ``TextRegion`` is an error the whole file pays for.
    """
    kind = _kind_of(region)
    element = _sub_element(parent, element_for(kind), **_attributes(kind, region, ids))
    _sub_element(element, "Coords", points=points_attribute(region.contour))
    if kind is RegionKind.TEXTLINE and region.baseline is not None:
        # Only ``TextLineType`` has a ``Baseline`` -- ``TextRegionType``, ``WordType`` and
        # ``GlyphType`` do not, so writing one on any of them is invalid even though the parser
        # would read it back. The box and the line the glyphs sit on are two different things, and
        # only the first was ever written.
        _sub_element(element, "Baseline", points=points_attribute(region.baseline))
    for child in region.children:
        _region_element(element, child, ids)
    if region.text:
        # ``TextEquiv/Unicode``, which is what ``parser._text_of`` reads: the text belongs on the
        # ``Unicode`` child, not on ``TextEquiv`` itself, where it would be parsed away. An empty
        # ``<Unicode/>`` parses back as ``None``, so only real text is written -- and ``@conf`` rides
        # on the same ``TextEquiv`` the text came from, because a page with two scripts has two.
        equiv = _sub_element(element, "TextEquiv")
        if region.text_conf is not None:
            equiv.set("conf", f"{region.text_conf:.6g}")
        _sub_element(equiv, "Unicode").text = region.text
    return element


def _area_element(parent: ET.Element, kind: RegionKind, contour: np.ndarray) -> ET.Element:
    """``Page/Border`` or ``Page/PrintSpace``: an element holding one ``<Coords>`` and nothing else.

    Both are ``Page`` sequence slots rather than members of the region choice, and both types are
    ``<sequence><element name="Coords"/></sequence>`` with *no attribute declarations at all* -- not
    even the ``@id`` every region needs, which is why the old writer's ``<PrintSpace id="...">`` was
    rejected and why this helper takes no attributes to write.
    """
    area = _sub_element(parent, element_for(kind))
    _sub_element(area, "Coords", points=points_attribute(contour))
    return area


def _write_roots(page: Page, parent: ET.Element, ids: _IdFactory) -> None:
    """Write every page-level region, wrapping the nested-only ones in the parents they lack.

    A ``TextLine`` written as a child of ``<Page>`` -- which is exactly what the old writer emitted
    for a mask answer, and what the 239-region output was made of -- cannot be written where it is.
    It is wrapped in the chain :func:`~.vocabulary.ancestor_chain` gives (a ``TextLine`` goes inside
    a ``TextRegion``), each wrapper taking its contour from the region it wraps: the schema requires
    the wrapper, so the file gains a region rather than losing a line.
    """
    for region in page.root_regions():
        kind = _kind_of(region)
        vocabulary = VOCABULARY[kind]
        if not vocabulary.page_level and vocabulary.parent is None:
            continue  # Border and PrintSpace: their own Page slots, written before the regions
        target = parent
        for wrapper_kind in ancestor_chain(kind):
            # The wrapper is empty but for the contour it inherits: its text, if any, belongs to the
            # region inside it, and nesting the real region inside it is what makes that legal.
            wrapper = _sub_element(
                target, element_for(wrapper_kind), **_wrapper_attributes(wrapper_kind, ids)
            )
            _sub_element(wrapper, "Coords", points=points_attribute(region.contour))
            target = wrapper
        _region_element(target, region, ids)


def _reading_order_element(parent: ET.Element, region_index: dict[str, int]) -> None:
    """``Page/ReadingOrder/OrderedGroup`` with one ``RegionRefIndexed`` per ordered region id.

    The shape ``utils/xml.py:49-65`` writes and ``parser.py:424-439`` reads; the entries are sorted
    by their ``@index`` rather than emitted in dict order, so a page that was parsed from a
    document listing them out of order (``{"b": 7, "a": 2}``) comes back as the document said.
    """
    group = _sub_element(_sub_element(parent, "ReadingOrder"), "OrderedGroup", id=_ORDERED_GROUP_ID)
    for region_id, index in sorted(region_index.items(), key=lambda item: (item[1], item[0])):
        _sub_element(group, "RegionRefIndexed", index=str(index), regionRef=region_id)


def to_pagexml(page: Page) -> str:
    """Serialise *page* back to PageXML: the exact inverse of :func:`~.parser.parse_pagexml`.

    The output is a pure function of the page -- no timestamp, no uuid, no dict order -- so
    ``to_pagexml(parse_pagexml(to_pagexml(page))) == to_pagexml(page)`` holds, which is what lets
    :command:`sbb infer --xml-file` rewrite a page it has read without churning the file on every
    run (see the module docstring for the v3 generator, which could not say that).

    The children come in the order the schema lists them: ``Metadata``, then inside ``Page`` the
    ``Border`` and ``PrintSpace``, then the ``ReadingOrder``, then the regions -- the position the
    shipped fixtures use and the one the old writer got wrong by writing the regions first.
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
    # Their own slots, in the schema's order (Border before PrintSpace). The parser keeps both among
    # the page's regions, so a Border region is written here and nowhere in the region choice --
    # which is where the old writer's ``@id`` used to appear, on an element with no attributes at all.
    border = next((r.contour for r in page.regions if _kind_of(r) is RegionKind.BORDER), None)
    if border is not None:
        _area_element(page_element, RegionKind.BORDER, border)
    printspace = next(
        (r.contour for r in page.regions if _kind_of(r) is RegionKind.PRINTSPACE),
        page.printspace_contour,
    )
    if printspace is not None:
        _area_element(page_element, RegionKind.PRINTSPACE, printspace)
    if page.region_index:
        # Only when there is something to order: ``OrderedGroupType`` requires at least one indexed
        # child, which is the other thing the old writer's parentless <OrderedGroup> was guilty of.
        _reading_order_element(page_element, page.region_index)
    _write_roots(page, page_element, _IdFactory())

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
