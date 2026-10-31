"""Tests for :mod:`sbb_pytorch.gt` -- ALTO/PageXML in, label images out.

The six ``gt`` modules are covered here at unit level; until now they were only reached through
the CLI paths of ``tests/test_cli.py``. Expected values are derived from the geometry or from the
legacy contract each docstring states rather than read back off the implementation: contour areas
are ``cv2.contourArea`` arithmetic written out, a rendered label image is compared pixel-for-pixel
against the derived view it claims to be, and the pair builder is checked against the ``n*(n-1)``
count its module header promises.

The seven behaviours still marked with a ``# TODO(file:line)`` -- asserting what the code did rather
than what it documented -- are now asserted as correct: a ``float64`` contour is widened before
OpenCV measures it, ``mask_of`` takes the bare-int canvas its docstring advertises and fills every
channel of a multi-channel canvas, the rotated rectangle is the inscribed rectangle at every angle
instead of only at 90, a ``Coords/@points`` value may be fractional like its ``<Point>`` twin, a
document's own encoding declaration is honoured before the legacy ``iso-8859-5`` fallback, and
``num_col`` yields the digits up to the next delimiter. Each test names the legacy source the
expectation was derived from. The two defects that header listed besides them -- the class id built
on the per-process ``hash()``, now ``zlib.crc32``, and the background id that skipped the range
check -- had been fixed already, and their tests say so.
"""

from __future__ import annotations

import dataclasses
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from sbb_pytorch.gt import (
    RegionKind,
    generator,
    geometry,
    parser,
    reading_order,
    render,
    vocabulary,
    writer,
)

NS = "http://schema.primaresearch.org/PAGE/gts/schema/v2019.07.24"

# The page most of the module shares: 60x60, a print space, three text regions of three different
# types, one image region, and a reading order that deliberately differs from document order. The
# identifiers use the ALTO spelling (@ID); the PageXML spelling (@id) is pinned separately below.
PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/schema/v2019.07.24">
  <Page imageFilename="000123456-1800.png" imageWidth="60" imageHeight="60">
    <PrintSpace><Coords points="2,2 58,2 58,58 2,58"/></PrintSpace>
    <TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>
    <TextRegion ID="r2" type="header"><Coords points="4,24 56,24 56,32 4,32"/></TextRegion>
    <TextRegion ID="r3" type="paragraph"><Coords points="4,36 30,36 30,52 4,52"/></TextRegion>
    <ImageRegion ID="i1"><Coords points="42,4 56,4 56,20 42,20"/></ImageRegion>
    <ReadingOrder>
      <OrderedGroup id="og">
        <RegionRefIndexed index="0" regionRef="i1"/>
        <RegionRefIndexed index="1" regionRef="r1"/>
        <RegionRefIndexed index="2" regionRef="r2"/>
        <RegionRefIndexed index="3" regionRef="r3"/>
      </OrderedGroup>
    </ReadingOrder>
  </Page>
</PcGts>"""

# One text block holding three lines, in the shape a real annotation has: the lines are inside the
# block, and the block is the only page-level region. The attribute spellings (@id, @custom,
# TextEquiv/@conf, Baseline) and the 461-line page the user annotated are the reason each is here.
NESTED_LINES = """<TextRegion id="block" custom="structure {type:text_zone;}">
  <Coords points="10,10 50,10 50,50 10,50"/>
  <TextLine id="l1" custom="structure {type:text_line;}">
    <Coords points="12,12 48,12 48,20 12,20"/>
    <Baseline points="12,19 48,19"/>
    <TextEquiv conf="0.9999"><Unicode>first</Unicode></TextEquiv>
  </TextLine>
  <TextLine id="l2"><Coords points="12,24 48,24 48,32 12,32"/>
    <TextEquiv conf="0.5"><Unicode>second</Unicode></TextEquiv>
  </TextLine>
  <TextLine id="l3"><Coords points="12,36 48,36 48,44 12,44"/></TextLine>
</TextRegion>"""

# Contours whose geometry is closed-form: a 20x15 rectangle (area 300, boundingRect 21x16 because
# OpenCV counts both end pixels), a 3-4-5 triangle (area 6), a collinear triple (area 0) and a
# 5x15 strip (area 75) used to drive the area filter.
RECT = np.array([[10, 20], [30, 20], [30, 35], [10, 35]], dtype=np.int32)
TRIANGLE = np.array([[0, 0], [4, 0], [0, 3]], dtype=np.int32)
COLLINEAR = np.array([[1, 1], [5, 5], [9, 9]], dtype=np.int32)
STRIP = np.array([[10, 20], [15, 20], [15, 35], [10, 35]], dtype=np.int32)


def xml_document(
    regions: str,
    *,
    width: int = 60,
    height: int = 60,
    metadata: str = "",
    reading_order: str = "",
    namespace: str = NS,
) -> str:
    """A PageXML document holding *regions* on a ``width`` x ``height`` page.

    ``<Metadata>`` belongs beside ``<Page>`` and ``<ReadingOrder>`` inside it -- the distinction
    ``_columns_width`` and ``_region_index`` each look on their own side of. ``namespace=""``
    produces the namespace-less spelling the legacy prefix rule used to mangle.
    """
    root = f'PcGts xmlns="{namespace}"' if namespace else "PcGts"
    page = f'  <Page imageWidth="{width}" imageHeight="{height}">{regions}{reading_order}</Page>'
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<{root}>\n{metadata}{page}\n</{root.split()[0]}>'
    )


def write_xml(tmp_path: Path, name: str, content: str, *, encoding: str = "UTF-8") -> Path:
    """Write *content* to ``tmp_path/name`` in *encoding* and hand back the path."""
    path = tmp_path / name
    path.write_bytes(content.encode(encoding))
    return path


def standard_page(tmp_path: Path) -> parser.Page:
    """Parse the shared :data:`PAGE_XML`, under the name the generator section also writes."""
    return parser.parse_pagexml(write_xml(tmp_path, "000123456-1800.xml", PAGE_XML))


def region(
    kind: parser.RegionKind,
    region_id: str,
    contour: np.ndarray = RECT,
    type_name: str = "",
) -> parser.Region:
    """A :class:`~sbb_pytorch.gt.parser.Region` built by hand, for the sections XML cannot express."""
    return parser.Region(kind=kind, type_name=type_name, region_id=region_id, contour=contour)


# ------------------------------------------------------------------------------------- parser


def test_region_id_is_read_from_both_attribute_spellings(tmp_path: Path) -> None:
    """PageXML spells the identifier ``id``, ALTO spells it ``ID``; both parse (parser.py:383).

    The two spellings were broken at different points of the port, so each gets its own assert.
    ``_iter_regions`` reads ``element.get("ID") or element.get("id") or ""``, so on an element
    carrying both, ``ID`` wins -- pinned too, because silently swapping the precedence would break
    every ALTO file that also carries a lowercase attribute.
    """
    alto = standard_page(tmp_path)  # every region in PAGE_XML uses the ALTO spelling @ID
    assert [r.region_id for r in alto.regions if r.kind is parser.RegionKind.TEXT] == [
        "r1",
        "r2",
        "r3",
    ]

    pagexml = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "lower.xml",
            xml_document(
                '<TextRegion id="p1" type="paragraph">'
                '<Coords points="4,4 20,4 20,20 4,20"/></TextRegion>',
                width=30,
                height=30,
            ),
        )
    )
    assert pagexml.regions[0].region_id == "p1"  # the PageXML spelling @id

    both = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "both.xml",
            xml_document(
                '<TextRegion ID="upper" id="lower">'
                '<Coords points="4,4 20,4 20,20 4,20"/></TextRegion>',
                width=30,
                height=30,
            ),
        )
    )
    assert both.regions[0].region_id == "upper"  # @ID is consulted first


def nested_page(tmp_path: Path) -> parser.Page:
    """Parse :data:`NESTED_LINES` -- one block holding three lines -- as a page."""
    return parser.parse_pagexml(
        write_xml(tmp_path, "nested.xml", xml_document(NESTED_LINES, width=60, height=60))
    )


def test_nested_lines_are_their_regions_children_not_their_siblings(tmp_path: Path) -> None:
    """The defect this exists for: a page whose 38 lines sit in one block came back as 38 blocks.

    ``PAGE_XML`` above is flat, so nothing here noticed that ``_iter_regions``'s ``root.iter()``
    sweep turned a ``TextLine`` nested in a ``TextRegion`` into another ``TextRegion``.
    """
    page = nested_page(tmp_path)
    assert [r.region_id for r in page.regions] == ["block", "l1", "l2", "l3"]  # document order

    block = page.regions[0]
    assert block.kind is parser.RegionKind.TEXT
    assert [child.region_id for child in block.children] == ["l1", "l2", "l3"]
    assert [child.kind for child in block.children] == [parser.RegionKind.TEXTLINE] * 3
    assert all(child.parent is block for child in block.children)  # the edge the writer needs

    assert page.root_regions() == [block]  # the lines are not page-level regions
    assert [(r.region_id, depth) for r, depth in page.walk()] == [
        ("block", 0),
        ("l1", 1),
        ("l2", 1),
        ("l3", 1),
    ]  # walk() covers every region exactly once, at its own depth


def test_words_and_glyphs_nest_under_the_line_that_holds_them(tmp_path: Path) -> None:
    """``TextLine`` holds ``Word``\\ s and ``Word``\\ s hold ``Glyph``\\ s -- three levels deep."""
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "deep.xml",
            xml_document(
                '<TextRegion id="b" type="paragraph"><Coords points="0,0 30,0 30,10 0,10"/>'
                '<TextLine id="l"><Coords points="1,1 29,1 29,9 1,9"/>'
                '<Word><Coords points="1,1 14,1 14,9 1,9"/>'
                '<Glyph><Coords points="1,1 6,1 6,9 1,9"/></Glyph></Word>'
                '<Word><Coords points="16,1 29,1 29,9 16,9"/></Word></TextLine></TextRegion>',
                width=30,
                height=10,
            ),
        )
    )
    line = page.regions[1]
    assert line.kind is parser.RegionKind.TEXTLINE
    assert [w.kind for w in line.children] == [parser.RegionKind.WORD] * 2
    assert [g.kind for g in line.children[0].children] == [parser.RegionKind.GLYPH]
    assert len(list(page.walk())) == len(page.regions) == 5


def test_a_child_the_schema_forbids_where_it_is_found_is_kept_as_an_orphan(tmp_path: Path) -> None:
    """A region in a place the content model does not allow keeps existing -- just not nested.

    Re-parenting it to the nearest legal ancestor would invent a page that was never annotated,
    and dropping it would lose the annotation, so it is recorded at page level with the odd
    nesting left unstated.  That is how the old flat output of ``sbb infer`` round-trips.
    """
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "illegal.xml",
            xml_document(
                '<TextRegion id="outer"><Coords points="0,0 30,0 30,30 0,30"/>'
                '<TextLine id="line"><Coords points="1,1 29,1 29,9 1,9"/>'
                '<TextRegion id="nested"><Coords points="2,2 8,2 8,8 2,8"/></TextRegion>'
                '<Glyph id="g1"><Coords points="3,3 7,3 7,7 3,7"/></Glyph>'
                "</TextLine></TextRegion>",
                width=30,
                height=30,
            ),
        )
    )
    assert [r.region_id for r in page.regions] == ["outer", "line", "nested", "g1"]
    line = page.regions[1]
    assert line.children == []  # a TextLine holds Words -- not a TextRegion, not a Glyph
    # and both of those stay on the page, unparented, rather than being dropped or re-parented:
    assert [r.region_id for r in page.root_regions()] == ["outer", "nested", "g1"]


def test_custom_confidence_and_baseline_are_read(tmp_path: Path) -> None:
    """``@custom``, ``TextEquiv/@conf`` and ``<Baseline>`` -- three attributes once unread."""
    page = nested_page(tmp_path)
    block, first, second, third = page.regions

    assert block.custom == "structure {type:text_zone;}"
    assert first.custom == "structure {type:text_line;}"  # the line's own class
    assert second.custom == ""  # an absent attribute is "", never None
    assert (block.text, block.text_conf) == (None, None)  # a block with no TextEquiv of its own

    assert (first.text, first.text_conf) == ("first", pytest.approx(0.9999))
    assert first.baseline.tolist() == [[12, 19], [48, 19]]  # Coords is the box; this is the line
    assert (second.text, second.text_conf) == ("second", pytest.approx(0.5))
    assert second.baseline is None  # no <Baseline> element
    assert (third.text, third.text_conf) == (None, None)

    malformed = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "conf.xml",
            xml_document(
                '<TextLine id="x"><Coords points="0,0 9,0 9,4 0,4"/>'
                '<TextEquiv conf="high"><Unicode>text</Unicode></TextEquiv></TextLine>',
                width=10,
                height=10,
            ),
        )
    )
    assert (malformed.regions[0].text, malformed.regions[0].text_conf) == (
        "text",
        None,
    )  # a confidence that is not a number costs the number only


def test_a_region_without_geometry_is_dropped_but_the_lines_inside_survive(tmp_path: Path) -> None:
    """The old ``root.iter()`` sweep skipped an unusable element and went on inside it; so does
    the walk -- an element that cannot be drawn must not take its readable children with it.
    """
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "hollow.xml",
            xml_document(
                '<TextRegion id="hollow" type="paragraph">'
                '<TextLine id="inside"><Coords points="4,4 20,4 20,10 4,10"/></TextLine>'
                "</TextRegion>",
                width=30,
                height=20,
            ),
        )
    )
    assert [r.region_id for r in page.regions] == ["inside"]  # the block itself is not a region
    assert page.regions[0].parent is None
    assert page.root_regions() == page.regions  # promoted to page level, not lost with the block


def test_page_metadata_is_read_and_a_page_without_any_is_still_parsed(tmp_path: Path) -> None:
    """``Metadata`` used to be read only for ``num_col``; the rest of it is kept now."""
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "meta.xml",
            xml_document(
                "",
                width=10,
                height=10,
                metadata=(
                    '<Metadata externalRef="mets//sam_40_mss.zip/BL_Or_1443_300.jpg">'
                    "<Creator>escriptorium</Creator>"
                    "<Created>2025-05-10T19:51:07.113207+00:00</Created>"
                    "<LastChange>2025-05-10T20:04:11.553118+00:00</LastChange>"
                    "</Metadata>"
                ),
            ),
        )
    )
    assert page.external_ref == "mets//sam_40_mss.zip/BL_Or_1443_300.jpg"
    assert page.creator == "escriptorium"
    assert page.created == "2025-05-10T19:51:07.113207+00:00"
    assert page.last_change == "2025-05-10T20:04:11.553118+00:00"

    bare = parser.parse_pagexml(
        write_xml(tmp_path, "bare.xml", xml_document("", width=10, height=10))
    )
    assert (bare.external_ref, bare.creator, bare.created, bare.last_change) == (
        "",
        None,
        None,
        None,
    )  # a document with no <Metadata> at all still parses; "" for the attribute, None for elements


def test_the_parent_back_reference_is_excluded_from_equality_and_repr() -> None:
    """``parent`` is ``compare=False, repr=False`` because ``children`` holds the objects it points at.

    Comparing or formatting that field walks parent -> child -> parent forever; the two flags are
    the only thing that stops it, and the generator compares regions by value (``generator.py:551``
    notes the dataclass leaves them unhashable), so the rule is load-bearing rather than cosmetic.
    """
    line = parser.Region(kind=parser.RegionKind.TEXTLINE, type_name="", region_id="l", contour=RECT)
    block = parser.Region(
        kind=parser.RegionKind.TEXT, type_name="paragraph", region_id="b", contour=RECT
    )
    block.children.append(line)
    line.parent = block

    assert line == line  # would raise RecursionError were `parent` compared
    assert "parent=" not in repr(line)
    declared = {item.name: item for item in dataclasses.fields(parser.Region)}
    assert (declared["parent"].repr, declared["parent"].compare) == (False, False)
    assert declared["children"].compare is True  # the tree itself is part of a region's value


def test_page_geometry_and_element_local_names(tmp_path: Path) -> None:
    """``@imageWidth``/``@imageHeight`` become ``width``/``height``, and any case of a tag maps.

    The legacy readers matched both ``}TextRegion`` and ``}Textregion`` because real files carry
    both spellings; :func:`parser._local_name` lower-cases once instead, which is asserted here for
    the tagged and the mis-cased element together with the namespace helpers and the float dims
    ``int(float(...))`` accepts.
    """
    page = standard_page(tmp_path)
    assert (page.height, page.width) == (60, 60)
    assert page.file_name == "000123456-1800" and page.path.name == "000123456-1800.xml"
    assert parser._local_name("{u}TextRegion") == "textregion"
    assert parser._local_name("Textregion") == "textregion"
    assert parser._local_name("PcGts") == "pcgts"
    assert parser.parse_pagexml(str(page.path)).file_name == "000123456-1800"  # str path accepted

    miscased = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "case.xml",
            xml_document(
                '<Textregion ID="mixed"><Coords points="0,0 12,0 12,12 0,12"/></Textregion>',
                width=40,
                height=40,
            ),
        )
    )
    assert [(r.kind, r.region_id) for r in miscased.regions] == [(parser.RegionKind.TEXT, "mixed")]

    fractional_dims = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "dims.xml",
            '<?xml version="1.0"?>\n<PcGts><Page imageWidth="40.0" imageHeight="30.0"/></PcGts>',
        )
    )
    assert (fractional_dims.height, fractional_dims.width) == (30, 40)


def test_namespace_prefix_derived_from_the_tag_not_the_first_element(tmp_path: Path) -> None:
    """A namespace-less document used to yield the prefix ``"PcGts}"``; now it yields ``""``.

    The legacy code took ``alltags[0].split("}")[0] + "}"``, which leaves ``imageHeight`` and
    ``imageWidth`` unbound whenever the file declares no namespace. Both branches of
    :func:`parser._namespace_prefix` are asserted, plus a full parse of a namespace-less page.
    """
    assert parser._namespace_prefix(ET.fromstring(PAGE_XML)) == f"{{{NS}}}"
    assert parser._namespace_prefix(ET.fromstring("<PcGts/>")) == ""

    bare = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "bare.xml",
            xml_document(
                '<TextRegion ID="a"><Coords points="1,1 5,1 5,5 1,5"/></TextRegion>',
                width=60,
                height=50,
                namespace="",  # no xmlns declaration at all
            ),
        )
    )
    assert (bare.height, bare.width) == (50, 60)
    assert bare.regions[0].contour.tolist() == [[1, 1], [5, 1], [5, 5], [1, 5]]


def test_coords_points_become_an_int32_contour(tmp_path: Path) -> None:
    """``Coords/@points`` is the preferred encoding: an ``(N, 2)`` int32 array, verbatim."""
    page = standard_page(tmp_path)
    paragraph = next(r for r in page.regions if r.region_id == "r1")
    assert paragraph.contour.tolist() == [[4, 4], [40, 4], [40, 20], [4, 20]]
    assert paragraph.contour.dtype == np.int32
    assert paragraph.text is None  # r1 carries no TextEquiv

    with_text = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "text.xml",
            xml_document(
                '<TextRegion ID="t" type="paragraph"><Coords points="1,1 9,1 9,9 1,9"/>'
                "<TextEquiv><Unicode>body</Unicode></TextEquiv></TextRegion>",
                width=40,
                height=40,
            ),
        )
    )
    assert with_text.regions[0].text == "body"
    empty_unicode = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "empty_text.xml",
            xml_document(
                '<TextRegion ID="e"><Coords points="1,1 9,1 9,9 1,9"/><TextEquiv/></TextRegion>',
                width=40,
                height=40,
            ),
        )
    )
    assert empty_unicode.regions[0].text is None  # an empty TextEquiv is no text at all


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,2 3,4 5,6", [[1, 2], [3, 4], [5, 6]]),
        ("1 2 3 4 5 6", [[1, 2], [3, 4], [5, 6]]),  # space separated is accepted too
        ("1,2 3,4 5", [[1, 2], [3, 4]]),  # an odd trailing value is dropped, not padded
        ("1 2 3", [[1, 2]]),  # the same truncation with space-only delimiters
        ("1,2", [[1, 2]]),  # a single vertex: not a polygon, but parsed
        ("1,2 1,2 1,2", [[1, 2], [1, 2], [1, 2]]),  # duplicated vertices survive
        ("-3,-4 9,1 9,9 1,9", [[-3, -4], [9, 1], [9, 9], [1, 9]]),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_coords_points_shapes(raw: str, expected: list[list[int]] | None) -> None:
    """Every delimiter and degenerate length of a ``@points`` string, from the same parser.

    Fewer than two values is ``None`` (there is no pair to make), an odd count loses its last
    value, and everything else becomes an ``(N, 2)`` int32 array.
    """
    got = parser._parse_coords_points(raw)
    assert (None if got is None else got.tolist()) == expected


def test_fractional_coords_are_truncated_by_both_encodings(tmp_path: Path) -> None:
    """``@points`` and ``<Point>`` now share one conversion, so a fraction costs nothing but itself.

    The port used to write ``np.asarray(pairs, dtype=np.int32)`` for ``@points`` -- which raises
    :class:`ValueError` on ``"1.5"``, escaping :func:`parse_pagexml` uncaught and costing the whole
    page -- and ``int(float(x))`` for ``<Point>``, which truncates, exactly as the legacy Point
    reader did (``gt_gen_utils.py:197``). Both paths are asserted on the same three numbers: the
    fix was to give ``@points`` the conversion that never loses a page, not to invent a rounding
    rule either path documents.
    """
    fractional = parser._parse_coords_points("1.5,2.5 3.5,4.5 5.5,6.5")
    assert fractional is not None
    assert fractional.tolist() == [[1, 2], [3, 4], [5, 6]]  # truncated, as `int(float)` does

    points = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "floats.xml",
            xml_document(
                '<TextLine ID="l"><Coords>'
                '<Point x="1.5" y="2.5"/><Point x="3.5" y="4.5"/>'
                '<Point x="5.5" y="6.5"/></Coords></TextLine>',
                width=40,
                height=40,
            ),
        )
    )
    assert points.regions[0].contour.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_point_children_are_the_fallback_and_need_three_vertices(tmp_path: Path) -> None:
    """A ``Coords`` without ``@points`` falls back to ``<Point>`` children; two of them is nothing.

    ``_contour_of`` refuses anything below three vertices (a polygon cannot be drawn from two) and
    the caller then drops the region entirely -- ``p_short`` below never reaches ``Page.regions``,
    which is how a region with unusable coordinates disappears from a page.
    """
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "points.xml",
            xml_document(
                '<TextRegion ID="p1"><Coords>'
                '<Point x="2" y="3"/><Point x="8" y="3"/><Point x="8" y="9"/>'
                '<Point x="2" y="9"/></Coords></TextRegion>'
                '<TextRegion ID="p2"><Coords points="10,10 20,20 30,30 10,30"/></TextRegion>'
                '<TextRegion ID="p_short"><Coords>'
                '<Point x="1" y="1"/><Point x="2" y="2"/></Coords></TextRegion>',
                width=50,
                height=50,
            ),
        )
    )
    assert [r.region_id for r in page.regions] == ["p1", "p2"]  # p_short dropped: two points
    assert page.regions[0].contour.tolist() == [[2, 3], [8, 3], [8, 9], [2, 9]]
    assert page.regions[0].kind is parser.RegionKind.TEXT


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "cannot parse"),  # an empty file
        ("not xml at all", "cannot parse"),  # junk
        ("<PcGts><Page>", "cannot parse"),  # unclosed tag
        ('<PcGts><Page imageWidth="x" imageHeight="60"/></PcGts>', "lacks imageHeight"),
        ("<alto><Layout/></alto>", r"expected a <PcGts> root, found <alto>"),
    ],
)
def test_unusable_documents_are_page_xml_errors(tmp_path: Path, body: str, message: str) -> None:
    """Malformed, empty, dimension-less and non-PageXML documents all raise ``PageXMLError``.

    ``PageXMLError`` subclasses ``ValueError`` (asserted once here), so callers that catch the
    parser's own error class also catch codec and syntax failures. The ``<alto>`` row is the one
    gap in the "ALTO/PageXML" promise the module header makes: only a PageXML root -- and its
    ``imageWidth``/``imageHeight`` spelling, not ALTO's ``WIDTH``/``HEIGHT`` -- is understood, and
    the generator's ``--xml`` flag advertises both formats.
    """
    assert issubclass(parser.PageXMLError, ValueError)
    path = write_xml(tmp_path, f"bad-{abs(hash(body))}.xml", body)
    with pytest.raises(parser.PageXMLError, match=message):
        parser.parse_pagexml(path)


def test_missing_file_is_a_page_xml_error_not_an_oserror(tmp_path: Path) -> None:
    """``ET.parse`` raises ``FileNotFoundError``; the port wraps it instead of propagating it."""
    with pytest.raises(parser.PageXMLError, match=r"cannot parse .*missing\.xml"):
        parser.parse_pagexml(tmp_path / "missing.xml")


def test_the_declared_encoding_wins_and_iso_8859_5_is_the_fallback(tmp_path: Path) -> None:
    """The XML declaration is read first, with ``DEFAULT_XML_ENCODING`` behind it.

    The legacy readers passed ``ET.XMLParser(encoding='iso-8859-5')`` for every file
    (``gt_gen_utils.py:145``), and an explicit ``encoding`` *overrides* the declaration instead of
    reading it -- so a UTF-8-authored page, what a modern editor writes, arrived as two mojibake
    characters, which is what this test used to pin. Both spellings of the same text now read back
    as ``Ю``: the iso-8859-5 bytes whose ``encoding="UTF-8"`` declaration cannot decode them, which
    the fallback rescues, and the UTF-8 bytes that declaration correctly names.
    """
    assert parser.DEFAULT_XML_ENCODING == "iso-8859-5"
    cyrillic = xml_document(
        '<TextRegion ID="c"><Coords points="1,1 9,1 9,9 1,9"/>'
        "<TextEquiv><Unicode>Ю</Unicode></TextEquiv></TextRegion>",
        width=20,
        height=20,
    )
    assert (
        parser.parse_pagexml(write_xml(tmp_path, "cyr.xml", cyrillic, encoding="iso-8859-5"))
        .regions[0]
        .text
        == "Ю"
    )
    # Same text, same document, UTF-8 bytes: the declaration the file really satisfies.
    assert parser.parse_pagexml(write_xml(tmp_path, "utf8.xml", cyrillic)).regions[0].text == "Ю"
    # A document with no declaration takes the XML spec's default, not the legacy codec.
    assert parser._declared_encoding(b"<PcGts/>") == parser._XML_DEFAULT_ENCODING


def test_discover_xml_files_and_image_lookup(tmp_path: Path) -> None:
    """XML discovery is sorted, case-insensitive on the suffix and never returns a directory.

    The legacy ``get_content_of_dir`` ran ``os.listdir`` once per iteration and handed the
    subdirectory ``nested.xml`` to the parser; ``image_for`` likewise used to raise ``IndexError``
    for a page without a photo. Both contracts are pinned here, plus the missing-directory case.
    """
    folder = tmp_path / "xml"
    folder.mkdir()
    write_xml(folder, "b.XML", "<PcGts/>")
    write_xml(folder, "a.xml", "<PcGts/>")
    write_xml(folder, "c.txt", "x")
    (folder / "nested.xml").mkdir()

    found = parser.discover_xml_files(folder)
    assert [p.name for p in found] == ["a.xml", "b.XML"]  # .XML matches, .txt and the dir do not
    assert found == sorted(found)
    assert parser.discover_xml_files(str(folder)) == found  # a str path is accepted too
    with pytest.raises(FileNotFoundError):
        parser.discover_xml_files(tmp_path / "absent")

    page = standard_page(tmp_path)  # writes 000123456-1800.xml into tmp_path
    (tmp_path / "000123456-1800.tiff").write_bytes(b"")
    assert parser.image_for(page, tmp_path) == tmp_path / "000123456-1800.tiff"
    (tmp_path / "000123456-1800.png").write_bytes(b"")
    assert parser.image_for(page, str(tmp_path)) == tmp_path / "000123456-1800.png"  # first match
    assert (
        parser.image_for(
            parser.Page(path=tmp_path / "zzz.xml", file_name="zzz", height=5, width=5), tmp_path
        )
        is None
    )


def test_discovery_descends_into_child_dataset_folders(tmp_path: Path) -> None:
    """A parent of child dataset folders is one listing now, and ``recursive=False`` is the old one.

    ``smp-private/`` holding one folder per export has no XML file of its own, so the one-level
    listing answered it with nothing at all -- and a page whose photo sits beside its XML in
    ``childB/`` was invisible to :func:`~sbb_pytorch.gt.parser.image_for` through the parent, so
    the source image of every page in a subfolder was dropped quietly.
    """
    parent = tmp_path / "smp-private"
    child_a = parent / "export_doc11040_mss_pagexml"
    child_b = parent / "sam_44_mss_variants"
    child_a.mkdir(parents=True)
    child_b.mkdir(parents=True)
    write_xml(child_a, "882241214-0113.xml", "<PcGts/>")
    write_xml(child_b, "901234567-0113.xml", "<PcGts/>")

    assert parser.discover_xml_files(parent) == [
        child_a / "882241214-0113.xml",
        child_b / "901234567-0113.xml",
    ]
    assert parser.discover_xml_files(parent, recursive=False) == []  # what the old listing said
    assert parser.discover_xml_files(child_a) == [child_a / "882241214-0113.xml"]  # one child
    with pytest.raises(NotADirectoryError):
        parser.discover_xml_files(child_a / "882241214-0113.xml")  # a file is not a folder

    page = parser.Page(
        path=child_b / "901234567-0113.xml", file_name="901234567-0113", height=5, width=5
    )
    (child_b / "901234567-0113.png").write_bytes(b"")
    assert parser.image_for(page, parent) == child_b / "901234567-0113.png"
    assert parser.image_for(page, parent, recursive=False) is None
    assert parser.image_for(page, child_b) == child_b / "901234567-0113.png"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ("<Metadata><Comments>#num_col0456</Comments></Metadata>", 456),
        ("<Metadata><Comments>no marker here</Comments></Metadata>", None),
        ("<Metadata><Comments/></Metadata>", None),
        ("<Metadata><Labels><Label content='num_col0777'/></Labels></Metadata>", None),
        ("", None),
        # A two-column page lists both widths, exactly as the shipped tool writes them. The value
        # is the digit run between the marker and the comma -- what the legacy
        # `int(text_comments.split('num_col')[1])` (gt_gen_utils.py:164) reads, and what it raised
        # on for this very string -- where the port read 4561500 by keeping every later digit too.
        ("<Metadata><Comments>#num_col0456, num_col1500</Comments></Metadata>", 456),
    ],
)
def test_columns_width_only_ever_reads_comments(
    tmp_path: Path, metadata: str, expected: int | None
) -> None:
    """``Metadata/Comments`` is the only place the column count is looked for (``Labels`` is not)."""
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            f"cw-{abs(hash(metadata))}.xml",
            xml_document(
                '<TextRegion ID="x"><Coords points="0,0 4,0 4,4 0,4"/></TextRegion>',
                width=30,
                height=20,
                metadata=metadata,
            ),
        )
    )
    assert page.columns_width == expected


def test_region_index_skips_unusable_references(tmp_path: Path) -> None:
    """``RegionRefIndexed`` becomes a dict; entries missing ``@index``/``@regionRef`` or holding a
    non-numeric index are skipped rather than aborting the page (the legacy parallel lists did)."""
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "index.xml",
            xml_document(
                '<TextRegion ID="a"><Coords points="0,0 8,0 8,8 0,8"/></TextRegion>'
                '<TextRegion ID="b"><Coords points="20,20 28,20 28,28 20,28"/></TextRegion>'
                '<TextRegion ID="c"><Coords points="30,30 38,30 38,38 30,38"/></TextRegion>',
                width=40,
                height=40,
                reading_order=(
                    "<ReadingOrder><OrderedGroup id='og'>"
                    '<RegionRefIndexed index="7" regionRef="b"/>'
                    '<RegionRefIndexed index="2" regionRef="a"/>'
                    '<RegionRefIndexed index="x" regionRef="c"/>'
                    "<RegionRefIndexed regionRef='a'/>"
                    "</OrderedGroup></ReadingOrder>"
                ),
            ),
        )
    )
    assert page.region_index == {"b": 7, "a": 2}
    assert page.ordered_ids(parser.RegionKind.TEXT) == ["a", "b", "c"]


def test_printspace_contour_bbox_border_and_absence(tmp_path: Path) -> None:
    """Print space geometry, the ``<Border>`` fallback, the off-page clamp and the empty case."""
    page = standard_page(tmp_path)
    assert page.printspace_bbox == (2, 2, 57, 57)
    assert page.printspace_contour.tolist() == [[2, 2], [58, 2], [58, 58], [2, 58]]
    # The print space is not a labelled region family of its own here: it is reachable through
    # Page.printspace_*, and its Region carries the kind `printspace` with no @type and no @ID.
    assert [(r.kind.value, r.type_name, r.region_id) for r in page.regions][:1] == [
        ("printspace", "", "")
    ]

    border = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "border.xml",
            xml_document(
                '<Border><Coords points="6,8 36,8 36,38 6,38"/></Border>', width=40, height=40
            ),
        )
    )
    assert border.printspace_bbox == (6, 8, 31, 31)  # ALTO's <Border> stands in for PrintSpace
    assert [r.kind for r in border.regions] == [parser.RegionKind.BORDER]

    off_page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "offpage.xml",
            xml_document(
                '<PrintSpace><Coords points="6,8 36,8 36,38 6,38"/></PrintSpace>',
                width=10,
                height=10,
            ),
        )
    )
    assert off_page.printspace_bbox == (6, 8, 4, 2)  # clipped to the declared 10x10 page

    empty = parser.parse_pagexml(
        write_xml(tmp_path, "bare_page.xml", xml_document("", width=10, height=10))
    )
    assert (empty.printspace_bbox, empty.printspace_contour) == (None, None)


@pytest.mark.parametrize(
    ("raw", "configured", "expected"),
    [
        ("rest_as_paragraph", None, "paragraph"),  # the rest_as_ prefix names the label itself
        ("rest_as_paragraph", ["header"], "paragraph"),
        ("rest_as_catch-word", ["paragraph"], "catch-word"),
        ("paragraph", ["paragraph"], "paragraph"),
        ("header", ["paragraph"], "header"),  # an active type is never renamed
        ("marginalia", None, "marginalia"),
        ("", ["paragraph"], ""),
        # read_xml had catch-word/page-number commented out, routing them onto paragraph -- but
        # only when the caller passes the configured types (label generation), never for the
        # faithful visualisation read, which is what configured_types=None means.
        ("catch-word", ["paragraph", "header"], "paragraph"),
        ("page-number", ["paragraph"], "paragraph"),
        ("catch-word", ["catch-word"], "catch-word"),  # declared again -> no routing
        ("page-number", ["header"], "page-number"),
        ("catch-word", None, "catch-word"),
    ],
)
def test_normalise_text_type_matrix(raw: str, configured: list[str] | None, expected: str) -> None:
    """Both legacy rename rules, including the one that only applies to configured types."""
    assert parser.normalise_text_type(raw, configured) == expected


def test_page_collection_helpers_and_constants(tmp_path: Path) -> None:
    """``contours_for`` / ``grouped`` / ``ordered_ids`` filter by kind (and optionally by type)."""
    page = standard_page(tmp_path)
    assert len(page.contours_for(parser.RegionKind.TEXT)) == 3
    assert len(page.contours_for(parser.RegionKind.TEXT, "paragraph")) == 2
    assert page.contours_for(parser.RegionKind.WORD) == []  # a kind this page lacks
    assert sorted(page.grouped(parser.RegionKind.TEXT)) == ["header", "paragraph"]
    assert len(page.grouped(parser.RegionKind.TEXT)["paragraph"]) == 2
    assert list(page.grouped(parser.RegionKind.IMAGE)) == [""]  # an untyped region groups under ""
    assert page.ordered_ids(parser.RegionKind.IMAGE) == ["i1"]
    assert page.ordered_ids(parser.RegionKind.WORD) == []  # an absent id is never listed
    assert page.ordered_ids(parser.RegionKind.TEXT, "header") == ["r2"]

    assert parser.DEFAULT_XML_ENCODING == "iso-8859-5"
    assert frozenset({"catch-word", "page-number"}) == parser.LEGACY_INACTIVE_TEXT_TYPES
    assert set(parser.TEXT_REGION_TYPES) >= {"paragraph", "header", "heading", "footnote"}
    assert set(parser.GRAPHIC_REGION_TYPES) >= {"decoration", "stamp", "signature"}
    assert {k.value for k in parser.RegionKind} >= {
        "textregion",
        "graphicregion",
        "imageregion",
        "separatorregion",
        "tableregion",
        "noiseregion",
        "textline",
        "word",
        "glyph",
        "printspace",
        "border",
    }


# ----------------------------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    ("contour", "expected"),
    [
        (RECT, 300.0),  # 20 x 15 closed polygon
        (RECT.astype(np.float32), 300.0),  # the other dtype OpenCV measures natively
        (np.asarray(RECT, dtype=np.uint16), 300.0),
        (np.asarray(RECT, dtype=np.int64), 300.0),
        (np.asarray(RECT, dtype=np.uint8), 300.0),
        (np.asarray(RECT, dtype=np.float64), 300.0),  # widened, no longer an OpenCV error
        (TRIANGLE, 6.0),  # the 3-4-5 triangle: 4 * 3 / 2
        (COLLINEAR, 0.0),  # three collinear points enclose nothing
        (np.array([[0, 0], [4, 5]], dtype=np.int32), 0.0),  # two points are not a polygon
        (np.array([[3, 4]], dtype=np.int32), 0.0),  # a single pixel encloses nothing
        (np.zeros((0, 2), dtype=np.int32), 0.0),  # an empty mask
        (geometry._as_cv_contour(RECT), 300.0),  # the (N, 1, 2) spelling OpenCV wants
        (np.stack([RECT, RECT]).astype(np.int32), 600.0),  # a stacked pair is measured summed
    ],
)
def test_contour_area_accepts_every_shape_and_dtype(contour: np.ndarray, expected: float) -> None:
    """``cv2.contourArea`` arithmetic for every shape and dtype a contour arrives as."""
    assert geometry.contour_area(contour) == pytest.approx(expected)


def test_contour_area_widens_float64_instead_of_refusing_it() -> None:
    """``float64`` -- the dtype almost any numpy arithmetic hands back -- is cast, not refused.

    ``cv2.contourArea`` measures ``int32`` and ``float32`` points only, but the whitelist this test
    used to pin listed ``float64`` as already measurable and handed it over unconverted, so
    ``contour * 2.0``, a promoted ``cv2.transform`` output or a JSON round-trip raised ``cv2.error``
    where the closely related ``float32`` worked. Every dtype now goes through the one narrowing
    :func:`geometry._as_cv_contour` does for all four OpenCV entry points.
    """
    assert geometry.contour_area(np.asarray(RECT, dtype=np.float64)) == pytest.approx(300.0)
    assert geometry.contour_area(np.asarray(RECT * 2, dtype=np.float64)) == pytest.approx(1200.0)


@pytest.mark.parametrize(
    ("contour", "canvas", "expected"),
    [
        (RECT, (40, 40), (10, 20, 21, 16)),  # inclusive: 20-10+1 wide, 35-20+1 tall
        (RECT, None, (10, 20, 21, 16)),  # no canvas: nothing to clip against
        (RECT, (100, 100), (10, 20, 21, 16)),  # a bigger canvas does not shrink the box
        (RECT, (20, 20), (10, 20, 10, 0)),  # clipped on both axes at once
        (RECT, (1, 1), (10, 20, 0, 0)),  # a canvas smaller than the point collapses to 0x0
        (RECT, (50, 50, 3), (10, 20, 21, 16)),  # a channel axis is ignored
        (np.zeros((0, 2), dtype=np.int32), None, (0, 0, 0, 0)),  # empty mask
        (np.array([[3, 4]], dtype=np.int32), None, (3, 4, 1, 1)),  # single pixel
    ],
)
def test_contour_bbox_clips_to_the_canvas(
    contour: np.ndarray, canvas: tuple[int, ...] | None, expected: tuple[int, int, int, int]
) -> None:
    """``contour_bbox`` is ``(x, y, w, h)`` with the inclusive ``w = max - min + 1`` convention,
    clipped to ``canvas`` when one is given."""
    assert geometry.contour_bbox(contour, canvas) == expected


@pytest.mark.parametrize(
    ("contour", "expected"),
    [
        (RECT, (20.0, 27.5)),  # the geometric centre of a 20x15 box at (10, 20)
        (COLLINEAR, (5.5, 5.5)),  # m00 == 0, so the fallback box centre of a 1..9 diagonal
        (TRIANGLE, (4.0 / 3.0, 1.0)),  # the centroid of the 0,0 / 4,0 / 0,3 triangle
        (np.array([[3, 4]], dtype=np.int32), (3.5, 4.5)),  # pixel centre, again via the fallback
    ],
)
def test_contour_centroid_falls_back_to_the_box_centre(
    contour: np.ndarray, expected: tuple[float, float]
) -> None:
    """``cv2.moments`` gives ``m00 == 0`` for anything without area; the fallback is the box centre."""
    assert geometry.contour_centroid(contour) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("contour", "canvas", "painted"),
    [
        (RECT, (40, 40), 336),  # a 20x15 box drawn inclusively fills 21 x 16 pixels
        (TRIANGLE, (40, 40), 13),  # the 4x3 triangle rasterises to 13 pixels, not its area 6
        (np.array([[3, 4]], dtype=np.int32), (40, 40), 1),  # a single pixel
        (np.array([[0, 0], [4, 5]], dtype=np.int32), (40, 40), 6),  # two points draw a line
        ([RECT, TRIANGLE], (40, 40), 349),  # a list is rasterised into one merged canvas
        (np.stack([RECT, RECT]).astype(np.int32), (40, 40), 336),  # a stacked pair draws once
    ],
)
def test_mask_of_rasterises_any_contour_shape(
    contour: np.ndarray | list[np.ndarray], canvas: tuple[int, ...], painted: int
) -> None:
    """``mask_of`` rasterises a contour, a list of them or a ``(K, N, 2)`` batch onto one canvas.

    Pixel counts come from the rasterised polygon (drawn inclusively), not from
    ``cv2.contourArea``: the 20x15 rectangle covers 21x16 = 336 pixels and the 3-4-5 triangle 13.
    ``fill_value=1`` keeps the count readable -- the default is 255, which is what
    :func:`render.to_label_map` and the generator write, and what the uint8 case below shows.
    """
    mask = geometry.mask_of(contour, canvas, fill_value=1)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert int((mask > 0).sum()) == painted


@pytest.mark.parametrize(
    "contour", [np.zeros((0, 2), dtype=np.int32), np.asarray(RECT, dtype=np.uint8)]
)
def test_mask_of_widens_the_contours_fillpoly_cannot_take(contour: np.ndarray) -> None:
    """An empty or ``uint8`` contour is normalised instead of raising ``cv2.error``.

    ``cv2.fillPoly`` takes ``int32`` points only -- a rule :func:`geometry.contour_area` applied
    while ``mask_of`` handed the contour over raw, so a ``uint8`` view of a measurable rectangle and
    a ``np.zeros((0, 2))`` both raised. The conversion now lives in :func:`geometry._as_cv_contour`,
    which both call, and a point-less contour paints nothing onto an otherwise blank canvas instead
    of being handed to ``fillPoly`` at all.
    """
    mask = geometry.mask_of(contour, (40, 40), fill_value=1)
    assert mask.dtype == np.uint8
    assert int((mask > 0).sum()) == (336 if contour.shape[0] else 0)


def test_mask_of_accepts_the_bare_int_canvas_it_documents() -> None:
    """``shape`` accepts the bare int ``40`` as well as ``(40,)`` and ``(40, 40)``.

    ``target = tuple(shape)`` accepted a sequence only, so the plain-int form the docstring
    advertises raised ``TypeError: 'int' object is not iterable``. ``render._hw`` and
    ``render.label_image`` take the same form now; see
    ``test_label_image_is_empty_background_without_entries``.
    """
    assert geometry.mask_of(RECT, (40,)).shape == (40, 40)
    assert geometry.mask_of(RECT, 40).shape == (40, 40)
    assert np.array_equal(geometry.mask_of(RECT, 40), geometry.mask_of(RECT, (40, 40)))


def test_mask_of_paints_every_channel_of_a_multichannel_canvas() -> None:
    """The fill is broadcast across the channels, as the docstring promises it is.

    ``cv2.fillPoly`` with a scalar colour writes only the first channel of a multi-channel image,
    so ``fill_value=7`` on a ``(40, 40, 3)`` canvas used to paint 2352 in channel 0 and leave the
    other two at 0; the colour is now passed as ``(fill_value,) * channels``.
    """
    canvas = geometry.mask_of(RECT, (40, 40, 3), fill_value=7)
    assert canvas.shape == (40, 40, 3)
    assert int(canvas[..., 0].sum()) == 336 * 7
    assert int(canvas[..., 1:].sum()) == 2 * 336 * 7  # the two channels the scalar left empty


@pytest.mark.parametrize(
    ("mask", "mode", "min_area", "expected"),
    [
        (geometry.mask_of(RECT, (40, 40)), cv2.RETR_EXTERNAL, 0.0, [(4, 1, 2)]),
        (geometry.mask_of(RECT, (40, 40)), cv2.RETR_EXTERNAL, 999.0, []),  # floor too high
        (np.zeros((0, 0), dtype=np.uint8), cv2.RETR_EXTERNAL, 0.0, []),  # empty image
        (np.zeros((6, 6), dtype=np.uint8), cv2.RETR_EXTERNAL, 0.0, []),  # all background
    ],
)
def test_contours_of_image_applies_an_absolute_area_floor(
    mask: np.ndarray, mode: int, min_area: float, expected: list[tuple[int, ...]]
) -> None:
    """``contours_of_image`` returns raw contours and filters on *absolute* pixel area, unlike
    :func:`filter_contours_by_area`, which takes a page fraction."""
    found = geometry.contours_of_image(mask, mode=mode, min_area=min_area)
    assert [tuple(c.shape) for c in found] == [tuple(s) for s in expected]


def test_contours_of_region_reads_the_first_channel() -> None:
    """A 3-channel image is reduced to channel 0 before ``findContours``; an empty one yields []."""
    labelled = geometry.mask_of(RECT, (40, 40), fill_value=200)
    assert [geometry.contour_area(c) for c in geometry.contours_of_region(labelled)] == [300.0]
    three_ch = geometry.mask_of(RECT, (40, 40, 3), fill_value=200)
    assert [geometry.contour_area(c) for c in geometry.contours_of_region(three_ch)] == [300.0]
    assert geometry.contours_of_region(np.zeros((0, 0), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    ("contours", "page", "min_area", "max_area", "expected"),
    [
        ([RECT, TRIANGLE, STRIP], (40, 40), 0.0002, 1.0, [300.0, 6.0, 75.0]),  # the defaults
        ([RECT, TRIANGLE, STRIP], (40, 40), 0.01, 1.0, [300.0, 75.0]),  # 6/1600 < 0.01 drops
        ([RECT, TRIANGLE, STRIP], (40, 40), 0.0, 0.1, [6.0, 75.0]),  # 300/1600 > 0.1 drops
        ([RECT, TRIANGLE, STRIP], (40, 40), 0.5, 0.5, []),  # both bounds exclude everything
        ([np.array([[0, 0], [4, 5]], np.int32)], (40, 40), 0.0, 1.0, []),  # 2-point: area 0
        ([RECT], (40,), 0.0002, 1.0, []),  # a 1-dimensional page shape -> page_area 1.0
        ([], (40, 40), 0.0, 1.0, []),  # nothing in, nothing out
    ],
)
def test_filter_contours_by_area_bounds_and_page_shapes(
    contours: list[np.ndarray],
    page: tuple[int, ...],
    min_area: float,
    max_area: float,
    expected: list[float],
) -> None:
    """``min_area``/``max_area`` are page *fractions*, so the page shape decides what survives."""
    result = geometry.filter_contours_by_area(contours, page, min_area=min_area, max_area=max_area)
    assert [geometry.contour_area(c) for c in result.contours] == expected


def test_filter_contours_by_area_accepts_an_array_as_the_page() -> None:
    """The page may be a mask instead of a shape tuple; its area is still ``H * W``."""
    page = np.zeros((40, 40), dtype=np.uint8)
    result = geometry.filter_contours_by_area([RECT, TRIANGLE, STRIP], page)
    assert [geometry.contour_area(c) for c in result.contours] == [300.0, 6.0, 75.0]
    assert result.order_index == [] and result.below_earlier_min is None
    empty = geometry.filter_contours_by_area([], (40, 40))
    assert (empty.contours, empty.order_index, empty.below_earlier_min) == ([], [], None)


def test_filter_contours_by_area_counts_below_earlier_min() -> None:
    """``min_area_early`` marks the contours that only fail the stricter *early* threshold.

    The legacy tool froze ``min_area_early`` after the first frame; the port recomputes it per
    call, which the ``below_earlier_min`` list here pins (all three contours are below 0.5 of the
    page yet above the ordinary 0.001 floor).
    """
    result = geometry.filter_contours_by_area(
        [RECT, TRIANGLE, STRIP],
        (40, 40),
        min_area=0.001,
        min_area_early=0.5,
        order_index=[9, 8, 7],
    )
    assert [geometry.contour_area(c) for c in result.contours] == [300.0, 6.0, 75.0]
    assert result.order_index == [9, 8, 7]
    assert result.below_earlier_min == [1, 1, 1]


def test_return_parent_contours_by_hierarchy_and_by_containment() -> None:
    """With a hierarchy, column 3 (the parent) decides; without one, containment decides instead.

    The docstring describes the OpenCV hierarchy path; the branch that actually runs for most
    callers is the containment fallback, which keeps every contour no other contour contains
    (``roots or contour_list[:1]``). ``FAR`` is not a parent of anything and is still kept -- the
    documented "roots" fallback -- while ``INNER`` disappears inside ``OUTER``.
    """
    outer = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.int32)
    inner = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32)
    far = np.array([[100, 100], [120, 100], [120, 120], [100, 120]], dtype=np.int32)

    fallback = geometry.return_parent_contours([inner, outer, far])
    assert [c.reshape(-1, 2)[0].tolist() for c in fallback] == [[0, 0], [100, 100]]
    assert geometry.return_parent_contours([]) == []
    nested = geometry.return_parent_contours([inner, inner[:3].copy()])
    assert len(nested) == 1 and np.array_equal(nested[0], inner)

    found, found_hierarchy = cv2.findContours(
        geometry.mask_of([outer, inner], (41, 41)), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    via_hierarchy = geometry.return_parent_contours(list(found), found_hierarchy)
    assert [c.reshape(-1, 2)[0].tolist() for c in via_hierarchy] == [[0, 0]]


def test_update_region_contours_erodes_and_carves_the_boundary_ring() -> None:
    """The single-class erosion/dilation recipe, measured on the 20x15 rectangle.

    ``erode``/``dilate`` with a 5x5 kernel shrink and grow by 2 pixels per iteration, so
    ``erosion_rate=0, dilation_rate=2`` plus ``erosion_early=2`` leaves a 13x8 island (area 84)
    inside the original box while the boundary ring holds the other 232 pixels of the painted
    mask. The two-contour and 1-pixel rows pin that the eroded area of a contour that cannot
    survive erosion collapses to a single pixel rather than vanishing.
    """
    eroded, boundary = geometry.update_region_contours(
        [RECT], (40, 40), erosion_rate=0, dilation_rate=2, erosion_early=2, dilation_early=0
    )
    assert [geometry.contour_area(c) for c in eroded] == [84.0]
    assert [geometry.contour_bbox(c) for c in eroded] == [(14, 24, 13, 8)]
    assert int(boundary.sum()) == 232
    assert tuple(np.argwhere(boundary > 0).min(axis=0)) == (20, 10)
    assert tuple(np.argwhere(boundary > 0).max(axis=0)) == (35, 30)

    both, both_boundary = geometry.update_region_contours(
        [RECT, TRIANGLE], (40, 40), erosion_rate=0, dilation_rate=2, erosion_early=2
    )
    assert [geometry.contour_area(c) for c in both] == [84.0, 6.0]
    assert int(both_boundary.sum()) == 232

    untouched, empty_ring = geometry.update_region_contours(
        [RECT], (40, 40), erosion_rate=0, dilation_rate=0
    )
    assert [geometry.contour_area(c) for c in untouched] == [300.0]  # no erosion, no dilation
    assert int(empty_ring.sum()) == 0  # so there is no ring to add back


def test_update_region_contours_accepts_a_bare_box_contour() -> None:
    """A 1x1 contour survives erosion as one pixel -- the erode-then-findContours loop cannot
    produce an empty result for a non-empty input without dropping the region entirely."""
    tiny = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    eroded, boundary = geometry.update_region_contours(
        [tiny], (40, 40), erosion_rate=0, dilation_rate=2
    )
    assert [geometry.contour_area(c) for c in eroded] == [1.0]
    assert eroded[0].reshape(-1, 2)[0].tolist() == [0, 0]
    assert int(boundary.sum()) == 32


@pytest.mark.parametrize(
    ("contour", "canvas", "area", "bbox", "centroid", "aspect", "extent", "y_at_x_min"),
    [
        (RECT, None, 300.0, (10, 20, 21, 16), (20.0, 27.5), 1.3125, 0.8928571428571429, 20),
        (RECT, (25, 25), 300.0, (10, 20, 15, 5), (20.0, 27.5), 3.0, 4.0, 20),
        (COLLINEAR, None, 0.0, (1, 1, 9, 9), (5.5, 5.5), 1.0, 0.0, 1),
        (np.array([[3, 4]], dtype=np.int32), None, 0.0, (3, 4, 1, 1), (3.5, 4.5), 1.0, 0.0, 4),
        (TRIANGLE, None, 6.0, (0, 0, 5, 4), (4.0 / 3.0, 1.0), 1.25, 0.3, 0),
    ],
)
def test_contour_features_one_record_per_contour(
    contour: np.ndarray,
    canvas: tuple[int, int] | None,
    area: float,
    bbox: tuple[int, int, int, int],
    centroid: tuple[float, float],
    aspect: float,
    extent: float,
    y_at_x_min: int,
) -> None:
    """Every field of :class:`ContourFeatures` derived from one contour, hand-computed.

    The clipped row is the interesting one: ``area`` comes from the unclipped contour while
    ``bbox`` is clipped to the canvas, so ``extent`` (area / box area) exceeds 1 -- 300/75 == 4.0.
    """
    features = geometry.ContourFeatures.from_contour(contour, canvas)
    assert features.area == pytest.approx(area)
    assert features.bbox == bbox
    assert features.centroid == pytest.approx(centroid)
    assert features.aspect_ratio == pytest.approx(aspect)
    assert features.extent == pytest.approx(extent)
    assert features.y_at_x_min == y_at_x_min


def test_contour_features_over_a_list_of_contours() -> None:
    """A list produces one feature per entry, in order (the generator's ``labelled_contours`` feed)."""
    features = geometry.contour_features([RECT, TRIANGLE, STRIP])
    assert [(f.bbox, f.area) for f in features] == [
        ((10, 20, 21, 16), 300.0),
        ((0, 0, 5, 4), 6.0),
        ((10, 20, 6, 16), 75.0),
    ]


@pytest.mark.parametrize(
    ("contour", "kwargs", "expected"),
    [
        (RECT, {}, (10, 20, 21, 16)),  # the plain case
        (RECT, {"color": 3, "corr_order_index": 12}, (10, 20, 21, 16)),  # legacy args ignored
        (RECT, {"scale_x": 2.0, "scale_y": 3.0}, (20, 60, 41, 46)),  # scaled about the first vertex
        (np.zeros((0, 2), dtype=np.int32), {"color": 1}, (0, 0, 0, 0)),  # empty contour
    ],
)
def test_bounding_box_scales_and_ignores_the_legacy_arguments(
    contour: np.ndarray, kwargs: dict[str, object], expected: tuple[int, int, int, int]
) -> None:
    """``bounding_box`` derives the box from the contour alone, plus an optional scale about the
    contour's own top-left vertex; the vestigial ``color``/``corr_order_index`` parameters -- the
    legacy ``class_id``/``height``/``width`` trio whose unused ``width`` reference raised the
    ``NameError`` the port documents -- are accepted and never consulted."""
    assert geometry.bounding_box(contour, **kwargs) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"scale_x": 2.0, "scale_y": 1.0}, [[20, 20], [60, 20], [60, 35], [20, 35]]),
        ({"scale_x": 0.5, "scale_y": 0.5}, [[5, 10], [15, 10], [15, 18], [5, 18]]),
        # An explicit origin moves the pivot off the contour: the box scales about (0, 0), so a
        # unit scale lands the rectangle at its own size rather than at its position.
        ({"scale_x": 1.0, "scale_y": 1.0, "origin": (0, 0)}, [[0, 0], [20, 0], [20, 15], [0, 15]]),
        ({"scale_x": 1.0, "scale_y": 1.0}, [[10, 20], [30, 20], [30, 35], [10, 35]]),
    ],
)
def test_scale_contour_about_itself_or_an_explicit_origin(
    kwargs: dict[str, object], expected: list[list[int]]
) -> None:
    """``scale_contour`` scales about the contour's own origin unless ``origin`` says otherwise."""
    assert geometry.scale_contour(RECT, **kwargs).tolist() == expected  # type: ignore[arg-type]


def test_largest_contour_and_the_merged_bounding_box() -> None:
    """``largest_contour`` picks by area; ``largest_contour_bbox`` merges everything into one box."""
    assert geometry.contour_area(geometry.largest_contour([RECT, TRIANGLE, STRIP])) == 300.0
    assert geometry.largest_contour([]) is None
    assert geometry.largest_contour_bbox([RECT, TRIANGLE, STRIP]) == (10, 20, 21, 16)
    assert geometry.largest_contour_bbox([RECT, TRIANGLE, STRIP], (60, 60)) == (10, 20, 21, 16)
    assert geometry.largest_contour_bbox([]) is None
    # Two overlapping boxes merge into the box that holds both, not the biggest single one.
    adjacent = np.array([[28, 32], [32, 32], [32, 36], [28, 36]], dtype=np.int32)
    assert geometry.largest_contour_bbox([RECT, adjacent]) == (10, 20, 23, 17)
    assert geometry.largest_contour_bbox([np.array([[1, 1], [4, 6]], np.int32)]) is None


@pytest.mark.parametrize(
    ("width", "height", "angle", "expected"),
    [
        (100, 50, 0, [[0, 0], [100, 0], [100, 50], [0, 50]]),  # unrotated: the page itself
        (100, 50, 180, [[0, 0], [100, 0], [100, 50], [0, 50]]),  # 180 is the same rotation
        # 50 x 100: taller than the page, so the corners leave the canvas.
        (100, 50, 90, [[25, -25], [75, -25], [75, 75], [25, 75]]),
        (100, 50, 45, [[32, 7], [68, 7], [68, 43], [32, 43]]),  # the 35.36 x 35.36 square at 45
        (100, 50, 30, [[25, 11], [75, 11], [75, 39], [25, 39]]),  # 50 x 28.87 at 30
        (50, 100, 30, [[11, 25], [39, 25], [39, 75], [11, 75]]),  # the same on a portrait page
        (0, 50, 30, [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]),  # zero width: nothing to inscribe
        (-3, 50, 30, [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]),  # negative width
    ],
)
def test_rotated_rect_with_max_area_corner_cases(
    width: int, height: int, angle: int, expected: list[list[int]]
) -> None:
    """The rectangle the legacy formula inscribes, at every angle the port used to collapse.

    The sizes are :func:`utils.rotatedRectWithMaxArea`'s (``geometry.py`` cited it as the port but
    computed only 90 degrees; every other angle came back as the ``[-1, -1]`` sentinel or a quad
    with a repeated vertex) and the placement is the one its caller ``utils.py:346`` used, centred
    on ``w // 2, h // 2``: 100 x 50 unrotated is the page itself, at 90 it is 50 x 100 -- taller
    than the page, hence the corners at ``y = -25`` and ``y = 75`` that the legacy negative ``y1``
    produced -- at 45 the half-constrained ``0.5 * 50 / sin 45`` square, at 30 the
    ``25 / sin 30`` x ``25 / cos 30`` one. ``[-1, -1]`` survives only where there is no page.
    """
    rect = geometry.rotated_rect_with_max_area(width, height, angle)
    assert rect.dtype == np.int32
    assert rect.tolist() == expected


# ------------------------------------------------------------------------------------- render


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (0, (0, 0, 0)),  # the legacy palette's first entry: background
        (1, (255, 0, 0)),
        (6, (0, 125, 255)),
        (18, (125, 255, 0)),  # the last of the 19 entries
        (19, (0, 0, 0)),  # wraps to entry 0 instead of raising IndexError past 18
        (20, (255, 0, 0)),
        (-1, (125, 255, 0)),  # python modulo: -1 % 19 == 18
    ],
)
def test_label_color_wraps_the_legacy_palette(label: int, expected: tuple[int, int, int]) -> None:
    """``DEFAULT_LABEL_PALETTE`` is the legacy ``labels_rgb_color`` table, kept in BGR order,
    and ids past its 19 entries wrap around instead of raising."""
    assert len(render.DEFAULT_LABEL_PALETTE) == 19
    assert render.label_color(label) == expected


def test_label_color_of_an_explicit_mapping_must_contain_the_label() -> None:
    """A ``label -> colour`` mapping raises where a sequence wraps: inventing a colour for a
    configured label would hide a config typo (render.py:146)."""
    assert render.label_color(5, {5: (1, 2, 3)}) == (1, 2, 3)
    with pytest.raises(KeyError, match="no palette entry for label 6"):
        render.label_color(6, {5: (1, 2, 3)})
    # The mapping is keyed by ``int``, so the JSON spelling of a key does not resolve.
    with pytest.raises(KeyError, match="no palette entry for label 5"):
        render.label_color(5, {"5": (1, 2, 3)})


@pytest.mark.parametrize(
    ("raw", "member"),
    [
        ("2d", render.OutputType.TWO_D),
        ("2D", render.OutputType.TWO_D),
        (" 3d ", render.OutputType.THREE_D),
        ("3D", render.OutputType.THREE_D),
    ],
)
def test_output_type_parse_and_reject(raw: str, member: render.OutputType) -> None:
    """``--type-output`` accepts both cases and surrounding space; anything else is a usage error
    rather than the legacy ``UnboundLocalError`` from a fall-through ``if``/``elif``."""
    assert render.OutputType.parse(raw) is member
    assert render.OutputType.parse(member) is member
    assert member.value == raw.strip().lower()


def test_output_type_rejects_a_third_format() -> None:
    with pytest.raises(ValueError, match=r"unknown output type '4d'; expected '2d' or '3d'"):
        render.OutputType.parse("4d")


@pytest.mark.parametrize(
    ("contour", "canvas", "painted"),
    [
        (RECT, (40, 40), 336),  # 21 x 16 pixels: fillPoly counts both end columns and rows
        (TRIANGLE, (40, 40), 13),  # the 3-4-5 triangle rasterises to 13 px, not its area 6
        (COLLINEAR, (40, 40), 9),  # the diagonal line the "polygon" degenerates to
        (STRIP, (40, 40), 96),  # 6 x 16
        (RECT, (20, 20), 10),  # clipped on both axes, the polygon collapses onto row 19
        (RECT, (100, 100), 336),  # a larger canvas changes nothing
    ],
)
def test_label_image_rasterises_any_contour_shape(
    contour: np.ndarray, canvas: tuple[int, int], painted: int
) -> None:
    """An ``(H, W)`` uint8 image whose value *is* the class id, and the clip ``cv2.fillPoly``
    cannot do on its own (see ``geometry._as_contour`` for the off-canvas row)."""
    image = render.label_image([(contour, 7)], canvas, background=3)
    assert image.dtype == np.uint8
    assert image.shape == canvas
    assert int(np.count_nonzero(image == 7)) == painted
    assert int(np.count_nonzero(image == 3)) == canvas[0] * canvas[1] - painted


def test_label_image_is_empty_background_without_entries() -> None:
    assert render.label_image([], (5, 5)).tolist() == [[0] * 5] * 5
    assert render.label_image([], 5).shape == (5, 5), "a bare int edge means a square canvas"
    assert render.label_image([], (2, 3), background=4).tolist() == [[4, 4, 4], [4, 4, 4]]


def test_later_entries_overwrite_earlier_ones() -> None:
    """The painter's order ``render_page`` and ``_layout_map`` both depend on: STRIP inside RECT,
    painted second, keeps its 96 pixels; painted first, all 336 of the rect end up as class 5."""
    forward = render.label_image([(RECT, 5), (STRIP, 7)], (40, 40))
    assert dict(
        zip(*[c.tolist() for c in np.unique(forward, return_counts=True)], strict=False)
    ) == {
        0: 1264,
        5: 240,
        7: 96,
    }
    reverse = render.label_image([(STRIP, 7), (RECT, 5)], (40, 40))
    assert dict(
        zip(*[c.tolist() for c in np.unique(reverse, return_counts=True)], strict=False)
    ) == {
        0: 1264,
        5: 336,
    }


@pytest.mark.parametrize(("label", "message"), [(-1, "class id -1"), (256, "class id 256")])
def test_label_image_rejects_class_ids_that_overflow_uint8(label: int, message: str) -> None:
    """The class id is range-checked before painting, so a ``uint8`` image cannot be silently
    wrapped -- the legacy ``fillPoly(..., color=class_id)`` on an ``int8`` array silently did."""
    with pytest.raises(ValueError, match=rf"{message} does not fit a uint8 label map"):
        render.label_image([(RECT, label)], (40, 40))


def test_label_image_validates_every_entry_before_painting_any() -> None:
    entries = [(RECT, 5), (STRIP, 300), (TRIANGLE, 6)]
    with pytest.raises(ValueError, match="class id 300"):
        render.label_image(entries, (40, 40))


def test_label_image_background_is_range_checked_like_the_entries() -> None:
    """``background`` gets the same ``ValueError`` as an entry label. It used to skip the check and
    reach ``np.full`` with the value, where numpy answered ``OverflowError`` -- and a background
    above 255 is the easy way to mistype a class id (``render.py:191``, pinned here as a bug before
    it was fixed)."""
    with pytest.raises(ValueError, match="background id 999 does not fit a uint8 label map"):
        render.label_image([], (4, 4), background=999)
    assert render.label_image([], (4, 4), background=7).min() == 7


def test_label_map_to_image_matches_the_colour_function_it_calls() -> None:
    """The ``3d`` path and :func:`label_color` cannot disagree, because one indexes the table the
    other returns -- asserted cell by cell rather than by eye."""
    label_map = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    expected = np.array([[list(render.label_color(v)) for v in row] for row in label_map])
    assert render.label_map_to_image(label_map).tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.array([[0, 1], [2, 3]], dtype=np.uint8), np.uint8),  # the plain case
        (np.array([[0, 1], [2, 3]], dtype=np.int64), np.uint8),  # a non-uint8 label map
    ],
)
def test_label_map_to_image_accepts_any_index_dtype(array: np.ndarray, expected: np.dtype) -> None:
    assert render.label_map_to_image(array).dtype == expected


def test_label_map_to_image_takes_channel_zero_and_clips_out_of_range_ids() -> None:
    """A 3-channel input is treated as a label map in channel 0 (the legacy ``img_poly`` read),
    and ids past the palette clamp to its last entry instead of raising."""
    label_map = np.array([[0, 1]], dtype=np.uint8)
    stacked = np.stack([label_map, label_map * 7, label_map * 13], axis=-1)
    assert render.label_map_to_image(stacked)[0, 0].tolist() == list(render.label_color(0))
    clipped = render.label_map_to_image(np.array([[19], [400]], dtype=np.int64))
    assert clipped[0, 0].tolist() == clipped[1, 0].tolist() == list(render.label_color(18))


def test_label_map_to_image_through_a_mapping_leaves_unlisted_labels_black() -> None:
    image = render.label_map_to_image(np.array([[0, 1], [2, 3]], dtype=np.uint8), {1: (9, 9, 9)})
    assert image.tolist() == [[[0, 0, 0], [9, 9, 9]], [[0, 0, 0], [0, 0, 0]]]


def test_to_label_map_repeats_channel_zero_three_times() -> None:
    """The ``2d`` format: three times the memory for one number per pixel, kept because every
    consumer of the existing datasets reads a 3-channel image."""
    label_map = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    three = render.to_label_map(label_map)
    assert three.shape == (2, 2, 3)
    assert np.array_equal(three[..., 0], label_map)
    assert np.array_equal(three[..., 1], label_map)
    assert np.array_equal(three[..., 2], label_map)
    assert np.array_equal(
        render.to_label_map(np.stack([label_map, label_map * 7, label_map * 13], axis=-1)),
        three,
    )


def test_paint_contours_fills_exactly_the_contour_mask() -> None:
    image = render.paint_contours([RECT], (40, 40), (0, 255, 0))
    assert image.shape == (40, 40, 3)
    assert image.dtype == np.uint8
    green = (image == (0, 255, 0)).all(axis=-1)
    assert int(green.sum()) == 336  # the same 336 pixels label_image paints for RECT
    assert int((image != (0, 0, 0)).any(axis=-1).sum()) == 336  # nothing else is lit


def test_paint_contours_without_contours_is_black() -> None:
    assert render.paint_contours([], (5, 6), (1, 2, 3)).tolist() == [[[0, 0, 0]] * 6] * 5


def test_the_outline_is_the_stroked_ring_and_the_fill_keeps_the_rest() -> None:
    """Every pixel ``cv2.drawContours`` strokes belongs to the outline, and ``filled`` keeps only
    what the ring does not cover -- so the parameter works at its default thickness.

    Both counts come from the two numbers this file already pins: the fill-only call paints 336
    pixels for RECT, and a 3-pixel stroke puts 152 of its ring outside that fill. At thickness 1
    the whole 70-pixel ring lies inside the fill, so the split is 70 outline + 266 fill = the same
    336. At thickness 3 the ring is 152 + 186 = 338 pixels, all outline, and the green drops to
    336 - 186 = 150. The first version asked for ``stroked & ~filled``, which is empty at
    thickness 1 -- ``outline_color`` did nothing until the stroke grew past the fill
    (``render.py:272``, pinned here as a bug before it was fixed)."""
    thin = render.paint_contours([RECT], (40, 40), (0, 255, 0), outline_color=(0, 0, 255))
    assert int((thin == (0, 0, 255)).all(axis=-1).sum()) == 70
    assert int((thin == (0, 255, 0)).all(axis=-1).sum()) == 266
    assert int((thin != (0, 0, 0)).any(axis=-1).sum()) == 336  # the fill-only total, unchanged
    thick = render.paint_contours(
        [RECT], (40, 40), (0, 255, 0), outline_color=(0, 0, 255), outline_thickness=3
    )
    assert int((thick == (0, 0, 255)).all(axis=-1).sum()) == 338  # 152 outside + 186 inside
    assert int((thick == (0, 255, 0)).all(axis=-1).sum()) == 150  # 336 - the 186 the ring covers


@pytest.mark.parametrize(
    ("base", "top", "alpha", "expected"),
    [
        (np.full((4, 4, 3), 200, np.uint8), np.zeros((4, 4, 3), np.uint8), 0.5, [100, 100, 100]),
        (np.zeros((2, 2, 3), np.int32), np.full((2, 2, 3), 100, np.int32), 0.5, [50, 50, 50]),
        (np.full((2, 2), 200, np.uint8), np.zeros((2, 2), np.uint8), 1.0, [200, 200, 200]),
    ],
)
def test_overlay_normalises_the_weights_and_the_operand_types(
    base: np.ndarray, top: np.ndarray, alpha: float, expected: list[int]
) -> None:
    """``alpha`` and ``1 - alpha`` sum to 1 -- the legacy ``addWeighted(img, 0.5, layout, 0.1, 0)``
    summed to 0.6 -- and an ``int32``/single-channel operand no longer writes a 32-bit PNG."""
    assert render.overlay(base, top, alpha).tolist()[0][0] == expected


def test_overlay_resizes_the_top_layer_onto_the_base() -> None:
    """A prediction that missed the requested output size used to raise ``cv2.error``; the top
    layer is now nearest-upsampled to the base instead."""
    top = np.full((2, 2, 3), 200, np.uint8)
    blended = render.overlay(np.zeros((4, 4, 3), np.uint8), top, 0.5)
    assert blended.shape == (4, 4, 3)
    assert (blended == [100, 100, 100]).all(axis=-1).sum() == 16  # every pixel, not just 4


def test_default_overlay_alpha_is_the_legacy_ratio_normalised() -> None:
    assert pytest.approx(0.5 / 0.6) == render.DEFAULT_OVERLAY_ALPHA


@pytest.mark.parametrize(
    ("array", "shape", "corner"),
    [
        (np.full((2, 2), 7, np.uint8), (2, 2, 3), [7, 7, 7]),  # 2-D grey
        (np.full((2, 2), 0.5, np.float32), (2, 2, 3), [127, 127, 127]),  # a [0, 1] prediction
        (np.full((2, 2), 300.0), (2, 2, 3), [255, 255, 255]),  # out-of-range float, clipped
        (np.full((2, 2, 3), 9, np.uint8), (2, 2, 3), [9, 9, 9]),  # already BGR
        (np.full((2, 2, 4), 9, np.uint8), (2, 2, 3), [9, 9, 9]),  # BGRA dropped
        (np.full((1, 3, 4, 3), 0, np.uint8), (3, 4, 3), [0, 0, 0]),  # a batch, first item
        (np.full((2, 2, 3), 100, np.int32), (2, 2, 3), [100, 100, 100]),  # int32 demoted
    ],
)
def test_to_uint8_bgr_normalises_every_layout_the_legacy_code_produced(
    array: np.ndarray, shape: tuple[int, ...], corner: list[int]
) -> None:
    """Four divergent ``astype(uint8)``/``cvtColor`` pairs collapsed into one public function."""
    assert render.to_uint8_bgr(array).shape == shape
    assert render.to_uint8_bgr(array)[0, 0].tolist() == corner


def test_to_uint8_bgr_rejects_an_empty_array() -> None:
    with pytest.raises(ValueError, match="empty image"):
        render.to_uint8_bgr(np.zeros((0, 0, 3), dtype=np.uint8))


def test_write_image_and_read_image_round_trip(tmp_path: Path) -> None:
    """``write_image`` creates the parent directory and always writes 8-bit; ``read_image`` hands
    back the ``None`` every legacy caller forgot to check for."""
    image = np.arange(36, dtype=np.uint8).reshape(2, 6, 3)
    target = render.write_image(tmp_path / "deep" / "nested" / "page.png", image)
    assert target == tmp_path / "deep" / "nested" / "page.png"
    assert np.array_equal(render.read_image(target), image)
    assert render.read_image(tmp_path / "absent.png") is None


def test_write_image_downgrades_a_non_uint8_array(tmp_path: Path) -> None:
    """The legacy ``cv2.imwrite`` of an ``int32`` array produced a 32-bit PNG; the array is
    normalised here, so what comes back off disk is the 8-bit image it should have been."""
    target = render.write_image(tmp_path / "float.png", np.full((3, 3), 0.5))
    assert render.read_image(target).dtype == np.uint8
    assert render.read_image(target)[0, 0].tolist() == [127, 127, 127]


def test_image_for_page_locates_the_raster_by_stem(tmp_path: Path) -> None:
    page = standard_page(tmp_path)  # 000123456-1800.xml, and its raster is 000123456-1800.png
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    render.write_image(tmp_path / "000123456-1800.png", image)
    assert np.array_equal(render.image_for_page(page, tmp_path), image)
    (tmp_path / "000123456-1800.png").unlink()
    assert render.image_for_page(page, tmp_path) is None


def test_render_page_equals_the_view_label_image_and_the_palette_derive(tmp_path: Path) -> None:
    """``render_page`` is asserted against the image built by hand from the same entries -- the
    docstring's claim that the two output formats are derived views of one label map."""
    page = standard_page(tmp_path)
    labels = {parser.RegionKind.TEXT: 1, parser.RegionKind.IMAGE: 4}
    entries = sorted(
        ((region.contour, render._label_of(region, labels)) for region in page.regions),
        key=lambda entry: geometry.contour_area(entry[0]),
        reverse=True,
    )
    derived = render.label_image(entries, (60, 60), background=0)
    rendered = render.render_page(page, labels=labels)
    assert np.array_equal(
        rendered, render.label_map_to_image(derived, render.DEFAULT_LABEL_PALETTE)
    )
    assert np.array_equal(
        render.render_page(page, labels=labels, output_type="2d"), render.to_label_map(derived)
    )
    assert render.render_page(page, labels=labels, size=(30, 30)).shape == (30, 30, 3)


@pytest.mark.parametrize(
    ("region", "labels", "expected"),
    [
        (
            region(parser.RegionKind.TEXT, "r1", RECT, "paragraph"),
            {parser.RegionKind.TEXT: 12},
            12,  # the configured mapping wins outright
        ),
        (
            region(parser.RegionKind.TEXT, "paragraph_2", RECT, "paragraph"),
            None,
            2,  # no mapping: the digits of the ALTO @ID, the legacy class encoding
        ),
        (region(parser.RegionKind.TEXT, "r7", RECT, "header"), None, 7),
        (region(parser.RegionKind.IMAGE, "i1", RECT), None, 1),
    ],
)
def test_label_of_prefers_the_mapping_then_the_region_id_digits(
    region: parser.Region, labels: dict[parser.RegionKind, int] | None, expected: int
) -> None:
    """``_label_of`` reads the configured ``RegionKind -> id`` table first, then the digits of
    ``@ID`` -- never falling back to 0, which would bury the region in the background."""
    assert render._label_of(region, labels) == expected


def test_label_of_without_any_digits_hashes_the_identity() -> None:
    """A region with no digits in its ``@ID`` gets a ``zlib.crc32`` of its identity, written out
    here rather than recomputed with the builtin ``hash()`` -- which CPython randomises per
    process, so the value this test asserted used to change on every interpreter start
    (``render.py:418``, pinned here as a bug before it was fixed)."""
    subject = region(parser.RegionKind.TABLE, "", RECT, "table")
    assert render._label_of(subject, None) == 4  # zlib.crc32(b"tableregion:table") % 19


def test_make_image_from_bb_clamps_every_box_to_the_canvas() -> None:
    """``[x, y, w, h]`` boxes on a 10 x 10 canvas: the negative box is clamped to the origin, the
    overflowing box is cut, the off-canvas box is skipped, and the overlap is counted once."""
    boxes = [[2, 2, 3, 4], [8, 8, 10, 10], [-5, -5, 4, 4], [20, 20, 2, 2]]
    image = render.make_image_from_bb(10, 10, boxes, value=1)
    assert image.dtype == np.float64  # the legacy default dtype, kept
    assert image.sum() == 28.0  # 12 + 4 + 16 minus the 4 pixels the first and third boxes share
    assert image[4, 2] == 1.0
    assert np.flatnonzero(image.any(axis=1)).tolist() == [0, 1, 2, 3, 4, 5, 8, 9]
    typed = render.make_image_from_bb(4, 4, [[0, 0, 4, 4]], value=3, dtype=np.uint8)
    assert typed.dtype == np.uint8
    assert set(typed.ravel().tolist()) == {3}


@pytest.mark.parametrize(
    ("image", "bbox", "shape", "rows"),
    [
        (
            np.arange(60, dtype=np.uint8).reshape(6, 10),
            (2, 1, 3, 2),
            (2, 3),
            [[12, 13, 14], [22, 23, 24]],
        ),
        (np.zeros((6, 10, 3), np.uint8), (1, 1, 2, 2), (2, 2, 3), None),  # channels survive
        (np.zeros((6, 10, 3), np.uint8), (4, 4, 9, 9), (2, 6, 3), None),  # numpy clips the slice
    ],
)
def test_crop_to_slices_both_layouts(
    image: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, ...], rows: object
) -> None:
    """``[y : y + h, x : x + w]`` with the channel axis left alone -- the three copies of this
    slice in the legacy code all forgot the degenerate-size case. A box bigger than the image is
    clipped by numpy rather than clamped the way :func:`make_image_from_bb` clamps its boxes, so
    the crop silently comes out smaller than the box asks."""
    cropped = render.crop_to(image, bbox)
    assert cropped.shape == shape
    if rows is not None:
        assert cropped.tolist() == rows


def test_crop_to_without_a_box_or_an_empty_box() -> None:
    """``None``, a zero/negative size and a box that starts outside the image all return the image
    itself instead of the zero-length slice numpy would otherwise hand back."""
    image = np.arange(60, dtype=np.uint8).reshape(6, 10)
    assert render.crop_to(image, None) is image
    assert render.crop_to(image, (0, 0, 0, 5)) is image  # zero width -> unchanged, not empty slice
    assert render.crop_to(image, (0, 0, 3, -1)) is image
    assert render.crop_to(image, (20, 20, 4, 4)) is image  # the origin is past both edges
    assert render.crop_to(image, (1, 1, 2, 2)).tolist() == [[11, 12], [21, 22]]


# ------------------------------------------------------------------------------ reading_order


@pytest.mark.parametrize(("missed", "suffix"), [(False, ""), (True, "_missed")])
def test_pair_suffix_marks_the_newly_admitted_samples(missed: bool, suffix: str) -> None:
    """``_missed`` is the legacy marker for the pairs a *lowered* ``min_area`` newly admitted."""
    pair = reading_order.ReadingOrderPair(
        name="page_0", class_type=1, modal_image=np.zeros((2, 2, 3), np.uint8), missed=missed
    )
    assert pair.suffix == suffix
    assert pair.name == "page_0" and pair.class_type == 1


def test_ordered_regions_follows_the_recorded_index_and_parks_unindexed_regions(
    tmp_path: Path,
) -> None:
    """Order comes from ``RegionRefIndexed/@index``; a region the order never names sorts last
    (``int64`` max) instead of at the front, and ``limit`` truncates the ranked list."""
    page = standard_page(
        tmp_path
    )  # the order names i1=0, r1=1, r2=2, r3=3; the PrintSpace has no @ID
    assert [r.region_id for r in reading_order.ordered_regions(page)] == [
        "i1",
        "r1",
        "r2",
        "r3",
        "",
    ]
    assert [r.region_id for r in reading_order.ordered_regions(page, limit=2)] == ["i1", "r1"]
    assert [r.region_id for r in reading_order.ordered_regions(page, limit=99)][:4] == [
        "i1",
        "r1",
        "r2",
        "r3",
    ]


def test_ordered_regions_falls_back_to_a_top_to_bottom_sweep(tmp_path: Path) -> None:
    """With no ``<ReadingOrder>`` the fallback is (min y, min x): the two regions on the first row
    come back left-to-right before the one below them. The legacy code kept ``pr_list`` and
    ``pos_list`` as parallel lists its consumer re-sorted with ``list.index`` per element."""
    regions = (
        '    <TextRegion ID="top-right" type="paragraph">'
        '<Coords points="30,2 50,2 50,8 30,8"/></TextRegion>\n'
        '    <TextRegion ID="top-left" type="paragraph">'
        '<Coords points="2,2 20,2 20,8 2,8"/></TextRegion>\n'
        '    <TextRegion ID="bottom" type="paragraph">'
        '<Coords points="10,30 40,30 40,36 10,36"/></TextRegion>\n'
    )
    page = parser.parse_pagexml(write_xml(tmp_path, "sweep.xml", xml_document(regions)))
    assert [r.region_id for r in reading_order.ordered_regions(page)] == [
        "top-left",
        "top-right",
        "bottom",
    ]


def test_partial_reading_order_leaves_the_unnamed_regions_at_the_end(tmp_path: Path) -> None:
    """Two of three regions carry an index; the third is unindexed and lands last, in document
    order -- the ``page.region_index.get(id, unindexed)`` default."""
    regions = (
        '    <TextRegion ID="a" type="paragraph"><Coords points="2,2 20,2 20,8 2,8"/></TextRegion>\n'
        '    <TextRegion ID="b" type="paragraph"><Coords points="2,20 20,20 20,26 2,26"/></TextRegion>\n'
        '    <TextRegion ID="c" type="paragraph"><Coords points="2,30 20,30 20,36 2,36"/></TextRegion>\n'
    )
    order = (
        '    <ReadingOrder><OrderedGroup id="og">'
        '<RegionRefIndexed index="7" regionRef="c"/>'
        '<RegionRefIndexed index="3" regionRef="b"/></OrderedGroup></ReadingOrder>\n'
    )
    page = parser.parse_pagexml(
        write_xml(tmp_path, "partial.xml", xml_document(regions, reading_order=order))
    )
    assert [r.region_id for r in reading_order.ordered_regions(page)] == ["b", "c", "a"]


def test_ordered_text_regions_carries_the_reading_order_of_the_kept_contours(
    tmp_path: Path,
) -> None:
    """The filter returns the surviving contours *and* their reading-order indices, so
    ``generate_pairs`` can label a pair without looking the ids up again. r1's index is ``1`` --
    the value ``RegionRefIndexed`` records -- not the ``0`` its position in the region list would
    give it, which is how the wiring between the parser and the filter is proven here."""
    page = standard_page(tmp_path)
    filtered = reading_order.ordered_text_regions(page)
    # The filter hands back the ``(N, 1, 2)`` spelling OpenCV wants; the coordinates are the page's.
    assert [c.reshape(-1, 2).tolist() for c in filtered.contours] == [
        c.reshape(-1, 2).tolist() for c in page.contours_for(parser.RegionKind.TEXT)
    ]
    assert filtered.order_index == [1, 2, 3]
    assert filtered.below_earlier_min is None  # no earlier threshold given -> no _missed flags


def test_ordered_text_regions_flags_the_contours_an_earlier_min_rejected(tmp_path: Path) -> None:
    """``min_area_early`` is how the tool labelled only the samples a lowered threshold newly
    admitted: every contour the earlier, larger bound rejected is flagged ``1``."""
    page = standard_page(tmp_path)
    # Areas are r1 576, r2 416, r3 416 of a 3600-pixel page; a bound of 0.14 (== 504 px) keeps r1
    # alone, so the two smaller regions are the ones an earlier run would have skipped.
    filtered = reading_order.ordered_text_regions(page, min_area=0.0002, min_area_early=0.14)
    assert filtered.below_earlier_min == [0, 1, 1]
    roomy = reading_order.ordered_text_regions(page, min_area=0.0002, min_area_early=0.05)
    assert roomy.below_earlier_min == [
        0,
        0,
        0,
    ]  # a lower bound drops nothing, so nothing is flagged


@pytest.mark.parametrize(
    ("target", "rows", "cols", "count"),
    [
        ((60, 60), range(33, 45), range(4, 57), 636),  # y + h == 33, the band runs 12 rows down
        ((30, 30), range(17, 29), range(2, 29), 324),  # the same header at half scale
    ],
)
def test_header_separator_band_sits_under_the_header(
    tmp_path: Path, target: tuple[int, int], rows: range, cols: range, count: int
) -> None:
    """The legacy ``img_header_and_sep[y_max:y_max + 12, x_min:x_max] = 1``: a 12-pixel strip
    starting one row below the header's bottom edge, spanning its width, at ``HEADER_LABEL``. The
    second row asks for a 30 x 30 network size, which scales the *contour* by 0.5 (see
    ``geometry.scale_contour``: the coordinates are multiplied, the raster is not resized), so the
    band halves with it -- 12 rows by 27 columns of a 30 x 30 canvas."""
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "header.xml",
            xml_document(
                '<TextRegion ID="r2" type="header">'
                '<Coords points="4,24 56,24 56,32 4,32"/></TextRegion>',
            ),
        )
    )
    mask = reading_order.header_separator_mask(page, (page.height, page.width), target_size=target)
    hit = np.nonzero(mask == reading_order.HEADER_LABEL)
    assert (set(hit[0].tolist()), set(hit[1].tolist())) == (set(rows), set(cols))
    assert int(np.count_nonzero(mask == reading_order.HEADER_LABEL)) == count


def test_every_header_spelling_the_page_uses_gets_the_label(tmp_path: Path) -> None:
    """The page's own type list is what decides, through ``parser.normalise_text_type``: the bare
    ``header``, the legacy ``heading`` and the ``rest_as_header`` a page-to-label config writes are
    all headers. The first version compared the raw ``@type`` against the literal ``"header"`` and
    left the other two without a separator band, which is all the ``page`` argument was for
    (``reading_order.py:206``, pinned here as a bug before it was fixed)."""
    header = region(parser.RegionKind.TEXT, "h", TRIANGLE, "header")
    heading = region(parser.RegionKind.TEXT, "g", TRIANGLE, "heading")
    aliased = region(parser.RegionKind.TEXT, "r", TRIANGLE, "rest_as_header")
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "kinds.xml",
            xml_document(
                '<TextRegion ID="h" type="header"><Coords points="4,4 20,4 20,8 4,8"/></TextRegion>'
                '<TextRegion ID="g" type="heading"><Coords points="4,10 20,10 20,14 4,14"/></TextRegion>'
            ),
        )
    )
    assert reading_order._is_header(page, header) is True
    assert reading_order._is_header(page, heading) is True
    assert reading_order._is_header(page, aliased) is True
    assert page.regions[1].type_name == "heading"  # parsed as written, normalised for the lookup


def test_region_label_map_applies_the_query_then_the_other_kinds(tmp_path: Path) -> None:
    """The three-label alphabet (query 1, image 2, header 3) over the shared page: the queried
    region's 629 pixels, the image region's 255 and the header band's 636 overwrite what they
    cover, exactly as the legacy order of ``fillPoly`` calls did."""
    page = standard_page(tmp_path)
    queried = next(r for r in page.regions if r.region_id == "r1")  # 37 x 17 = 629 pixels
    counts = dict(
        zip(
            *[
                c.tolist()
                for c in np.unique(
                    reading_order.region_label_map(queried.contour, page, (60, 60)),
                    return_counts=True,
                )
            ],
            strict=False,
        )
    )
    assert counts == {0: 2080, 1: 629, 2: 255, 3: 636}


def test_region_label_map_lets_the_other_kinds_win_the_overlap(tmp_path: Path) -> None:
    """Where the queried region and the image region overlap, the image label (2) wins -- the
    legacy overwrites ran *after* the query fill and are reproduced in that order."""
    page = standard_page(tmp_path)
    image = next(r for r in page.regions if r.kind is parser.RegionKind.IMAGE)
    mask = reading_order.region_label_map(image.contour, page, (60, 60))
    assert int((mask == reading_order.QUERY_LABEL).sum()) == 0  # the query is entirely overwritten
    assert int((mask == reading_order.IMAGE_REGION_LABEL).sum()) == 255


def test_generate_pairs_emits_every_ordered_pair_of_one_page(tmp_path: Path) -> None:
    """``n * (n - 1)`` samples, the counter in the file names shared by both directions, and the
    label ``1`` exactly when the first region precedes the second in reading order."""
    page = standard_page(tmp_path)
    pairs = reading_order.generate_pairs(page)
    assert [(p.name, p.class_type) for p in pairs] == [
        ("000123456-1800_0", 1),  # (r1, r2) -- index 1 before 2
        ("000123456-1800_1", 1),  # (r1, r3)
        ("000123456-1800_2", 0),  # (r2, r1) -- the reverse pair is a real sample, not a duplicate
        ("000123456-1800_3", 1),  # (r2, r3)
        ("000123456-1800_4", 0),  # (r3, r1)
        ("000123456-1800_5", 0),  # (r3, r2)
    ]
    renamed = [(p.name, p.class_type) for p in reading_order.generate_pairs(page, index_start=5)]
    assert renamed[0] == ("000123456-1800_5", 1)  # the counter is the only thing that moves
    assert len(renamed) == 6


def test_generate_pairs_stacks_the_three_modal_channels(tmp_path: Path) -> None:
    """Channel 0 is the first region's label map, channel 1 the page layout raster, channel 2 the
    second region's -- each rebuilt here from the same helpers the pair builder calls."""
    page = standard_page(tmp_path)
    filtered = reading_order.ordered_text_regions(page)
    pair = reading_order.generate_pairs(page)[0]
    assert pair.modal_image.shape == (60, 60, 3)
    assert pair.modal_image.dtype == np.uint8  # the legacy int8 array wrapped every value > 127
    first = reading_order.region_label_map(filtered.contours[0], page, (60, 60))
    second = reading_order.region_label_map(filtered.contours[1], page, (60, 60))
    layout = reading_order._layout_channel(page, (60, 60), (60, 60))
    assert np.array_equal(pair.modal_image[..., 0], first)
    assert np.array_equal(pair.modal_image[..., 1], layout)
    assert np.array_equal(pair.modal_image[..., 2], second)


def test_generate_pairs_needs_two_text_regions(tmp_path: Path) -> None:
    page = parser.parse_pagexml(
        write_xml(
            tmp_path,
            "one.xml",
            xml_document(
                '<TextRegion ID="a" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>'
            ),
        )
    )
    assert reading_order.generate_pairs(page) == []


def test_generate_pairs_rasterises_every_channel_at_the_requested_size(tmp_path: Path) -> None:
    """An ``input_height``/``input_width`` other than the page's own used to raise ``ValueError:
    all input arrays must have the same shape`` from ``np.stack`` -- the layout channel was
    resampled to the network size while the two label maps stayed at page size, so every config
    whose input size differed from the page produced no samples at all
    (``reading_order.py:315``, pinned here as a bug before it was fixed). All three channels now
    share the requested grid, which is what the docstring's "rasterised at this size directly"
    always promised."""
    page = standard_page(tmp_path)
    pairs = reading_order.generate_pairs(page, input_height=30, input_width=30)
    assert len(pairs) == 6  # the same n * (n - 1) the page-sized call yields
    assert {pair.modal_image.shape for pair in pairs} == {(30, 30, 3)}
    assert [pair.name for pair in pairs][:2] == ["000123456-1800_0", "000123456-1800_1"]


@pytest.mark.parametrize(("missed", "stem"), [(False, "page_0"), (True, "page_0_missed")])
def test_write_pair_composes_the_name_and_the_missed_suffix(
    tmp_path: Path, missed: bool, stem: str
) -> None:
    """``<name><suffix>.png`` beside the matching ``.npy`` holding the scalar class as a 0-d array,
    so ``np.load`` returns what the legacy ``np.save(..., class_type)`` did. ``name`` holds the stem
    and ``suffix`` the ``_missed`` marker, and this function joins them -- the first version wrote
    ``name`` alone and left the property unused, so only a pair whose marker ``generate_pairs`` had
    folded into the name came out right (``reading_order.py:123``, pinned here as a bug before it
    was fixed)."""
    image = np.zeros((6, 6, 3), np.uint8)
    image[..., 2] = 7
    pair = reading_order.ReadingOrderPair(
        name="page_0", class_type=1, modal_image=image, missed=missed
    )
    image_path, class_path = reading_order.write_pair(pair, tmp_path / "imgs", tmp_path / "cls")
    assert (image_path, class_path) == (
        tmp_path / f"imgs/{stem}.png",
        tmp_path / f"cls/{stem}.npy",
    )
    assert np.array_equal(render.read_image(image_path), image)
    stored = np.load(class_path)
    assert (stored.dtype, stored.ndim, stored.item()) == (np.dtype("int8"), 0, 1)


def test_visualize_reading_order_draws_the_chain_and_refuses_an_empty_one(tmp_path: Path) -> None:
    """The arrow port: the arrow canvas holds 255 at every centroid it joins, and blending it onto
    a black page gives the normalised ``alpha`` value there instead of the legacy 0.6-bright image."""
    page = standard_page(tmp_path)
    regions = [r for r in page.regions if r.region_id in {"r1", "r3"}]
    arrows = reading_order._arrows(regions, (60, 60), (0, 0, 255), 2)
    assert arrows[12, 22].tolist() == [0, 0, 255]  # r1's centroid, the first arrow's tail
    assert int((arrows != 0).any(axis=-1).sum()) == 106  # exactly the two arrowed segments
    blended = reading_order.visualize_reading_order(np.zeros((60, 60, 3), np.uint8), regions)
    assert blended[12, 22].tolist() == [0, 0, 128]  # 255 * (1 - 0.5) on a black base
    single = reading_order.visualize_reading_order(np.full((60, 60, 3), 200, np.uint8), regions[:1])
    assert set(single.ravel().tolist()) == {100}  # one region -> no arrows -> base * alpha only
    with pytest.raises(ValueError, match="visualize_reading_order..: no regions given"):
        reading_order.visualize_reading_order(np.zeros((6, 6, 3), np.uint8), [])


@pytest.mark.parametrize(
    ("order_index", "i", "j", "expected"),
    [
        ([0, 1, 2], 0, 1, 1),
        ([0, 1, 2], 1, 0, 0),
        ([0, 1, 2], 0, 2, 1),
        ([5], 0, 0, 0),
        ([], 0, 1, 0),
    ],
)
def test_pair_label_is_the_legacy_index_comparison(
    order_index: list[int], i: int, j: int, expected: int
) -> None:
    """``int(index_i < index_j)`` -- including the legacy behaviour of labelling every pair ``0``
    when no index was recorded at all."""
    assert reading_order._pair_label(order_index, i, j) == expected


def test_ordered_pairs_is_the_full_matrix(tmp_path: Path) -> None:
    assert list(reading_order._ordered_pairs(2)) == [(0, 1), (1, 0)]
    assert len(list(reading_order._ordered_pairs(3))) == 6
    assert len(list(reading_order._ordered_pairs(1))) == 0


# ---------------------------------------------------------------------------------- generator


@pytest.mark.parametrize(
    "raw",
    ["layout", "textline", "word", "glyph", "printspace", "binarization", "reading_order"],
)
def test_use_case_covers_all_seven_and_rejects_an_eighth(raw: str) -> None:
    """The enum has seven members; the CLI used to offer the five of ``config.models`` and so
    refused ``binarization`` and ``reading_order`` outright. Case and padding are tolerated, and an
    existing member passes through -- the legacy code compared four string literals with ``or``
    and let anything else fall through both branches, writing nothing."""
    assert generator.UseCase.parse(raw) is generator.UseCase(raw)
    assert generator.UseCase.parse(f"  {raw.upper()} ") is generator.UseCase(raw)


def test_use_case_parse_is_identity_and_names_the_bad_value() -> None:
    member = generator.UseCase.GLYPH
    assert generator.UseCase.parse(member) is member
    assert {u.value for u in generator.UseCase} == {
        "layout",
        "textline",
        "word",
        "glyph",
        "printspace",
        "binarization",
        "reading_order",
    }
    with pytest.raises(ValueError, match=r"unknown use case 'layout2d'"):
        generator.UseCase.parse("layout2d")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("000123456-1800.xml", "1800"),  # the legacy rule: the segment after the first hyphen
        ("000123456.xml", "000123456"),  # no hyphen -> the whole stem, the legacy ``except`` path
        ("000123456-1800.tar.gz", "1800"),  # ``split('.')[0]`` stops at 1800, not 1800.tar
        ("a-b-c.xml", "b"),  # ``split('-')[1]`` is the segment between the first two hyphens
        ("x.XML", "x"),
    ],
)
def test_output_stem_never_raises_where_the_legacy_split_did(name: str, expected: str) -> None:
    """The ported ``gt_list[index].split('-')[1].split('.')[0]``, including its ``except`` branch.
    An earlier port used ``Path.stem`` with ``partition("-")``, which answered ``b-c`` and
    ``1800.tar`` for the two rows below -- a second file beside the one every legacy dataset
    already has (``generator.py:700``, pinned here as a bug before it was fixed)."""
    assert generator.output_stem(name) == expected


def test_two_files_answering_one_stem_now_get_two_files() -> None:
    """Two documents' page ``0113`` used to write the same PNG: 2455 XMLs became 476 labels.

    The legacy stem is the segment between the first hyphen and the first dot, so a second
    document's page ``0113`` answers it too, and so does a page's ALTO beside its PageXML
    (``882241214-0113.alto.xml`` reads as ``0113``) -- every one of them overwriting the file the
    one before wrote. The first claim keeps the legacy name, so a dataset generated before this
    rule still matches what the next run writes, and the ones behind it take the whole file stem
    and then ``_1``.
    """
    taken: set[str] = set()
    assert generator.unique_output_stem("882241214-0113.xml", taken) == "0113"
    assert generator.unique_output_stem("901234567-0113.xml", taken) == "901234567-0113"
    assert generator.unique_output_stem("882241214-0113.alto.xml", taken) == "882241214-0113.alto"
    assert generator.unique_output_stem("a-b.xml", taken) == "b"
    assert generator.unique_output_stem("a-b.xml", taken) == "a-b"  # the whole file stem
    assert generator.unique_output_stem("a-b.xml", taken) == "a-b_1"  # legacy and full taken
    assert generator.unique_output_stem("000123456.xml", taken) == "000123456"
    assert generator.unique_output_stem("000123456.xml", taken) == "000123456_1"


def test_from_dict_reads_the_legacy_and_the_sbbconfig_spellings() -> None:
    """``custom_config_page2label.json`` keeps the fields at the top level; an ``SBBConfig`` nests
    them under ``page2label``. Without the unwrap, ``sbb gt generate -c configs/layout.json`` read
    an empty mapping and every page came back "no regions labelled"."""
    legacy = generator.GroundTruthConfig.from_dict(
        {"use_case": "layout", "textregions": {"paragraph": 1}, "imageregion": 4}
    )
    assert (legacy.use_case, legacy.textregions, legacy.imageregion) == (
        generator.UseCase.LAYOUT,
        {"paragraph": 1},
        4,
    )
    nested = generator.GroundTruthConfig.from_dict(
        {
            "use_case": "textline",
            "page2label": {"use_case": "layout", "textregions": {"paragraph": 3}},
        }
    )
    assert nested.use_case is generator.UseCase.LAYOUT  # the section wins over the surrounding key
    assert nested.textregions == {"paragraph": 3}


def test_from_dict_normalises_the_integer_tables() -> None:
    """``_int_map`` accepts a mapping or a bare list of names (numbered from 1) and rejects a
    scalar; ``_columns_width_map`` keys by ``int``, which is the spelling ``_column_size`` looks
    up with -- the JSON ``"1"`` used to make every lookup miss."""
    config = generator.GroundTruthConfig.from_dict(
        {"textregions": ["paragraph", "header"], "columns_width": {"1": 112, 2: 30}}
    )
    assert config.textregions == {"paragraph": 1, "header": 2}
    assert config.columns_width == {1: 112, 2: 30}
    with pytest.raises(TypeError, match="expected a mapping of names to class ids, got str"):
        generator.GroundTruthConfig.from_dict({"textregions": "paragraph"})
    with pytest.raises(TypeError, match="expected a mapping of column counts to widths, got list"):
        generator.GroundTruthConfig.from_dict({"columns_width": [112, 30]})
    empty = generator.GroundTruthConfig.from_dict({})
    assert (empty.textregions, empty.graphicregions, empty.columns_width) == ({}, {}, {})
    assert (empty.imageregion, empty.tableregion, empty.artificial_class_label) == (
        None,
        None,
        None,
    )


def test_from_dict_reads_type_output_instead_of_defaulting_to_three_channel_rgb() -> None:
    """``type_output`` was in both the JSON schema and ``config.models.OutputType`` while
    ``GroundTruthConfig`` had no such field, so ``sbb gt generate --config`` had nothing to read and
    fell back to its own ``3d`` flag -- and ``sbb train`` then refused the RGB label it wrote. The
    field exists now, and an ``SBBConfig`` file is read through the same unwrap as the region
    tables."""
    assert generator.GroundTruthConfig().type_output is generator.OutputType.THREE_D
    two_d = generator.GroundTruthConfig.from_dict(
        {"use_case": "textline", "page2label": {"use_case": "layout", "type_output": "2D"}}
    )
    assert two_d.type_output is generator.OutputType.TWO_D  # the section wins, as above
    top_level = generator.GroundTruthConfig.from_dict({"use_case": "layout", "type_output": "2d"})
    assert top_level.type_output is generator.OutputType.TWO_D
    with pytest.raises(ValueError, match="unknown output type 'grey'"):
        generator.GroundTruthConfig.from_dict({"type_output": "grey"})


def test_case_label_defaults_to_one_not_to_the_boundary_class() -> None:
    """``case_label`` is the legacy single-class label: omitted means ``1``, and the boundary
    class is a separate field that used to be borrowed as its fallback
    (``generator.py:249``, pinned here as a bug before it was fixed)."""
    assert generator.GroundTruthConfig.from_dict({}).case_label == 1
    assert generator.GroundTruthConfig.from_dict({"case_label": 3}).case_label == 3
    assert generator.GroundTruthConfig.from_dict({"artificial_class_label": 9}).case_label == 1


def test_label_for_under_a_layout_config_covers_every_region_kind(tmp_path: Path) -> None:
    """The ``layout`` lookup table, one assert per kind, plus the ``rest_as_`` normalisation on
    *both* sides: a config keyed ``rest_as_paragraph`` and a page saying ``paragraph`` match, and
    the same config with a ``heading`` region is unlabelled."""
    config = generator.GroundTruthConfig.from_dict(
        {
            "use_case": "layout",
            "textregions": {"rest_as_paragraph": 6},
            "graphicregions": {"decoration": 5},
            "imageregion": 4,
            "tableregion": 8,
        }
    )
    page = standard_page(tmp_path)
    lookup = {r.region_id or r.kind.value: config.label_for(r) for r in page.regions}
    assert lookup == {"r1": 6, "r3": 6, "r2": None, "i1": 4, "printspace": None}
    assert config.label_for(region(parser.RegionKind.TABLE, "t1", TRIANGLE, "table")) == 8
    assert config.label_for(region(parser.RegionKind.GRAPHIC, "g1", TRIANGLE, "decoration")) == 5
    assert config.label_for(region(parser.RegionKind.GRAPHIC, "g2", TRIANGLE, "signature")) is None
    assert config.label_for(region(parser.RegionKind.SEPARATOR, "s1", TRIANGLE)) is None
    # The PrintSpace is only labelled when ``printspace_as_class_in_layout`` names a class id.
    with_ps = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, printspace_as_class_in_layout=7
    )
    printspace = next(r for r in page.regions if r.kind is parser.RegionKind.PRINTSPACE)
    assert with_ps.label_for(printspace) == 7


@pytest.mark.parametrize(
    ("use_case", "kind"),
    [
        (generator.UseCase.TEXTLINE, parser.RegionKind.TEXTLINE),
        (generator.UseCase.WORD, parser.RegionKind.WORD),
        (generator.UseCase.GLYPH, parser.RegionKind.GLYPH),
        (generator.UseCase.PRINTSPACE, parser.RegionKind.PRINTSPACE),
        (generator.UseCase.BINARIZATION, parser.RegionKind.PRINTSPACE),
    ],
)
def test_single_class_use_cases_answer_with_case_label(
    use_case: generator.UseCase, kind: parser.RegionKind
) -> None:
    """``case_label`` is consulted for the four single-class use cases and for binarization -- the
    old code applied it to *every* region whatever the use case, which labelled a page the config
    could not annotate."""
    config = generator.GroundTruthConfig(use_case=use_case, case_label=7)
    assert config.label_for(region(kind, "x", TRIANGLE)) == 7


def test_a_textline_region_under_a_layout_config_is_not_labelled() -> None:
    """The legacy answer to a region family the config does not list was ``KeyError``; here it is
    ``None``, and only the use cases that annotate textlines give them a class."""
    layout = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )
    textline = region(parser.RegionKind.TEXTLINE, "tl", TRIANGLE)
    assert layout.label_for(textline) is None
    assert (
        generator.GroundTruthConfig(
            use_case=generator.UseCase.TEXTLINE, textregions={"paragraph": 1}
        ).label_for(region(parser.RegionKind.TEXT, "r", TRIANGLE, "paragraph"))
        == 1
    )


def test_build_label_map_single_class_equals_the_derived_image(tmp_path: Path) -> None:
    """The textline/word/glyph/printspace branch: every contour painted with ``case_label``."""
    page = standard_page(tmp_path)
    config = generator.GroundTruthConfig(use_case=generator.UseCase.PRINTSPACE, case_label=1)
    built = generator.build_label_map(page, config)
    assert np.array_equal(
        built, render.label_image([(page.printspace_contour, 1)], (60, 60), background=0)
    )
    assert dict(zip(*[c.tolist() for c in np.unique(built, return_counts=True)], strict=True)) == {
        0: 351,
        1: 3249,
    }


def test_build_label_map_single_class_carves_the_boundary_class(tmp_path: Path) -> None:
    """With ``artificial_class_label`` set, the erode/dilate pair of
    :data:`SINGLE_CLASS_BOUNDARY_RATES` runs first and the boundary is written over the regions,
    exactly as the legacy single-class branch ordered it."""
    page = standard_page(tmp_path)
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.PRINTSPACE, case_label=1, artificial_class_label=5
    )
    built = generator.build_label_map(page, config)
    eroded, boundary = geometry.update_region_contours(
        [page.printspace_contour],
        (60, 60),
        erosion_rate=0,
        dilation_rate=2,
        erosion_early=2,
        dilation_early=0,
    )
    expected = render.label_image([(eroded[0], 1)], (60, 60), background=0)
    expected[boundary.astype(bool)] = 5
    assert np.array_equal(built, expected)


def test_the_layout_map_paints_big_over_small_whatever_the_document_order(tmp_path: Path) -> None:
    """The order :func:`_layout_map` documents -- biggest first, so a small region keeps its own
    colour inside a large one -- and not the document order the port used to follow, where a header
    declared before its paragraph was painted first and buried, leaving ``{0, 1}`` on a page whose
    sibling spelling gave ``{0, 1, 2}`` (``generator.py:534``, pinned here as a bug before it was
    fixed)."""
    small_first = _page(
        tmp_path,
        '<TextRegion ID="h1" type="header"><Coords points="4,4 20,4 20,12 4,12"/></TextRegion>'
        '<TextRegion ID="p1" type="paragraph"><Coords points="4,4 58,4 58,58 4,58"/></TextRegion>',
    )
    big_first = _page(
        tmp_path / "b",
        '<TextRegion ID="p1" type="paragraph"><Coords points="4,4 58,4 58,58 4,58"/></TextRegion>'
        '<TextRegion ID="h1" type="header"><Coords points="4,4 20,4 20,12 4,12"/></TextRegion>',
    )
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"header": 2, "paragraph": 1}
    )
    assert set(np.unique(generator.build_label_map(small_first, config)).tolist()) == {0, 1, 2}
    assert set(np.unique(generator.build_label_map(big_first, config)).tolist()) == {0, 1, 2}


def _page(tmp_path: Path, regions: str, **kwargs: object) -> parser.Page:
    """Parse *regions* into a page under ``tmp_path``, creating the directory if asked to."""
    folder = Path(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    return parser.parse_pagexml(
        write_xml(folder, "000123456-1800.xml", xml_document(regions, **kwargs))
    )


def test_build_label_map_is_none_when_the_page_has_nothing_to_label(tmp_path: Path) -> None:
    page = _page(tmp_path, '<ImageRegion ID="i1"><Coords points="4,4 8,4 8,8 4,8"/></ImageRegion>')
    assert generator.build_label_map(page, generator.GroundTruthConfig()) is None


def test_a_pages_alto_is_not_a_second_page(tmp_path: Path) -> None:
    """``X.xml`` and ``X.alto.xml`` name one page, and a folder of both lists one file.

    ALTO ships beside PageXML under its own name: the PageXML is ``X.xml`` (or ``X.page.xml``) and
    the ALTO is ``X.alto.xml`` (or a bare ``X.alto``), and the two describe the same page -- one
    the annotation, the other the OCR markup it was made on.  Listing them as two files makes two
    labels of one page, so :func:`~sbb_pytorch.gt.parser.page_identity` answers ``X`` for all
    three spellings and :func:`~sbb_pytorch.gt.parser.select_page_files` keeps the PageXML.
    """
    assert parser.page_identity(tmp_path / "000123456-1800.xml") == (
        str(tmp_path),
        "000123456-1800",
    )
    assert parser.page_identity(tmp_path / "000123456-1800.alto.xml") == (
        str(tmp_path),
        "000123456-1800",
    )
    assert parser.page_identity(tmp_path / "000123456-1800.alto") == (
        str(tmp_path),
        "000123456-1800",
    )
    assert parser.page_identity(tmp_path / "000123456-1800.PAGE.xml") == (
        str(tmp_path),
        "000123456-1800.PAGE",  # PageXML's other spelling -- never the ALTO's
    )
    # Two children of a parent dataset holding the same page name are two pages, not one.
    assert parser.page_identity(tmp_path / "a" / "X.xml") != parser.page_identity(
        tmp_path / "b" / "X.xml"
    )

    folder = tmp_path / "mixed"
    folder.mkdir()
    write_xml(folder, "000123456-1800.xml", "<PcGts/>")
    write_xml(folder, "000123456-1800.alto.xml", "<PcGts/>")
    assert parser.select_page_files(folder.glob("*.xml")) == [folder / "000123456-1800.xml"]
    # A folder with only the ALTO of a page still yields it -- nothing is dropped for having no twin.
    assert parser.select_page_files([folder / "000123456-1800.alto.xml"]) == [
        folder / "000123456-1800.alto.xml"
    ]


def test_generate_ground_truth_writes_the_stem_named_label(tmp_path: Path) -> None:
    """One XML in, one PNG named ``1800`` out, and the exact ``summary()`` line the CLI prints --
    the call site that used to die on ``vars()`` of a slots dataclass."""
    xml_dir = tmp_path / "xml"
    _page(
        xml_dir,
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>',
    )
    page = parser.parse_pagexml(xml_dir / "000123456-1800.xml")
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )
    out = tmp_path / "out"
    result = generator.generate_ground_truth(config, xml_dir, out)
    assert [p.name for p in result.labels] == ["1800.png"]
    assert (result.images, result.skipped) == ([], {})
    assert len(result) == 1 and result.summary() == "1 labels, 0 images, 0 skipped"
    written = render.read_image(out / "1800.png")
    assert np.array_equal(
        written,
        render.label_map_to_image(
            generator.build_label_map(page, config), generator._palette(config)
        ),
    )


def test_generate_ground_truth_labels_a_page_filed_twice_once(tmp_path: Path) -> None:
    """A page filed as ``X.xml`` and ``X.alto.xml`` writes one label, from the PageXML.

    Every file in the folder used to get a label of its own, so the pair became two -- and the ALTO
    sorts first, so it claimed the legacy stem ``1800`` while the PageXML was pushed to
    ``000123456-1800.png``: one page, two labels, the annotated file renamed by its own OCR
    markup.  ``dedupe_pages=False`` is that answer, kept as the opt-out.
    """
    folder = tmp_path / "xml"
    folder.mkdir()
    body = xml_document(
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>'
    )
    write_xml(folder, "000123456-1800.xml", body)
    write_xml(folder, "000123456-1800.alto.xml", body)
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )

    result = generator.generate_ground_truth(config, folder, tmp_path / "out")
    assert [p.name for p in result.labels] == ["1800.png"]  # the PageXML's, named the legacy way

    both = generator.generate_ground_truth(config, folder, tmp_path / "both", dedupe_pages=False)
    assert sorted(p.name for p in both.labels) == ["000123456-1800.png", "1800.png"]


def test_generate_ground_truth_crops_and_copies_the_source_image(tmp_path: Path) -> None:
    """``printspace=True`` crops the label to the print-space box and applies the same crop to the
    copied photo; a missing photo is skipped quietly, not a ``ValueError`` that aborts the run."""
    xml_dir = tmp_path / "xml"
    page = _page(
        xml_dir,
        '<PrintSpace><Coords points="2,2 58,2 58,58 2,58"/></PrintSpace>'
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>',
    )
    assert page.printspace_bbox == (2, 2, 57, 57)  # the inclusive box both output images are cut to
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )
    images_in = tmp_path / "images"
    images_in.mkdir()
    render.write_image(images_in / "1800.png", np.zeros((60, 60, 3), np.uint8))
    result = generator.generate_ground_truth(
        config,
        xml_dir,
        tmp_path / "out",
        images_dir=images_in,
        images_output_dir=tmp_path / "copied",
    )
    assert [p.name for p in result.images] == ["1800.png"]
    assert render.read_image(result.images[0]).shape == (60, 60, 3)
    cropped = generator.generate_ground_truth(
        config,
        xml_dir,
        tmp_path / "out2",
        images_dir=images_in,
        images_output_dir=tmp_path / "c2",
        printspace=True,
        output_type="2d",
    )
    assert render.read_image(cropped.labels[0]).shape == (57, 57, 3)  # the inclusive 57 x 57 box
    assert render.read_image(cropped.images[0]).shape == (57, 57, 3)


def test_an_image_dir_holding_the_xml_still_finds_the_photo(tmp_path: Path) -> None:
    """Only suffixes in ``storage.images.IMAGE_SUFFIXES`` are candidates. The first version matched
    any file whose stem was the page's, and the sorted listing put ``000123456-1800.xml`` before
    ``1800.png`` -- so ``read_image`` answered ``None`` and the page's own photo went uncopied
    whenever the XML and image directories were the same (``generator.py:683``, pinned here as a
    bug before it was fixed)."""
    page = _page(tmp_path, '<ImageRegion ID="i1"><Coords points="4,4 8,4 8,8 4,8"/></ImageRegion>')
    render.write_image(tmp_path / "1800.png", np.zeros((60, 60, 3), np.uint8))
    config = generator.GroundTruthConfig(use_case=generator.UseCase.LAYOUT, imageregion=4)
    assert generator._find_image(page, tmp_path).name == "1800.png"
    assert generator._copy_source_image(page, config, tmp_path, tmp_path / "copies") == (
        tmp_path / "copies/1800.png"
    )


def test_generate_ground_truth_reports_unusable_pages(tmp_path: Path) -> None:
    """One parse failure and one unlabelled page beside a good one: both land in ``skipped`` with
    their reason instead of aborting the directory the way the commented-out ``try`` did."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(
        xml_dir,
        "000111111-100.xml",
        xml_document(
            '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>'
        ),
    )
    write_xml(
        xml_dir,
        "000222222-200.xml",
        xml_document('<ImageRegion ID="i1"><Coords points="4,4 8,4 8,8 4,8"/></ImageRegion>'),
    )
    write_xml(xml_dir, "000333333-300.xml", "<not-xml")
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )
    result = generator.generate_ground_truth(config, xml_dir, tmp_path / "out")
    assert [p.name for p in result.labels] == ["100.png"]  # only the page the config can label
    reasons = {path.name: reason for path, reason in result.skipped.items()}
    assert reasons["000333333-300.xml"].startswith("parse failed:")
    assert reasons["000222222-200.xml"] == "no regions labelled by config"
    assert result.summary() == "1 labels, 0 images, 2 skipped"


def test_generate_ground_truth_for_the_reading_order_use_case(tmp_path: Path) -> None:
    """The seventh use case writes the pair images instead of a label map, and keeps the *full*
    file stem in their names (``000123456-1800_0.png``) where every other use case writes
    ``1800.png``."""
    page = _page(
        tmp_path,
        '<PrintSpace><Coords points="2,2 58,2 58,58 2,58"/></PrintSpace>'
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>'
        '<TextRegion ID="r2" type="header"><Coords points="4,24 56,24 56,32 4,32"/></TextRegion>'
        '<TextRegion ID="r3" type="paragraph"><Coords points="4,36 30,36 30,52 4,52"/></TextRegion>',
    )
    expected = [f"{page.file_name}_{i}.png" for i in range(6)]
    result = generator.generate_ground_truth(
        {"use_case": "reading_order"}, tmp_path, tmp_path / "out"
    )
    assert sorted(p.name for p in result.labels) == sorted(expected)
    assert result.summary() == "6 labels, 0 images, 0 skipped"


def test_xml_files_and_the_progress_callback_bypass_discovery(tmp_path: Path) -> None:
    """``xml_files=`` is what the CLI's progress bar feeds; ``on_progress`` fires once per file,
    which is how the bar advances without the package importing ``rich``."""
    page = _page(
        tmp_path,
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>',
    )
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}
    )
    seen: list[Path] = []
    result = generator.generate_ground_truth(
        config, tmp_path, tmp_path / "out", xml_files=[page.path], on_progress=seen.append
    )
    assert seen == [page.path]
    assert [p.name for p in result.labels] == ["1800.png"]


def test_a_json_config_names_itself_in_the_skip_reason(tmp_path: Path) -> None:
    """``GroundTruthConfig.parse`` keeps the file it came from, and the skip reason quotes it -- the
    message the legacy tool never gave, having crashed on ``KeyError`` before it could say either."""
    folder = tmp_path / "in"
    folder.mkdir()
    xml = folder / "000123456-1800.xml"
    xml.write_text(
        xml_document('<ImageRegion ID="i1"><Coords points="4,4 8,4 8,8 4,8"/></ImageRegion>')
    )
    config_file = tmp_path / "gt.json"
    config_file.write_text(json.dumps({"use_case": "layout", "textregions": {"paragraph": 1}}))
    config = generator.GroundTruthConfig.from_json(config_file)
    result = generator.generate_ground_truth(config, folder, tmp_path / "out")
    assert result.labels == []
    assert result.skipped[xml] == f"no regions labelled by {config_file}"


def test_column_size_comes_from_the_page_column_count(tmp_path: Path) -> None:
    """``Metadata/Comments`` ``num_col0002`` plus ``columns_width {2: 30}`` resizes the page to
    ``30 x 30``; an unlisted column count yields no resize instead of the legacy ``KeyError``."""
    page = _page(
        tmp_path,
        '<TextRegion ID="r1" type="paragraph"><Coords points="4,4 40,4 40,20 4,20"/></TextRegion>',
        metadata="<Metadata><Comments>num_col0002</Comments></Metadata>",
    )
    assert generator._column_size(
        page,
        generator.GroundTruthConfig(
            use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}, columns_width={2: 30}
        ),
    ) == (30, 30)
    assert (
        generator._column_size(
            page,
            generator.GroundTruthConfig(use_case=generator.UseCase.LAYOUT, columns_width={1: 30}),
        )
        is None
    )
    out = tmp_path / "out"
    result = generator.generate_ground_truth(
        generator.GroundTruthConfig(
            use_case=generator.UseCase.LAYOUT, textregions={"paragraph": 1}, columns_width={2: 30}
        ),
        tmp_path,
        out,
    )
    assert render.read_image(result.labels[0]).shape == (30, 30, 3)


def test_palette_pins_the_boundary_and_printspace_colours(tmp_path: Path) -> None:
    """The legacy code read ``labels_rgb_color`` for the configured classes but hardcoded two
    colours; merging them into one mapping keeps those and stops ``IndexError`` past id 18."""
    base = dict(enumerate(render.DEFAULT_LABEL_PALETTE))
    config = generator.GroundTruthConfig(
        use_case=generator.UseCase.LAYOUT,
        printspace_as_class_in_layout=7,
        artificial_class_label=9,
    )
    palette = generator._palette(config)
    assert palette[7] == render.PRINTSPACE_CLASS_COLOR
    assert palette[9] == render.ARTIFICIAL_CLASS_COLOR
    assert {k: v for k, v in palette.items() if k not in (7, 9)} == {
        k: v for k, v in base.items() if k not in (7, 9)
    }
    plain = generator._palette(generator.GroundTruthConfig())
    assert plain == base  # 19 entries, nothing overridden, so id 20 resolves by wrapping


def test_labelled_contours_pairs_every_contour_with_one_label() -> None:
    assert generator.labelled_contours([RECT, TRIANGLE], np.int64(5)) == [(RECT, 5), (TRIANGLE, 5)]
    assert generator.labelled_contours([], 5) == []


# ---------------------------------------------------------------------------------- vocabulary


def test_the_table_covers_every_kind_and_every_kind_recognises_its_own_element() -> None:
    """``VOCABULARY``, ``REGION_TAGS`` and ``element_for`` are one mapping checked both ways.

    The parser kept the tags it would read and the writer kept its own table of elements per kind
    (``writer._REGION_ELEMENTS``), and they agreed only because both had been copied from the kind
    enum. Every kind has a row, every row names an element the schema has, and reading that element
    back -- as a tag or as a class name -- gives the same kind.
    """
    assert set(vocabulary.VOCABULARY) == set(vocabulary.REGION_TAGS.values())
    for kind, row in vocabulary.VOCABULARY.items():
        # Every element is either one of the fifteen a <Page> may hold (plus Border and PrintSpace,
        # which are its own slots) or one of the three that nest inside a region.
        assert row.element in vocabulary.PAGE_ELEMENT_NAMES or row.element in {
            "TextLine",
            "Word",
            "Glyph",
        }
        assert vocabulary.kind_for_element(vocabulary.element_for(kind)) is kind
        assert vocabulary.kind_for_class_name(vocabulary.element_for(kind)) is kind
        assert vocabulary.element_for(kind) == row.element


def test_type_for_answers_only_with_values_the_element_in_question_has() -> None:
    """``@type`` is per element, not global: 18 values on ``TextRegion``, 11 on ``GraphicRegion``.

    The rest of the fifteen have no ``@type`` at all, and the two enumerations are closed -- so
    ``"class_1"``, ``"header, heading"`` and the ``rest_as_`` spelling all come back empty from the
    element that cannot say them, which is what stopped the writer inventing a ``@type`` for an
    ``ImageRegion``.
    """
    assert vocabulary.type_for(RegionKind.TEXT, "header") == "header"
    assert vocabulary.type_for(RegionKind.TEXT, "TOC-entry") == "TOC-entry"
    assert vocabulary.type_for(RegionKind.TEXT, "rest_as_paragraph") == "paragraph"
    assert vocabulary.type_for(RegionKind.TEXT, "class_1") == ""  # not one of the eighteen
    assert vocabulary.type_for(RegionKind.TEXT, "") == ""  # no name, no type, whatever the element
    assert vocabulary.type_for(RegionKind.IMAGE, "header") == ""  # ImageRegion has no @type at all
    assert vocabulary.type_for(RegionKind.GRAPHIC, "decoration") == "decoration"
    assert vocabulary.type_for(RegionKind.TEXTLINE, "header") == ""  # nor does a TextLine
    # The two tuples the parser still re-exports are these rows, and nothing else.
    assert set(vocabulary.TEXT_REGION_TYPES) == vocabulary.VOCABULARY[RegionKind.TEXT].types
    assert set(vocabulary.GRAPHIC_REGION_TYPES) == vocabulary.VOCABULARY[RegionKind.GRAPHIC].types
    assert len(vocabulary.TEXT_REGION_TYPES) == 18  # the schema's TextTypeSimpleType, all of them
    assert len(vocabulary.GRAPHIC_REGION_TYPES) == 11  # and GraphicsTypeSimpleType


def test_custom_for_writes_the_dialect_the_annotation_tools_use() -> None:
    """``structure {type:<name>;}`` -- the only spelling of ``@custom`` this package writes.

    A line is ``text_line`` whatever its block was (``custom_from_class`` is off for it), a block
    with no class is ``text_zone``, and a kind with neither writes nothing.
    """
    assert vocabulary.custom_for(RegionKind.TEXT, "header") == "structure {type:header;}"
    assert vocabulary.custom_for(RegionKind.TEXT) == "structure {type:text_zone;}"
    assert vocabulary.custom_for(RegionKind.TEXTLINE) == "structure {type:text_line;}"
    # A line never carries its block's class: the element already says what it is.
    assert vocabulary.custom_for(RegionKind.TEXTLINE, "header") == "structure {type:text_line;}"
    assert vocabulary.custom_for(RegionKind.IMAGE) == ""  # nothing to say, so no attribute
    # ``class_1`` is the placeholder a number gets when no table names it, and Aletheia rejects a
    # class it has never heard of, so it counts as no class: the block is named ``text_zone`` the
    # way escriptorium names every block it has no class for, and a kind with no word of its own
    # still writes nothing.
    assert vocabulary.custom_for(RegionKind.TEXT, "class_1") == "structure {type:text_zone;}"
    assert vocabulary.custom_for(RegionKind.TABLE, "class_3") == ""


def test_kind_for_class_name_reads_every_way_a_config_names_a_class() -> None:
    """One lookup for the three spellings a ``Page2Label`` uses: an element name, its bracketed
    singleton form, and any ``@type`` value an element declares."""
    assert vocabulary.kind_for_class_name("header") is RegionKind.TEXT
    assert vocabulary.kind_for_class_name("decoration") is RegionKind.GRAPHIC
    assert vocabulary.kind_for_class_name("<imageregion>") is RegionKind.IMAGE
    assert vocabulary.kind_for_class_name("ImageRegion") is RegionKind.IMAGE
    assert vocabulary.kind_for_class_name("imageRegion") is RegionKind.IMAGE  # any casing
    assert vocabulary.kind_for_class_name("class_1") is None  # names nothing the schema has
    assert vocabulary.kind_for_element("ReadingOrder") is None  # not a region at all


def test_ancestor_chain_is_the_parents_the_schema_wants_and_no_others() -> None:
    """``Glyph`` -> ``Word`` -> ``TextLine`` -> ``TextRegion``, and nothing above a region.

    The chain the writer builds around a nested-only orphan, straight from the same ``parent`` fields
    the parser's ``_ALLOWED_CHILDREN`` used -- one table says where a region may sit, and this reads
    it back as the ancestors it needs to sit in.
    """
    assert vocabulary.ancestor_chain(RegionKind.GLYPH) == (
        RegionKind.WORD,
        RegionKind.TEXTLINE,
        RegionKind.TEXT,
    )
    assert vocabulary.ancestor_chain(RegionKind.WORD) == (RegionKind.TEXTLINE, RegionKind.TEXT)
    assert vocabulary.ancestor_chain(RegionKind.TEXTLINE) == (RegionKind.TEXT,)
    for kind in (
        RegionKind.TEXT,
        RegionKind.IMAGE,
        RegionKind.GRAPHIC,
        RegionKind.TABLE,
        RegionKind.SEPARATOR,
        RegionKind.NOISE,
        RegionKind.PRINTSPACE,
        RegionKind.BORDER,
    ):
        assert vocabulary.ancestor_chain(kind) == ()


# ------------------------------------------------------------------------------------- writer


def test_a_parsed_page_written_and_parsed_again_is_the_same_page(tmp_path: Path) -> None:
    """``parse_pagexml(write_pagexml(page))`` gives back the page it was given, field for field.

    The shared :data:`PAGE_XML` is the fixture the rest of the file shares -- print space, three text
    regions of two types, one image region, and a reading order deliberately *not* in document order
    -- so one call exercises every element the writer has to reproduce, including the two it derives
    rather than stores (``printspace_bbox`` from the contour, ``region_index`` from the
    ``RegionRefIndexed`` elements) and the ``@ID`` spelling the fixture uses where the schema now
    spells it ``@id``.
    """
    page = standard_page(tmp_path)
    again = parser.parse_pagexml(writer.write_pagexml(page, tmp_path / f"{page.file_name}.xml"))

    assert again.file_name == page.file_name  # @imageFilename comes from the page, not the new path
    assert (again.height, again.width) == (page.height, page.width)
    assert again.columns_width == page.columns_width
    assert again.printspace_contour.tolist() == page.printspace_contour.tolist()
    assert again.printspace_bbox == page.printspace_bbox
    assert again.region_index == page.region_index  # the order survives, unsorted as it is
    # ``@custom`` is the one field the writer adds rather than echoes -- a region parsed with a
    # ``@type`` and no ``@custom`` gains ``structure {type:<name>;}`` on the way out, which is what
    # :func:`sbb_pytorch.gt.vocabulary.custom_for` is for -- so it is asserted where it is decided
    # (``test_a_type_the_element_cannot_say_is_written_in_custom_instead``) and not compared here.
    assert [
        (r.kind, r.type_name, r.region_id, r.contour.tolist(), r.text) for r in again.regions
    ] == [(r.kind, r.type_name, r.region_id, r.contour.tolist(), r.text) for r in page.regions]
    # The nesting too: the fixture's lines are children of their regions, and the page that comes
    # back says so with the same tree -- not as siblings, which is what the flat writer wrote.
    assert [(r.region_id, [c.region_id for c in r.children]) for r in again.regions] == [
        (r.region_id, [c.region_id for c in r.children]) for r in page.regions
    ]
    # The three Metadata elements the schema requires, which this fixture has none of and the old
    # writer never wrote at all: the page's own values when it has them (pinned by
    # ``test_a_parsed_page_keeps_the_author_and_stamps_a_rewrite_would_erase``), and the write's own
    # instant -- :func:`~sbb_pytorch.gt.writer.timestamp_now`'s shape, in both slots -- when it does
    # not, which is the case here. The epoch is gone: a viewer shows it as a date, and it lied.
    assert again.creator == writer.CREATOR
    assert again.created is not None and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", again.created
    )
    assert again.created == again.last_change  # LastChange follows Created when both are new


def qname(name: str) -> str:
    """``"Page"`` -> ``"{http://schema.primaresearch.org/...}Page"`` -- the tag ElementTree wants.

    Every element in an XPath has to be qualified on its own (``Page/TextRegion`` looks for a
    *namespace-less* ``TextRegion`` and finds none), which is why the tests below build their paths
    one qualified name at a time.
    """
    return f"{{{writer.PAGE_NAMESPACE}}}{name}"


def _names(element: ET.Element) -> list[str]:
    """The local names of *element*'s children, in the order they were written.

    The check the schema actually makes: not that the right regions exist, but that they are the
    right children of the right parent, in the order its content model lists them.
    """
    return [child.tag.rsplit("}", 1)[-1] for child in element]


def test_a_textline_is_written_inside_the_block_that_holds_it(tmp_path: Path) -> None:
    """The nested fixture comes back nested, which is the whole reason this writer exists.

    The page the user annotated holds one ``TextRegion`` with thirty-eight ``TextLine``\\ s in it, and
    the writer that preceded this one emitted them side by side -- ``<Page><TextRegion/><TextLine/>…``
    -- because ``Page`` had no field the nesting could live in. The schema allows a ``TextLine`` only
    inside a ``TextRegion``, so ``xmllint`` rejected the file; the check here is the shape of the
    written tree, element by element, in the order ``TextRegionType`` and ``TextLineType`` list.
    """
    page = nested_page(tmp_path)
    root = ET.fromstring(writer.to_pagexml(page))
    page_element = root.find(f"{{{writer.PAGE_NAMESPACE}}}Page")

    assert _names(page_element) == ["TextRegion"]  # one region, not a region plus three loose lines
    block = page_element[0]
    assert block.get("id") == "block"
    assert block.get("custom") == "structure {type:text_zone;}"
    assert _names(block) == ["Coords", "TextLine", "TextLine", "TextLine"]

    line = block[1]  # l1, the one fixture line with a baseline and a transcription
    assert line.get("id") == "l1"
    assert line.get("custom") == "structure {type:text_line;}"
    assert _names(line) == ["Coords", "Baseline", "TextEquiv"]  # TextLineType's own order
    equiv = line.find(f"{{{writer.PAGE_NAMESPACE}}}TextEquiv")
    assert (
        equiv.get("conf") == "0.9999"
    )  # the confidence the parse read, on the element it came from
    assert equiv.find(f"{{{writer.PAGE_NAMESPACE}}}Unicode").text == "first"
    assert _names(block[2]) == ["Coords", "TextEquiv"]  # l2: a baseline only when there is one
    assert _names(block[3]) == ["Coords"]  # l3: no text, so no empty <TextEquiv/> either


def test_a_region_with_no_id_gets_one_the_next_read_keeps(tmp_path: Path) -> None:
    """``@id`` is required, so an empty one is filled -- and filled with something stable.

    v3 issued ``uuid.uuid4().hex[:8]`` per region per write (``inference.py:766-767``), which renamed
    every region on every save. ``<Element>_<n>``, counted per element name in document order, is the
    same shape and is the same answer every time: the ids appear in the file, the next read finds
    them, and writing what was read is byte-identical.
    """
    block = parser.Region(kind="textregion", type_name="", region_id="", contour=RECT)
    line = parser.Region(kind="textline", type_name="", region_id="", contour=RECT)
    line.parent = block
    block.children.append(line)
    page = parser.Page(
        path=tmp_path / "p.xml", file_name="p", height=40, width=40, regions=[block, line]
    )
    text = writer.to_pagexml(page)
    page_element = ET.fromstring(text).find(f"{{{writer.PAGE_NAMESPACE}}}Page")

    assert [element.get("id") for element in page_element] == ["TextRegion_1"]
    assert page_element[0].find(f"{{{writer.PAGE_NAMESPACE}}}TextLine").get("id") == "TextLine_1"
    # The wrapper's id is its own, not the line's: ``xs:ID`` values are unique in the document, and
    # a region and the line inside it sharing one is what xmllint calls "not a valid value".
    assert (
        writer.to_pagexml(parser.parse_pagexml(writer.write_pagexml(page, tmp_path / "p.xml")))
        == text
    )


def test_ids_are_numbered_per_element_name_and_not_across_the_page(tmp_path: Path) -> None:
    """Two nameless text regions and one image region are ``TextRegion_1``, ``TextRegion_2``,
    ``ImageRegion_1`` -- a page-wide counter would renumber the second file's regions by reading the first."""
    regions = [
        parser.Region(kind="textregion", type_name="", region_id="", contour=RECT),
        parser.Region(kind="textregion", type_name="", region_id="", contour=TRIANGLE),
        parser.Region(kind="imageregion", type_name="", region_id="", contour=STRIP),
        parser.Region(kind="textregion", type_name="header", region_id="named", contour=RECT),
    ]
    page = parser.Page(path=tmp_path / "p.xml", file_name="p", height=40, width=40, regions=regions)
    page_element = ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Page")

    assert [element.get("id") for element in page_element] == [
        "TextRegion_1",
        "TextRegion_2",
        "ImageRegion_1",
        "named",  # a parsed id is used verbatim, and is not counted as the next free number
    ]


def test_the_three_required_metadata_elements_and_nothing_the_page_has_no_value_for() -> None:
    """``Creator``, ``Created``, ``LastChange`` are required; the old writer wrote one of the three.

    A page with no author and no timestamps gets :data:`~sbb_pytorch.gt.writer.CREATOR` and
    :func:`~sbb_pytorch.gt.writer.timestamp_now` -- the write's own instant, in the ``xs:dateTime``
    shape the schema types both elements as, and ``--created`` replaces it -- while ``@externalRef``
    and ``<Comments>`` stay off the file when the page has no value for them, because an optional
    attribute written empty is a different answer from one not written.
    """
    root = ET.fromstring(
        writer.to_pagexml(parser.Page(path=Path("p.xml"), file_name="p", height=10, width=10))
    )
    metadata = root.find(qname("Metadata"))
    assert _names(metadata) == ["Creator", "Created", "LastChange"]
    assert metadata.get("externalRef") is None
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}Creator").text == writer.CREATOR
    created = metadata.find(f"{{{writer.PAGE_NAMESPACE}}}Created").text
    assert created is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created)
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}LastChange").text == (
        created  # LastChange follows Created when the page states neither
    )
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}Comments") is None  # no columns, no marker


def test_a_parsed_page_keeps_the_author_and_stamps_a_rewrite_would_erase(tmp_path: Path) -> None:
    """``Creator``/``Created``/``LastChange``/``@externalRef`` survive a rewrite, unchanged.

    The parser used to discard all four, so every ``sbb infer --xml-file`` rewrote escriptorium's
    author as the tool's own name and dropped the archive path beside it.
    """
    source = write_xml(
        tmp_path,
        "stamped.xml",
        xml_document(
            "",
            metadata=(
                '<Metadata externalRef="/archive/stamped.jpg">'
                "<Creator>SBB_QURATOR</Creator>"
                "<Created>2020-01-02T03:04:05</Created>"
                "<LastChange>2021-06-07T08:09:10</LastChange>"
                "<Comments>#num_col0456</Comments></Metadata>"
            ),
        ),
    )
    page = parser.parse_pagexml(source)
    assert (page.external_ref, page.creator, page.created, page.last_change) == (
        "/archive/stamped.jpg",
        "SBB_QURATOR",
        "2020-01-02T03:04:05",
        "2021-06-07T08:09:10",
    )
    metadata = ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Metadata")
    assert metadata.get("externalRef") == "/archive/stamped.jpg"
    assert _names(metadata) == ["Creator", "Created", "LastChange", "Comments"]
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}Created").text == "2020-01-02T03:04:05"
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}LastChange").text == "2021-06-07T08:09:10"
    assert metadata.find(f"{{{writer.PAGE_NAMESPACE}}}Comments").text == "num_col456"
    # The fixture's own value is ``#num_col0456``; the parser reads the number and the writer writes
    # it back without the padding zero, which is the columns-width rule ``_columns_width`` documents.


def test_page_children_are_written_in_the_order_the_schema_lists_them(tmp_path: Path) -> None:
    """``Border``, ``PrintSpace``, ``ReadingOrder``, then the regions -- the order ``PageType`` gives.

    The writer that preceded this one emitted the regions first and the reading order last, which is
    where no version of the schema puts it; the fixtures all use the order asserted here.
    """
    page = parser.Page(
        path=tmp_path / "p.xml",
        file_name="p",
        height=40,
        width=40,
        regions=[
            parser.Region(kind="textregion", type_name="header", region_id="r1", contour=RECT)
        ],
        printspace_contour=TRIANGLE,
        printspace_bbox=(0, 0, 4, 3),
        region_index={"r1": 3},
    )
    assert _names(
        ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Page")
    ) == [
        "PrintSpace",
        "ReadingOrder",
        "TextRegion",
    ]


def test_border_and_printspace_are_written_as_slots_with_no_attributes_at_all(
    tmp_path: Path,
) -> None:
    """``BorderType`` and ``PrintSpaceType`` declare no attributes -- the old writer wrote an ``@id``.

    Both are ``Page`` sequence slots holding one ``<Coords>`` and nothing else, so the fix is not to
    name them better but to stop naming them: the attribute that every region requires is the one
    these two elements must not have.
    """
    page = parser.Page(
        path=tmp_path / "p.xml",
        file_name="p",
        height=40,
        width=40,
        regions=[parser.Region(kind="border", type_name="", region_id="b1", contour=RECT)],
        printspace_contour=TRIANGLE,
    )
    page_element = ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Page")
    assert _names(page_element) == ["Border", "PrintSpace"]
    for name in ("Border", "PrintSpace"):
        element = page_element.find(f"{{{writer.PAGE_NAMESPACE}}}{name}")
        assert element.attrib == {}
        assert _names(element) == ["Coords"]


def test_an_empty_region_index_writes_no_reading_order(tmp_path: Path) -> None:
    """``OrderedGroupType`` needs at least one indexed child, so an empty order is no order.

    The previous writer emitted ``<ReadingOrder><OrderedGroup/></ReadingOrder>`` -- missing the
    ``@id`` the type requires and holding none of the children it requires -- for every page, whether
    or not it had been given an order to keep.
    """
    bare = parser.Page(path=tmp_path / "p.xml", file_name="p", height=10, width=10)
    assert (
        _names(ET.fromstring(writer.to_pagexml(bare)).find(f"{{{writer.PAGE_NAMESPACE}}}Page"))
        == []
    )
    ordered = parser.Page(
        path=tmp_path / "p.xml",
        file_name="p",
        height=10,
        width=10,
        regions=[
            parser.Region(kind="textregion", type_name="", region_id="a", contour=RECT),
            parser.Region(kind="textregion", type_name="", region_id="b", contour=TRIANGLE),
        ],
        region_index={"b": 7, "a": 2},
    )
    group = ET.fromstring(writer.to_pagexml(ordered)).find(
        f"{qname('Page')}/{qname('ReadingOrder')}/{qname('OrderedGroup')}"
    )
    assert group.get("id") == writer._ORDERED_GROUP_ID  # the id the tool has used since eynollah
    assert [(ref.get("regionRef"), ref.get("index")) for ref in group] == [
        ("a", "2"),
        ("b", "7"),
    ]  # sorted by @index, as the document listed them, not as dict order


def test_a_type_the_element_cannot_say_is_written_in_custom_instead(tmp_path: Path) -> None:
    """``@type`` is a closed enumeration; a name outside it goes to ``@custom``, where any string is.

    ``<TextRegion type="<imageregion>">`` and ``type="header, heading"`` -- both of which this
    package wrote -- are invalid twice over (the value is not one of the eighteen, and four of the
    fifteen elements accept no ``@type`` at all), which is what
    :func:`sbb_pytorch.gt.vocabulary.type_for` and :func:`~sbb_pytorch.gt.vocabulary.custom_for`
    settle for every element at once.
    """
    page = parser.Page(
        path=tmp_path / "p.xml",
        file_name="p",
        height=40,
        width=40,
        regions=[
            parser.Region(kind="textregion", type_name="header", region_id="t1", contour=RECT),
            parser.Region(kind="textregion", type_name="class_1", region_id="t2", contour=TRIANGLE),
            parser.Region(kind="imageregion", type_name="class_2", region_id="i1", contour=STRIP),
        ],
    )
    page_element = ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Page")
    text_a = page_element.find(f"{{{writer.PAGE_NAMESPACE}}}TextRegion[@id='t1']")
    text_b = page_element.find(f"{{{writer.PAGE_NAMESPACE}}}TextRegion[@id='t2']")
    image = page_element.find(f"{{{writer.PAGE_NAMESPACE}}}ImageRegion")
    assert text_a.get("type") == "header"  # a name in the enumeration is written as @type
    assert text_a.get("custom") == "structure {type:header;}"
    # ``class_1`` is not in it, and it is not a class either: the placeholder counts as no class, so
    # @type stays empty (nothing to say) and @custom says the element's own word.
    assert text_b.get("type") is None
    assert text_b.get("custom") == "structure {type:text_zone;}"
    assert image.get("type") is None  # ImageRegion has no @type attribute to put a name in...
    assert image.get("custom") is None  # ...and with no class it has no @custom either


def test_a_loose_textline_is_wrapped_in_the_region_the_schema_wants(tmp_path: Path) -> None:
    """A ``TextLine`` with no parent becomes a ``TextRegion`` holding it, not an orphan at page level.

    This is what the writer used to emit -- lines as ``Page`` children -- and what the schema refuses;
    the alternative to wrapping is dropping the line, and a writer that loses the prediction it was
    handed is worse than one that invents the region around it (whose contour is the line's own, so
    nothing about the page's geometry is invented at all).
    """
    page = parser.Page(
        path=tmp_path / "p.xml",
        file_name="p",
        height=40,
        width=40,
        regions=[parser.Region(kind="textline", type_name="", region_id="l1", contour=RECT)],
    )
    page_element = ET.fromstring(writer.to_pagexml(page)).find(f"{{{writer.PAGE_NAMESPACE}}}Page")
    assert _names(page_element) == ["TextRegion"]
    wrapper = page_element[0]
    assert wrapper.get("id") == "TextRegion_1"  # its own id, not the line's (xs:ID must be unique)
    assert wrapper.get("custom") == "structure {type:text_zone;}"  # a region, not a labelled line
    assert _names(wrapper) == ["Coords", "TextLine"]
    assert wrapper[1].get("id") == "l1"
    assert wrapper[1].find(f"{{{writer.PAGE_NAMESPACE}}}Coords") is not None


@pytest.mark.parametrize(
    ("name", "document"),
    [
        ("000123456-1800", PAGE_XML),
        (
            "widths",
            xml_document(
                '<TextRegion ID="a" type="header">'
                '<Coords points="0,0 8,0 8,8 0,8"/>'
                "<TextEquiv><Unicode>Hallo</Unicode></TextEquiv></TextRegion>",
                width=40,
                height=30,
                metadata="<Metadata><Comments>#num_col0456, num_col1500</Comments></Metadata>",
            ),
        ),
        ("empty", xml_document("", width=10, height=10)),
        (
            "noprefix",
            xml_document(
                '<TextRegion ID="a"><Coords points="0,0 8,0 8,8 0,8"/></TextRegion>',
                width=10,
                height=10,
                namespace="",
            ),
        ),
        (
            "pointchildren",
            xml_document(
                '<TextRegion ID="p"><Point x="1.5" y="2"/><Point x="3" y="4.9"/>'
                '<Point x="5" y="6"/></TextRegion>',
                width=9,
                height=9,
            ),
        ),
    ],
)
def test_a_page_written_twice_is_the_same_bytes(tmp_path: Path, name: str, document: str) -> None:
    """Byte-identical output, and the written file as a fixed point of the round trip.

    The v3 generator could not make this promise: it stamped ``datetime.now`` into
    ``Created``/``LastChange`` (``inference.py:791-795``) and named every region from
    ``uuid.uuid4()`` (``inference.py:766-767``), so two writes of one page were two files and the
    ids the parser had read were gone. Nothing here reads a clock or a generator, so the second
    write -- and the write of the page read back from the first -- is the same text, which is what
    lets :command:`sbb infer --xml-file` annotate a page without churning it.
    """
    page = parser.parse_pagexml(write_xml(tmp_path, f"{name}.xml", document))
    text = writer.to_pagexml(page)

    assert writer.to_pagexml(page) == text  # no timestamp, no uuid: the page decides the bytes
    assert writer.write_pagexml(page, tmp_path / f"{name}.xml").read_text() == text
    assert writer.to_pagexml(parser.parse_pagexml(tmp_path / f"{name}.xml")) == text


def test_points_attribute_is_the_string_both_readers_expect() -> None:
    """``"x,y x,y ..."`` -- ``inference.py:842-844`` and the inverse of ``parser.py:159-170``."""
    assert writer.points_attribute(RECT) == "10,20 30,20 30,35 10,35"
    assert writer.points_attribute(TRIANGLE) == "0,0 4,0 0,3"
    # Coordinates are ints by the time the parser stores them (``parser.py:146-156`` truncates the
    # fractional ones), so widening the dtype here cannot change a single byte of the output.
    assert writer.points_attribute(np.asarray(RECT, dtype=np.float64)) == "10,20 30,20 30,35 10,35"
    with pytest.raises(ValueError, match="empty contour"):
        writer.points_attribute(np.empty((0, 2), dtype=np.int32))


def test_write_pagexml_keeps_a_name_that_says_xml_and_otherwise_adds_it(tmp_path: Path) -> None:
    """``writer.py`` follows ``eynollah/writer.py:34`` (``<stem>.xml``) for a nameless target.

    The two upstream writers disagreed -- eynollah assembled ``<stem>.xml`` from the page, v3 wrote
    the path the flag held (``inference.py:835-837``) -- so a path that already names a file is used
    as given and anything else gains the extension, which is also what stops ``--xml-file out`` from
    writing an extension-less file, and what lets it name a directory.
    """
    page = standard_page(tmp_path)
    assert writer.write_pagexml(page, tmp_path / "copy") == tmp_path / "copy.xml"
    assert writer.write_pagexml(page, tmp_path / "deep/nested") == tmp_path / "deep/nested.xml"
    named = tmp_path / "explicit.xml"
    assert writer.write_pagexml(page, named) == named
    assert named.read_text() == writer.to_pagexml(page)
