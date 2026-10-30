"""The PageXML vocabulary -- every element name and ``@type`` value the schema allows, in one table.

Three places used to spell these names, and they disagreed with each other and with the schema.
:mod:`.parser` kept the tags it would read (``_REGION_TAGS``) plus the two ``@type`` lists the
legacy readers had inherited -- ten text types and four graphic ones, where the schema has
eighteen and eleven.  :mod:`.writer` kept its own element-per-kind table (``_REGION_ELEMENTS``).
And ``cli._class_name`` invented a third set by joining *every* name a class id stood for, so a
class covering both ``header`` and ``heading`` was written as ``@type="header, heading"`` -- a
value no version of this schema has ever accepted, and one of the reasons the PAGE-XML
:command:`sbb infer` produced could not be opened.

The authority for everything in this module is ``pagecontent/2019-07-15``, the schema the writer
targets and the one :data:`~.writer.PAGE_NAMESPACE` names: the element names and the two ``@type``
enumerations are copied from it and nowhere else.  The one exception is the *legacy* spellings
:data:`LEGACY_INACTIVE_TEXT_TYPES` and the ``rest_as_`` rule in :func:`normalise_text_type`, which
are what the old tools wrote and therefore what this package still has to read.

Nothing here reads a file: this is the vocabulary, and :mod:`.parser` (which reads) and
:mod:`.writer` (which writes) both consult it, so an element name is decided in one place and
never again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

__all__ = [
    "GRAPHIC_REGION_TYPES",
    "LEGACY_INACTIVE_TEXT_TYPES",
    "PAGE_ELEMENT_NAMES",
    "REGION_TAGS",
    "TEXT_REGION_TYPES",
    "VOCABULARY",
    "RegionKind",
    "RegionVocabulary",
    "ancestor_chain",
    "custom_for",
    "element_for",
    "kind_for_class_name",
    "kind_for_element",
    "normalise_text_type",
    "type_for",
]


class RegionKind(str, Enum):
    """Region kinds, keyed by the lower-cased XML local name.

    Only the elements :mod:`.parser` models are members: the schema's page-level choice has fifteen
    names (``ChartRegion``, ``MapRegion``, ``MathsRegion`` and the rest) and this package reads six
    of them, plus ``TextLine``, ``Word``, ``Glyph``, ``PrintSpace`` and ``Border``.  The full set of
    names is :data:`PAGE_ELEMENT_NAMES`; the subset with a geometry this package can round-trip is
    what a :class:`~.parser.Region` can be.
    """

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


#: Canonical PageXML ``TextRegion/@type`` values -- all eighteen of the schema's
#: ``TextTypeSimpleType`` enumeration.  The legacy readers knew ten of them, which is why a file
#: annotated with ``caption``, ``footer``, ``credit`` or ``TOC-entry`` had its regions read as
#: untyped by this package and written back with a ``@type`` the schema rejects.
TEXT_REGION_TYPES: Final[tuple[str, ...]] = (
    "paragraph",
    "heading",
    "caption",
    "header",
    "footer",
    "page-number",
    "drop-capital",
    "credit",
    "floating",
    "signature-mark",
    "catch-word",
    "marginalia",
    "footnote",
    "footnote-continued",
    "endnote",
    "TOC-entry",
    "list-label",
    "other",
)

#: Canonical PageXML ``GraphicRegion/@type`` values -- all eleven ``GraphicsTypeSimpleType``
#: enumerations.  The four the legacy readers knew (``decoration``, ``handwritten-annotation``,
#: ``stamp``, ``signature``) are among them, so nothing that used to read stops reading.
GRAPHIC_REGION_TYPES: Final[tuple[str, ...]] = (
    "logo",
    "letterhead",
    "decoration",
    "frame",
    "handwritten-annotation",
    "stamp",
    "signature",
    "barcode",
    "paper-grow",
    "punch-hole",
    "other",
)

#: ``read_xml`` has its ``catch-word`` and ``page-number`` branches commented out,
#: so regions of those types were routed to ``paragraph``.  The layout visualiser
#: does collect them under their own type.  The distinction is preserved here
#: rather than silently unified.
LEGACY_INACTIVE_TEXT_TYPES: Final[frozenset[str]] = frozenset({"catch-word", "page-number"})

#: Every element name a ``<Page>`` may hold a region in: the fifteen members of the schema's region
#: choice, plus ``Border`` and ``PrintSpace``, which are separate ``Page`` sequence slots rather than
#: choice members (they carry no ``@id`` and no ``@type`` at all).  ``TextLine``, ``Word`` and
#: ``Glyph`` are deliberately absent -- they nest inside a region and can never be a page's child.
PAGE_ELEMENT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "TextRegion",
        "ImageRegion",
        "LineDrawingRegion",
        "GraphicRegion",
        "TableRegion",
        "ChartRegion",
        "MapRegion",
        "SeparatorRegion",
        "MathsRegion",
        "ChemRegion",
        "MusicRegion",
        "AdvertRegion",
        "NoiseRegion",
        "UnknownRegion",
        "CustomRegion",
        "Border",
        "PrintSpace",
    }
)


@dataclass(frozen=True, slots=True)
class RegionVocabulary:
    """What the schema lets a region of one :class:`RegionKind` say, and where it may appear.

    Attributes
    ----------
    element:
        The XML element name this kind is written as (``RegionKind.TEXT`` -> ``"TextRegion"``).
    types:
        The values ``@type`` may hold -- the schema's enumeration for this element, or ``None``
        when the element has no ``@type`` attribute at all.  Five of the fifteen page-level
        elements have one; ``ImageRegion``, ``SeparatorRegion``, ``TableRegion``, ``NoiseRegion``,
        ``TextLine``, ``Word``, ``Glyph``, ``Border`` and ``PrintSpace`` have none, and writing a
        ``@type`` on any of them makes the file invalid rather than merely odd.
    custom:
        The value ``@custom`` carries when the region's class gives no name of its own --
        ``text_zone`` for a block that has no class, ``text_line`` for a line.  ``""`` writes no
        ``@custom`` at all.
    custom_from_class:
        ``False`` for a kind whose ``@custom`` never names the class, because the element says it
        already: a ``TextLine`` is ``text_line`` whatever its block was.
    page_level:
        ``True`` for the region-choice members, which :class:`~.parser.Page` holds directly.
        ``TextLine``/``Word``/``Glyph`` are ``False`` (they nest), and so are ``PrintSpace`` and
        ``Border``, which live in their own ``Page`` sequence slots.
    parent:
        The kind this one nests in, when it is not page level: ``Word`` -> ``TextLine``.
    id_required:
        ``False`` for ``PrintSpace`` and ``Border`` -- the two elements the schema gives *no*
        attributes at all, so writing an ``@id`` on one is the error, not the cure.
    """

    element: str
    types: frozenset[str] | None = None
    custom: str = ""
    custom_from_class: bool = True
    page_level: bool = True
    parent: RegionKind | None = None
    id_required: bool = True


_TEXT_TYPES = frozenset(TEXT_REGION_TYPES)
_GRAPHIC_TYPES = frozenset(GRAPHIC_REGION_TYPES)

#: The table: one :class:`RegionVocabulary` per :class:`RegionKind`.
VOCABULARY: Final[dict[RegionKind, RegionVocabulary]] = {
    RegionKind.TEXT: RegionVocabulary(element="TextRegion", types=_TEXT_TYPES, custom="text_zone"),
    RegionKind.IMAGE: RegionVocabulary(element="ImageRegion"),
    RegionKind.GRAPHIC: RegionVocabulary(
        element="GraphicRegion", types=_GRAPHIC_TYPES, custom="decoration"
    ),
    RegionKind.SEPARATOR: RegionVocabulary(element="SeparatorRegion"),
    RegionKind.TABLE: RegionVocabulary(element="TableRegion"),
    RegionKind.NOISE: RegionVocabulary(element="NoiseRegion"),
    RegionKind.TEXTLINE: RegionVocabulary(
        element="TextLine",
        custom="text_line",
        custom_from_class=False,  # a line is a line, whatever class its block was
        page_level=False,
        parent=RegionKind.TEXT,
    ),
    RegionKind.WORD: RegionVocabulary(element="Word", page_level=False, parent=RegionKind.TEXTLINE),
    RegionKind.GLYPH: RegionVocabulary(element="Glyph", page_level=False, parent=RegionKind.WORD),
    RegionKind.PRINTSPACE: RegionVocabulary(
        element="PrintSpace", page_level=False, id_required=False
    ),
    RegionKind.BORDER: RegionVocabulary(element="Border", page_level=False, id_required=False),
}

#: The ``_REGION_TAGS`` table of :mod:`.parser` and the ``_REGION_ELEMENTS`` table of
#: :mod:`.writer` were the same mapping written twice, once for reading and once for writing, and
#: they agreed only because both were copied from the same kind enum.  This is the one table
#: (:func:`element_for` writes it, :func:`kind_for_element` reads it); ``parser._REGION_TAGS`` is
#: now an alias of it.
REGION_TAGS: Final[dict[str, RegionKind]] = {
    # The legacy code matched the pairs ``'}TextRegion'`` / ``'}Textregion'`` (and the
    # same for every other kind) because both spellings occur in real files; lower-casing the
    # local name covers both with one key per kind.
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

_KIND_BY_ELEMENT: Final[dict[str, RegionKind]] = dict(REGION_TAGS)


def element_for(kind: RegionKind) -> str:
    """The element name a region of *kind* is written as -- ``TEXT`` -> ``"TextRegion"``."""
    return VOCABULARY[kind].element


def kind_for_element(name: str) -> RegionKind | None:
    """The kind an element name spells, case-insensitively; ``None`` for one this package has no
    kind for (``ChartRegion``, ``MathsRegion``, a bare ``ReadingOrder``, ``Coords``)."""
    return _KIND_BY_ELEMENT.get(name.rsplit("}", 1)[-1].lower())


def kind_for_class_name(name: str) -> RegionKind | None:
    """The kind a *class* name is written as: ``header`` -> TEXT, ``<imageregion>`` -> IMAGE.

    :func:`kind_for_element` answers the question a parser asks (an XML tag it is looking at); this
    answers the question a config asks, and the two spellings overlap only partly.  A
    ``Page2Label`` names its classes by three shapes at once -- the element name (``ImageRegion``),
    the angle-bracket spelling its unnamed singletons get (``<imageregion>``, which
    :meth:`~sbb_pytorch.config.models.Page2Label.all_regions` invents), and the ``@type`` values of
    an element that has one (``header``, ``decoration``, any of the eighteen) -- and a class
    ``header`` is a ``TextRegion``, not a region with no element.

    Answering it from this table is what stops the writer from calling every predicted region a
    ``TextRegion`` -- ``<TextRegion type="<imageregion>">``, invalid twice over, was the old
    output.  ``None`` means the name belongs to no element this package models, and the caller
    chooses (``cli`` keeps ``TextRegion``, the shape the tool has always fallen back to).
    """
    element = name.strip().strip("<>").lower()
    if (kind := _KIND_BY_ELEMENT.get(element)) is not None:
        return kind
    normalised = normalise_text_type(name.strip())
    for kind, vocabulary in VOCABULARY.items():
        if vocabulary.types is not None and normalised in vocabulary.types:
            return kind
    return None


def type_for(kind: RegionKind, name: str = "") -> str:
    """The ``@type`` a region of *kind* whose class is *name* may carry, or ``""`` for none.

    Empty is the answer more often than it looks: five of the fifteen page-level elements have a
    ``@type`` at all, and the enumerations are closed -- ``"header, heading"``, ``"<imageregion>"``
    and ``"rest_as_paragraph"`` (all three of which this package used to write) come back empty and
    are carried in ``@custom`` instead, where any string is legal.
    """
    if not name:
        return ""
    normalised = normalise_text_type(name)
    types = VOCABULARY[kind].types
    if types is None or normalised not in types:
        return ""
    return normalised


def custom_for(kind: RegionKind, name: str = "") -> str:
    """``@custom`` for a region of *kind* whose class is *name*: ``structure {type:paragraph;}``.

    The escriptorium/Aletheia dialect keeps the class in ``@custom`` (on the pages whose ``@type``
    the schema cannot say what they mean, and on the lines and words that have no ``@type`` at all),
    and this is the one spelling of it the package uses.  A kind with no class and no default word
    of its own writes no ``@custom``.
    """
    vocabulary = VOCABULARY[kind]
    labelled = normalise_text_type(name) if name and vocabulary.custom_from_class else ""
    value = labelled or vocabulary.custom
    return f"structure {{type:{value};}}" if value else ""


def ancestor_chain(kind: RegionKind) -> tuple[RegionKind, ...]:
    """The kinds *kind* must sit inside to be legal, nearest first: ``GLYPH`` -> ``(WORD, TEXTLINE,
    TEXT)``.

    Used when a nested-only region turns up with no parent -- a ``TextLine`` written as a child of
    ``<Page>``, which is what :command:`sbb infer` used to produce.  Wrapping it in the chain here
    keeps it, keeps it legal, and invents nothing the schema did not already require.
    """
    chain: list[RegionKind] = []
    seen = {kind}
    current = kind
    while (parent := VOCABULARY[current].parent) is not None and parent not in seen:
        chain.append(parent)
        seen.add(parent)
        current = parent
    return tuple(chain)


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
