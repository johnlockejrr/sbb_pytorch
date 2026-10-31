"""Ground-truth generation: PageXML in, label images out.

Port of ``get_images_of_ground_truth`` (656 lines, ``gt_gen_utils.py:541``) together
with the ``pagexml2label`` command of ``generate_gt_for_training.py``.  The legacy
function was one body with two near-duplicate halves — a single-class branch for the
``textline``/``word``/``glyph``/``printspace`` use cases and a multi-class ``layout``
branch — each re-implementing XML traversal, contour extraction, print-space cropping,
column resizing and the ``fillPoly`` write-out.  Here those halves share every helper
and differ only in how they assemble the label map.

Output naming follows the legacy convention exactly, because existing datasets and the
training loaders depend on it: for a page file named ``000123456-1800.xml`` the label
image and the copied source image are both named ``1800`` (the segment after the first
hyphen), falling back to the whole stem when the name has no hyphen.  The legacy code
reached that with ``gt_list[index].split('-')[1].split('.')[0]`` inside a bare
``except:``; :func:`output_stem` does it with :attr:`pathlib.Path.stem` and no
exception.  ``machine_based_reading_order`` is the exception to this rule and keeps the
full stem, which :mod:`~.reading_order` preserves.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from ..config.models import (
    CASE_LABEL_USE_CASES,
    SINGLE_CLASS_USE_CASES,
    OutputType,
    UseCase,
)
from ..storage.images import IMAGE_SUFFIXES
from .geometry import (
    DEFAULT_MIN_AREA_FRACTION,
    contour_area,
    update_region_contours,
)
from .parser import (
    Page,
    Region,
    RegionKind,
    discover_xml_files,
    normalise_text_type,
    parse_pagexml,
)
from .render import (
    ARTIFICIAL_CLASS_COLOR,
    DEFAULT_LABEL_PALETTE,
    PRINTSPACE_CLASS_COLOR,
    crop_to,
    label_image,
    label_map_to_image,
    read_image,
    to_label_map,
    write_image,
)

__all__ = [
    "BOUNDARY_RATES",
    "SINGLE_CLASS_BOUNDARY_RATES",
    "UseCase",
    "BoundaryRates",
    "GenerationResult",
    "GroundTruthConfig",
    "build_label_map",
    "generate_ground_truth",
    "labelled_contours",
    "output_stem",
]


#: ``UseCase``, :data:`SINGLE_CLASS_USE_CASES` and :data:`CASE_LABEL_USE_CASES` are declared once,
#: in :mod:`sbb_pytorch.config.models`, and re-exported here: the schema and ``gt`` used to ship two
#: copies of the enum with different members.

#: The region kinds those use cases label — everything but ``layout``, where each
#: region kind and type has its own class id.
_SINGLE_CLASS_REGION_KINDS: Final[frozenset[RegionKind]] = frozenset(
    {RegionKind.TEXTLINE, RegionKind.WORD, RegionKind.GLYPH, RegionKind.PRINTSPACE}
)


@dataclass(frozen=True, slots=True)
class BoundaryRates:
    """Morphology rates used to carve the artificial boundary class out of a region.

    The legacy code wrote eleven of these as repeated literal assignments.  They are a
    lookup table here; :data:`BOUNDARY_RATES` holds the values verbatim.
    """

    erosion_rate: int = 0
    dilation_rate: int = 2
    erosion_early: int | None = None
    dilation_early: int | None = None


#: Rates for the single-class use cases (legacy ``erosion_rate = 0 #1``,
#: ``dilation_rate = 2``, ``dilation_early = 0``, ``erosion_early = 2``).
SINGLE_CLASS_BOUNDARY_RATES: Final[BoundaryRates] = BoundaryRates(
    erosion_rate=0, dilation_rate=2, erosion_early=2, dilation_early=0
)

#: Rates per region type for the ``layout`` use case — the eleven blocks at
#: ``gt_gen_utils.py:1006-1051``, kept as data instead of control flow.
#:
#: The legacy blocks each indexed ``co_text[<type>]`` unconditionally once the type
#: appeared in ``artificial_class_on_boundary``, so requesting a boundary for a type the
#: ``textregions`` config did not declare raised ``KeyError`` and aborted the page.  The
#: generator now consults this table only for types it actually found on the page.
BOUNDARY_RATES: Final[dict[str, BoundaryRates]] = {
    "paragraph": BoundaryRates(erosion_rate=2, dilation_rate=4),
    "drop-capital": BoundaryRates(erosion_rate=1, dilation_rate=3),
    "catch-word": BoundaryRates(erosion_rate=0, dilation_rate=3),
    "page-number": BoundaryRates(erosion_rate=0, dilation_rate=3),
    "header": BoundaryRates(erosion_rate=1, dilation_rate=4),
    "heading": BoundaryRates(erosion_rate=1, dilation_rate=4),
    "signature-mark": BoundaryRates(erosion_rate=1, dilation_rate=4),
    "marginalia": BoundaryRates(erosion_rate=2, dilation_rate=4),
    "footnote": BoundaryRates(erosion_rate=0, dilation_rate=2),
    "footnote-continued": BoundaryRates(erosion_rate=0, dilation_rate=2),
    "tableregion": BoundaryRates(erosion_rate=0, dilation_rate=3),
}


@dataclass(frozen=True, slots=True)
class GroundTruthConfig:
    """Page-to-label mapping for one generation run.

    This is the shape of the shipped ``custom_config_page2label.json``.  Every field is
    optional except the ones the requested use case needs, and the loaders never raise on
    a key the way ``config_params['textregions']`` did — that indexing is where the
    legacy tool crashed with ``KeyError`` on configs that omitted a region family, and
    with ``AttributeError: 'NoneType' object has no attribute 'keys'`` when ``--config``
    was omitted entirely despite the CLI help calling it ignorable.

    Attributes
    ----------
    use_case:
        What to label; see :class:`UseCase`.
    textregions:
        ``TextRegion/@type`` -> class id.  ``rest_as_<type>`` keys are honoured (see
        :func:`~.parser.normalise_text_type`).
    graphicregions, imageregion, separatorregion, tableregion, noiseregion:
        The remaining region families and their class ids.
    case_label:
        Class id for the single-class use cases (legacy implicitly ``1``).
    artificial_class_label:
        Class id of the boundary class.  Set it to enable the boundary; the legacy code
        used the key's presence as the switch and its value as the label.
    printspace_as_class_in_layout:
        Class id painted everywhere outside the print space.
    columns_width:
        Column count -> canonical page width, for the column-normalising resize.
    min_area, max_area:
        Region size bounds as fractions of the page area.
    type_output:
        How each label image is written -- ``2d`` (class id per pixel, replicated across the
        three channels) or ``3d`` (RGB from the palette).  The field the schema always had and
        ``sbb gt generate --config`` never read: the command passed its own ``--type-output``
        flag, which defaulted to ``3d``, straight to
        :func:`~sbb_pytorch.gt.generate_ground_truth`, so a config asking for ``2d`` got an RGB
        rendering -- and ``sbb train`` then refused that rendering as a label, because
        :func:`~sbb_pytorch.storage.images.remap_mask` found ``255`` where a class id belongs.
        ``sbb infer --save-layout`` and ``config.models.TypeOutput`` read the same value.
    """

    use_case: UseCase = UseCase.LAYOUT
    type_output: OutputType = OutputType.THREE_D
    textregions: dict[str, int] = field(default_factory=dict)
    graphicregions: dict[str, int] = field(default_factory=dict)
    imageregion: int | None = None
    separatorregion: int | None = None
    tableregion: int | None = None
    noiseregion: int | None = None
    case_label: int = 1
    artificial_class_label: int | None = None
    printspace_as_class_in_layout: int | None = None
    columns_width: dict[int, int] = field(default_factory=dict)
    min_area: float = DEFAULT_MIN_AREA_FRACTION
    max_area: float = 1.0
    source: Path | None = None

    @classmethod
    def parse(cls, config: GroundTruthConfig | str | Path | dict[str, Any]) -> GroundTruthConfig:
        """Accept an instance, a JSON path, or a parsed dict.

        Making the config a single typed object — instead of the legacy pair of
        ``config_file`` *and* ``config_params`` arguments, where only one of them had to
        be non-``None`` for the function to work — is the fix for that class of failure.
        """
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        return cls.from_json(config)

    @classmethod
    def from_json(cls, path: str | Path) -> GroundTruthConfig:
        """Load the config from a JSON file."""
        file = Path(path)
        with file.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), source=file)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Path | None = None) -> GroundTruthConfig:
        """Build a config from a mapping, tolerating absent keys.

        Two file shapes are accepted. The legacy one is ``custom_config_page2label.json``: the
        fields below at the top level. The other is an :class:`~sbb_pytorch.config.models.SBBConfig`,
        which carries the same fields inside its ``page2label`` section -- without this, ``sbb gt
        generate --config configs/layout.json`` parsed an empty mapping and every page came back
        "no regions labelled", because the region tables were nested one level deeper than the
        reader looked. The section wins over the surrounding keys.
        """
        section = data.get("page2label")
        data = {**data, **section} if isinstance(section, dict) else data
        return cls(
            use_case=UseCase.parse(data.get("use_case", UseCase.LAYOUT)),
            # Read from the top level, where an SBBConfig keeps it (the merge above keeps
            # surrounding keys unless the page2label section repeats them). It used to stop here
            # unread, which is why `sbb gt generate --config configs/x.json` wrote a 3d label for
            # a config whose own type_output said 2d.
            type_output=OutputType.parse(data.get("type_output", OutputType.THREE_D)),
            textregions=_int_map(data.get("textregions")),
            graphicregions=_int_map(data.get("graphicregions")),
            imageregion=_int_or_none(data.get("imageregion")),
            separatorregion=_int_or_none(data.get("separatorregion")),
            tableregion=_int_or_none(data.get("tableregion")),
            noiseregion=_int_or_none(data.get("noiseregion")),
            # The legacy single-class label, implicitly 1 in every legacy file that omitted it.
            # It used to fall back to ``artificial_class_label`` first, so a config setting only
            # the boundary class silently reused it as the region class.
            case_label=int(data.get("case_label", 1)),
            artificial_class_label=_int_or_none(data.get("artificial_class_label")),
            printspace_as_class_in_layout=_int_or_none(data.get("printspace_as_class_in_layout")),
            columns_width=_columns_width_map(data.get("columns_width")),
            min_area=float(data.get("min_area", DEFAULT_MIN_AREA_FRACTION)),
            max_area=float(data.get("max_area", 1.0)),
            source=source,
        )

    def label_for(self, region: Region) -> int | None:
        """Class id of *region* under this config, or ``None`` if it is not labelled.

        ``None`` means "this page carries a region family the config does not label",
        which the legacy tool answered with ``KeyError``; the generator skips the region
        and reports it instead.

        ``case_label`` is the class of the *single-class* use cases, so it is consulted
        only for them (and for binarization, whose one class is the print space — see
        :func:`_case_contours`).  Applying it to every textline/word/glyph/printspace
        region whatever the use case — the old behaviour — made a bare ``layout`` config
        label the page's PrintSpace with ``1`` even with ``printspace_as_class_in_layout``
        unset, so a page it could not label produced an image instead of being skipped.
        """
        if self.use_case in CASE_LABEL_USE_CASES and region.kind in _SINGLE_CLASS_REGION_KINDS:
            return self.case_label
        if region.kind in {RegionKind.BORDER, RegionKind.PRINTSPACE}:
            return self.printspace_as_class_in_layout
        table: dict[RegionKind, dict[str, int] | int | None] = {
            RegionKind.TEXT: self.textregions,
            RegionKind.GRAPHIC: self.graphicregions,
            RegionKind.IMAGE: self.imageregion,
            RegionKind.SEPARATOR: self.separatorregion,
            RegionKind.TABLE: self.tableregion,
            RegionKind.NOISE: self.noiseregion,
        }
        # A kind the layout config has no family for — a TextLine under ``layout``, which
        # is annotated by ``textline``, not by the region tables — is unlabelled, whereas
        # the legacy ``config_params[...]`` indexing raised ``KeyError`` on it.
        configured = table.get(region.kind)
        if configured is None:
            return None
        if isinstance(configured, int):
            return configured
        # Normalise both sides. The dict is keyed the way the JSON spells it
        # (``rest_as_paragraph``) while a PageXML region says ``paragraph``, and asking the
        # raw-keyed dict for the normalised name labelled nothing -- which made a real page
        # generate no image at all, because the whole page is skipped when no region matches.
        names = list(configured)
        lookup = {
            normalise_text_type(name, names): class_id for name, class_id in configured.items()
        }
        return lookup.get(normalise_text_type(region.type_name, names))


def _int_map(value: Any) -> dict[str, int]:
    """Normalise a ``{name: id}`` JSON object, whose keys arrive as strings."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key): int(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        # A bare list of names has no ids; number them from 1, as the legacy
        # ``elements_with_artificial_class`` lists effectively did.
        return {str(name): index for index, name in enumerate(value, start=1)}
    raise TypeError(f"expected a mapping of names to class ids, got {type(value).__name__}")


def _columns_width_map(value: Any) -> dict[int, int]:
    """Normalise ``{column count: page width}``, whose keys JSON spells as strings.

    The keys are numbers here, not region names: :func:`_column_size` looks a page's
    column count up as an ``int`` (``parser.Page.columns_width``), so keeping the JSON
    ``"1"`` spelling made every lookup miss and the normalising resize never happen.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {int(key): int(item) for key, item in value.items()}
    raise TypeError(f"expected a mapping of column counts to widths, got {type(value).__name__}")


def _int_or_none(value: Any) -> int | None:
    """``int(value)`` or ``None``, so an absent JSON key stays absent."""
    return None if value is None else int(value)


@dataclass(slots=True)
class GenerationResult:
    """What one :func:`generate_ground_truth` run produced.

    Attributes
    ----------
    labels:
        Label images written.
    images:
        Source images copied (or cropped/resized) into the images output directory.
    skipped:
        ``path -> reason`` for every page that produced no label image.  The legacy code
        wrapped its per-file work in a commented-out ``try``, so one bad page aborted the
        whole directory; failures are collected here instead.
    """

    labels: list[Path] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    skipped: dict[Path, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of label images written."""
        return len(self.labels)

    def summary(self) -> str:
        """One-line count of what happened, for the CLI to print."""
        return f"{len(self.labels)} labels, {len(self.images)} images, {len(self.skipped)} skipped"


def generate_ground_truth(
    config: GroundTruthConfig | str | Path | dict[str, Any],
    xml_dir: str | Path,
    output_dir: str | Path,
    *,
    output_type: OutputType | str = OutputType.THREE_D,
    printspace: bool = False,
    images_dir: str | Path | None = None,
    images_output_dir: str | Path | None = None,
    xml_files: Sequence[Path] | None = None,
    on_progress: Callable[[Path], None] | None = None,
) -> GenerationResult:
    """Generate ground-truth label images for every PageXML file in *xml_dir*.

    Parameters
    ----------
    config:
        The page-to-label config — an instance, a JSON path or a parsed dict.  Required:
        the legacy default of ``config_params=None`` crashed the moment the layout branch
        ran.
    xml_dir:
        Directory of PageXML/ALTO files.
    output_dir:
        Where the label images go.  Each is named by :func:`unique_output_stem`, so two files
        whose legacy stem matches -- two documents' page ``0113``, or a page's PageXML and its
        ALTO -- write two labels instead of one overwriting the other.
    output_type:
        ``3d`` (palette RGB) or ``2d`` (class id per pixel) — see
        :class:`~.render.OutputType`.
    printspace:
        Crop both the label and the copied source image to the print space.
    images_dir, images_output_dir:
        Source images to copy alongside the labels, and where to write them.  Needed for
        the print-space crop and the column resize to be reproducible on the photos.
    xml_files:
        Pre-resolved file list, e.g. from a progress bar; defaults to *xml_dir*'s contents.
    on_progress:
        Called with each file after it is handled, so a caller can drive a progress bar
        without this package depending on ``rich``.

    Returns
    -------
    GenerationResult
        The written paths and the per-file skip reasons.
    """
    resolved = GroundTruthConfig.parse(config)
    kind = OutputType.parse(output_type)
    files = list(xml_files) if xml_files is not None else discover_xml_files(xml_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = GenerationResult()
    taken: set[str] = set()  # stems this run has handed out -- see :func:`unique_output_stem`
    for xml_file in files:
        if on_progress is not None:
            on_progress(xml_file)
        try:
            page = parse_pagexml(xml_file)
        except Exception as exc:  # - one unreadable page must not stop the run.
            result.skipped[Path(xml_file)] = f"parse failed: {exc}"
            continue

        if resolved.use_case is UseCase.READING_ORDER:
            from .reading_order import generate_pairs  # Lazy: only this use case needs it.

            pairs = generate_pairs(page, min_area=resolved.min_area, max_area=resolved.max_area)
            if not pairs:
                result.skipped[page.path] = "no usable text regions"
                continue
            for pair in pairs:
                image_path = write_image(out / f"{pair.name}.png", pair.modal_image)
                result.labels.append(image_path)
            continue

        label_map = build_label_map(page, resolved)
        if label_map is None:
            result.skipped[page.path] = f"no regions labelled by {resolved.source or 'config'}"
            continue

        stem = unique_output_stem(page.path, taken)
        result.labels.append(
            _write_label_image(
                page, label_map, resolved, out, kind, printspace=printspace, stem=stem
            )
        )
        copied = _copy_source_image(
            page, resolved, images_dir, images_output_dir, printspace=printspace, stem=stem
        )
        if copied is not None:
            result.images.append(copied)
    return result


def build_label_map(page: Page, config: GroundTruthConfig) -> np.ndarray | None:
    """The ``(H, W)`` class-id image of *page*, before cropping or resizing.

    Dispatches between the two legacy branches: :func:`_single_class_map` for
    textline/word/glyph/printspace, :func:`_layout_map` for the layout use case.
    Returns ``None`` when the page carries nothing the config labels.
    """
    shape = (page.height, page.width)
    if config.use_case in SINGLE_CLASS_USE_CASES:
        return _single_class_map(page, config, shape)
    return _layout_map(page, config, shape)


def _single_class_map(
    page: Page, config: GroundTruthConfig, shape: tuple[int, int]
) -> np.ndarray | None:
    """Label map for the single-class use cases (the legacy ``use_case`` branch).

    The boundary class is carved out with :func:`~.geometry.update_region_contours` using
    :data:`SINGLE_CLASS_BOUNDARY_RATES`, exactly as the legacy branch did; the boundary is
    written *after* the regions, so it wins where they overlap — also as before.
    """
    contours = _case_contours(page, config.use_case)
    if not contours:
        return None
    label = config.case_label
    if config.artificial_class_label is None:
        return label_image([(contour, label) for contour in contours], shape, background=0)

    eroded, boundary = update_region_contours(
        contours,
        shape,
        erosion_rate=SINGLE_CLASS_BOUNDARY_RATES.erosion_rate,
        dilation_rate=SINGLE_CLASS_BOUNDARY_RATES.dilation_rate,
        erosion_early=SINGLE_CLASS_BOUNDARY_RATES.erosion_early,
        dilation_early=SINGLE_CLASS_BOUNDARY_RATES.dilation_early,
    )
    label_map = label_image([(contour, label) for contour in eroded], shape, background=0)
    label_map[boundary.astype(bool)] = config.artificial_class_label
    return label_map


def _case_contours(page: Page, use_case: UseCase) -> list[np.ndarray]:
    """Contours of the regions *use_case* annotates."""
    kinds = {
        UseCase.TEXTLINE: RegionKind.TEXTLINE,
        UseCase.WORD: RegionKind.WORD,
        UseCase.GLYPH: RegionKind.GLYPH,
        UseCase.PRINTSPACE: RegionKind.PRINTSPACE,
        UseCase.BINARIZATION: RegionKind.PRINTSPACE,
    }
    if use_case in kinds:
        kind = kinds[use_case]
        if page.printspace_contour is not None and kind is RegionKind.PRINTSPACE:
            return [page.printspace_contour]
        return page.contours_for(kind)
    return [region.contour for region in page.regions if region.kind is not RegionKind.BORDER]


def _layout_map(page: Page, config: GroundTruthConfig, shape: tuple[int, int]) -> np.ndarray | None:
    """Label map for the layout use case: one class per configured region type.

    Paint order is what makes this correct, and the legacy code got it by accident through
    dict iteration order.  Here it is explicit, largest region first:

    1. the print space as a class (its own contour, when configured);
    2. the configured regions, big over small, so a drop-capital is not buried by the
       paragraph that contains it;
    3. the artificial boundary class last, over every region it was carved from.
    """
    labelled = [(region, config.label_for(region)) for region in page.regions]
    usable = [(region, label) for region, label in labelled if label is not None]
    if not usable:
        return None
    # The order the docstring promises: biggest first, so a small region inside a large one
    # keeps its own colour.  Appending in document order let a header declared before its
    # paragraph be painted first and then buried by it.
    pairs = sorted(usable, key=lambda pair: contour_area(pair[0].contour), reverse=True)

    entries: list[tuple[np.ndarray, int]] = []
    if config.printspace_as_class_in_layout is not None and page.printspace_contour is not None:
        # The legacy code painted this *after* the regions with a mask covering everything
        # outside the print-space bounding box, which recoloured the whole page background;
        # drawing the print-space contour first keeps the same colour outside it while
        # letting the regions and the boundary class stay on top.
        entries.append((page.printspace_contour, config.printspace_as_class_in_layout))

    eroded_pairs, boundary = _apply_boundaries(pairs, config, shape)
    entries.extend(eroded_pairs)

    label_map = label_image(entries, shape, background=0)
    if config.artificial_class_label is not None and boundary.any():
        label_map[boundary.astype(bool)] = config.artificial_class_label
    return label_map


def _apply_boundaries(
    pairs: Sequence[tuple[Region, int]],
    config: GroundTruthConfig,
    shape: tuple[int, int],
) -> tuple[list[tuple[np.ndarray, int]], np.ndarray]:
    """Erode every labelable region and collect the shared boundary mask.

    Replaces the eleven ``if "<type>" in elements_with_artificial_class:`` blocks. Rates
    come from :data:`BOUNDARY_RATES` when the table knows the region's type; regions whose
    type has no entry are passed through un-eroded, which is what the legacy code did when
    a type was absent from the list.  The eroded contours are returned as
    ``(contour, label)`` pairs — not a mapping keyed by :class:`~.parser.Region`, whose
    dataclass ``__eq__`` leaves it unhashable.
    """
    if config.artificial_class_label is None:
        return [(region.contour, label) for region, label in pairs], np.zeros(shape, dtype=np.uint8)

    eroded_pairs: list[tuple[np.ndarray, int]] = []
    boundary = np.zeros(shape, dtype=np.uint8)
    for region, label in pairs:
        rates = BOUNDARY_RATES.get(_boundary_key(region))
        if rates is None:
            eroded_pairs.append((region.contour, label))
            continue
        region_eroded, region_boundary = update_region_contours(
            [region.contour],
            shape,
            erosion_rate=rates.erosion_rate,
            dilation_rate=rates.dilation_rate,
            erosion_early=rates.erosion_early,
            dilation_early=rates.dilation_early,
        )
        eroded_pairs.append((region_eroded[0] if region_eroded else region.contour, label))
        boundary = np.maximum(boundary, region_boundary)
    return eroded_pairs, boundary


def _boundary_key(region: Region) -> str:
    """The :data:`BOUNDARY_RATES` key for *region*: its configured type name, else its kind."""
    if region.kind is RegionKind.TEXT:
        return normalise_text_type(region.type_name)
    return region.type_name or region.kind.value


def _write_label_image(
    page: Page,
    label_map: np.ndarray,
    config: GroundTruthConfig,
    output_dir: Path,
    output_type: OutputType,
    *,
    printspace: bool = False,
    stem: str | None = None,
) -> Path:
    """Render, crop, resize and write one page's label map.

    The legacy sequence was *crop to print space → resize to the column width → write*, with
    the crop and the resize duplicated in each of the two branches and the ``2d``/``3d``
    choice made by filling an RGB image with the class id.  Both formats now come from the
    same label map (:func:`~.render.to_label_map` / :func:`~.render.label_map_to_image`).

    The file is named *stem*, which is :func:`unique_output_stem` for the page when the caller
    is a whole run; left off it falls back to :func:`output_stem` for a page written on its own.
    """
    image = (
        to_label_map(label_map)
        if output_type is OutputType.TWO_D
        else label_map_to_image(label_map, _palette(config))
    )
    cropped = crop_to(image, page.printspace_bbox) if printspace else image
    target = _column_size(page, config)
    if target is not None and tuple(cropped.shape[:2]) != target:
        interpolation = cv2.INTER_NEAREST if output_type is OutputType.TWO_D else cv2.INTER_AREA
        cropped = cv2.resize(cropped, (target[1], target[0]), interpolation=interpolation)
    return write_image(output_dir / f"{stem or output_stem(page.path)}.png", cropped)


def _palette(config: GroundTruthConfig) -> dict[int, tuple[int, int, int]]:
    """``label -> colour`` for *config*, with the two special classes pinned.

    The legacy code looked up ``labels_rgb_color[configured_id]`` for the configured
    classes but used two hardcoded colours for the artificial-boundary and print-space
    classes.  Merging both into one mapping keeps those colours and lets the configured ids
    resolve through the shared palette instead of raising ``IndexError`` past id 18.
    """
    palette = dict(enumerate(DEFAULT_LABEL_PALETTE))
    if config.artificial_class_label is not None:
        palette[config.artificial_class_label] = ARTIFICIAL_CLASS_COLOR
    if config.printspace_as_class_in_layout is not None:
        palette[config.printspace_as_class_in_layout] = PRINTSPACE_CLASS_COLOR
    return palette


def _column_size(page: Page, config: GroundTruthConfig) -> tuple[int, int] | None:
    """``(height, width)`` imposed by ``columns_width``, or ``None``.

    The legacy code computed ``x_new``/``y_new`` from the page's column count and applied
    them with ``resize_image``; a column count missing from the table raised ``KeyError``
    and a zero page width raised ``ZeroDivisionError``.  Both now yield ``None`` — no
    resize — which is also what happened when the page had no ``num_col`` comment.
    """
    if page.columns_width is None or not config.columns_width:
        return None
    width = config.columns_width.get(page.columns_width)
    if not width or page.width <= 0:
        return None
    return (int(width * page.height / page.width), int(width))


def _copy_source_image(
    page: Page,
    config: GroundTruthConfig,
    images_dir: str | Path | None,
    images_output_dir: str | Path | None,
    *,
    printspace: bool = False,
    stem: str | None = None,
) -> Path | None:
    """Copy (and match the label's transforms on) the page's source image.

    The legacy code did this inline in both branches and required the image to exist —
    ``ls_org_imgs_stem.index(...)`` raised ``ValueError`` for any page whose photo was
    missing, aborting the run.  A missing photo is skipped quietly here.

    *stem* is the name the page's label was written under, so the two stay a pair: the dataset
    loader matches ``images/X.png`` to ``labels/X.png`` by stem, and a label named one way and
    its image another would be a page the run generated and the training never sees.
    """
    if images_dir is None or images_output_dir is None:
        return None
    source = _find_image(page, images_dir)
    if source is None:
        return None
    image = read_image(source)
    if image is None:
        return None
    if printspace:
        image = crop_to(image, page.printspace_bbox)
    target = _column_size(page, config)
    if target is not None and tuple(image.shape[:2]) != target:
        image = cv2.resize(image, (target[1], target[0]), interpolation=cv2.INTER_AREA)
    name = stem or output_stem(page.path)
    return write_image(Path(images_output_dir) / f"{name}{source.suffix}", image)


def _find_image(page: Page, images_dir: str | Path) -> Path | None:
    """The photo of *page* in *images_dir*, under either of its two stems.

    Only files whose suffix is in :data:`~sbb_pytorch.storage.images.IMAGE_SUFFIXES` are
    candidates.  Matching any file let the page's own ``.xml`` win when the XML folder and
    the image folder were the same directory -- ``000123456-1800.xml`` sorts before
    ``1800.png`` -- after which ``read_image`` returned ``None`` and the source copy was
    dropped without a word.
    """
    base = Path(images_dir)
    if not base.is_dir():
        return None
    wanted = {output_stem(page.path), page.file_name}
    entries = sorted(base.iterdir())
    for entry in entries:
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES and entry.stem in wanted:
            return entry
    return None


def output_stem(path: str | Path) -> str:
    """Output stem for a PageXML file: the legacy ``name.split('-')[1].split('.')[0]``.

    ``000123456-1800.xml`` → ``1800``; ``a-b-c.xml`` → ``b``; ``000123456-1800.tar.gz`` →
    ``1800``; a name with no hyphen keeps its whole stem (``000123456.xml`` →
    ``000123456``), which is what the legacy ``except`` branch produced.  An earlier port
    used ``Path.stem`` plus ``partition("-")``, which answered ``b-c`` and ``1800.tar`` for
    the multi-hyphen and multi-dot spellings -- the segment between the first hyphen and the
    *last* extension is not what the legacy datasets were named with, so re-running one
    produced a second file beside the original instead of overwriting it.

    The rule is not *unique*: ``882241214-0113.xml`` and ``901234567-0113.xml``, and a page's
    PageXML and its ALTO, all answer ``0113``.  A run writes through
    :func:`unique_output_stem`, which keeps this answer for the first file to claim a stem and
    disambiguates the ones behind it.
    """
    name = Path(path).name
    parts = name.split("-")
    if len(parts) < 2:
        return name.split(".")[0]  # the legacy ``except`` branch, reached via IndexError
    return parts[1].split(".")[0]


def unique_output_stem(path: str | Path, taken: set[str]) -> str:
    """The stem *path*'s outputs are written under, once *taken* holds the names already claimed.

    :func:`output_stem` is the legacy rule and on a mixed dataset it collides: two documents
    that each hold a page ``0113`` both answer ``0113``, and so do a page's PageXML and its ALTO
    (``882241214-0113.xml`` and ``882241214-0113.alto.xml`` read the same between the first
    hyphen and the first dot).  Every file that wanted a name it had already given took the file
    after it down with it, which is how a folder of 2455 XMLs became 476 label images.  The
    first file to claim a stem keeps the legacy answer -- so a dataset generated before this
    rule still matches what the next run writes -- and the ones behind it take the whole file
    stem (``882241214-0113``, and ``882241214-0113.alto`` for the ALTO of a page whose PageXML
    got there first), then ``_1``, ``_2``.  *taken* is updated in place, so a caller threading it
    through a run gives every page in that run a file of its own.
    """
    stem = output_stem(path)
    if stem not in taken:
        taken.add(stem)
        return stem
    full = Path(path).stem
    index = 1
    while full in taken:
        full = f"{Path(path).stem}_{index}"
        index += 1
    taken.add(full)
    return full


def labelled_contours(contours: Iterable[np.ndarray], label: int) -> list[tuple[np.ndarray, int]]:
    """Pair each contour with *label*, the input shape :func:`~.render.label_image` takes."""
    return [(contour, int(label)) for contour in contours]
