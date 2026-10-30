"""Ground-truth generation for the SBB pixel-wise segmentation pipeline.

Public API of the :mod:`sbb_pytorch.gt` subpackage.  Every name below is re-exported
explicitly — no star imports — so ``from sbb_pytorch.gt import ...`` has a fixed surface
that linters and type checkers can resolve.

Dependency direction inside the subpackage (no cycles):

::

    geometry  <-  parser  <-  render  <-  generator
        ^          ┌──┴──┐                    ^
        │          │     └──  writer          │
        └──────────┼──────────────────────────┘
                   └──  reading_order

Modules here depend only on the standard library, ``numpy`` and ``cv2``; :mod:`.writer`
needs nothing beyond the standard library's ``xml.etree``.  The legacy
``gt_gen_utils.py`` additionally imported ``shapely`` and ``matplotlib``; neither is
needed (see :mod:`.geometry` and :mod:`.render`), and ``click`` lives in the package's
single CLI entry point, never in a library module.
"""

from .generator import (
    BOUNDARY_RATES,
    SINGLE_CLASS_BOUNDARY_RATES,
    BoundaryRates,
    GenerationResult,
    GroundTruthConfig,
    UseCase,
    build_label_map,
    generate_ground_truth,
    labelled_contours,
    output_stem,
)
from .geometry import (
    DEFAULT_KERNEL,
    DEFAULT_MIN_AREA_FRACTION,
    DEFAULT_MIN_AREA_PIXELS,
    BoundingBox,
    ContourFeatures,
    FilteredContours,
    bounding_box,
    contour_area,
    contour_bbox,
    contour_centroid,
    contour_features,
    contours_of_image,
    contours_of_region,
    filter_contours_by_area,
    largest_contour,
    largest_contour_bbox,
    mask_of,
    return_parent_contours,
    rotated_rect_with_max_area,
    scale_contour,
    update_region_contours,
)
from .parser import (
    DEFAULT_XML_ENCODING,
    GRAPHIC_REGION_TYPES,
    LEGACY_INACTIVE_TEXT_TYPES,
    TEXT_REGION_TYPES,
    Contour,
    Page,
    PageXMLError,
    Region,
    RegionKind,
    discover_xml_files,
    image_for,
    normalise_text_type,
    parse_pagexml,
)
from .reading_order import (
    BACKGROUND_LABEL,
    DEFAULT_MAX_AREA_FRACTION,
    HEADER_LABEL,
    HEADER_SEPARATOR_BAND_WIDTH,
    IMAGE_REGION_LABEL,
    QUERY_LABEL,
    ReadingOrderPair,
    generate_pairs,
    header_separator_mask,
    ordered_regions,
    ordered_text_regions,
    region_label_map,
    visualize_reading_order,
    write_pair,
)
from .render import (
    ARTIFICIAL_CLASS_COLOR,
    BINARY_BACKGROUND_COLOR,
    BINARY_FOREGROUND_COLOR,
    DEFAULT_LABEL_PALETTE,
    DEFAULT_OVERLAY_ALPHA,
    PRINTSPACE_CLASS_COLOR,
    OutputType,
    crop_to,
    image_for_page,
    label_color,
    label_image,
    label_map_to_image,
    make_image_from_bb,
    overlay,
    paint_contours,
    read_image,
    render_page,
    to_label_map,
    to_uint8_bgr,
    write_image,
)
from .writer import (
    CREATOR,
    PAGE_NAMESPACE,
    SCHEMA_LOCATION,
    points_attribute,
    to_pagexml,
    write_pagexml,
)

__all__ = [
    # parser — ALTO/PageXML schema
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
    "image_for",
    "normalise_text_type",
    "parse_pagexml",
    # geometry — contour arithmetic
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
    # render — rasterising regions
    "ARTIFICIAL_CLASS_COLOR",
    "BINARY_BACKGROUND_COLOR",
    "BINARY_FOREGROUND_COLOR",
    "DEFAULT_LABEL_PALETTE",
    "DEFAULT_OVERLAY_ALPHA",
    "PRINTSPACE_CLASS_COLOR",
    "OutputType",
    "crop_to",
    "image_for_page",
    "label_color",
    "label_image",
    "label_map_to_image",
    "make_image_from_bb",
    "overlay",
    "paint_contours",
    "read_image",
    "render_page",
    "to_label_map",
    "to_uint8_bgr",
    "write_image",
    # reading_order — pairwise training samples
    "HEADER_SEPARATOR_BAND_WIDTH",
    "BACKGROUND_LABEL",
    "DEFAULT_MAX_AREA_FRACTION",
    "HEADER_LABEL",
    "IMAGE_REGION_LABEL",
    "QUERY_LABEL",
    "ReadingOrderPair",
    "generate_pairs",
    "header_separator_mask",
    "ordered_regions",
    "ordered_text_regions",
    "region_label_map",
    "visualize_reading_order",
    "write_pair",
    # generator — the pipeline entry point
    "BOUNDARY_RATES",
    "SINGLE_CLASS_BOUNDARY_RATES",
    "BoundaryRates",
    "GenerationResult",
    "GroundTruthConfig",
    "UseCase",
    "build_label_map",
    "generate_ground_truth",
    "labelled_contours",
    "output_stem",
    # writer — PageXML serialisation
    "CREATOR",
    "PAGE_NAMESPACE",
    "SCHEMA_LOCATION",
    "points_attribute",
    "to_pagexml",
    "write_pagexml",
]
