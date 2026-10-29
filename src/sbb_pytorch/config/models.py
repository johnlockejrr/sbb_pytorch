"""Configuration models -- three pydantic models replacing eight hand-written dataclasses.

The pre-refactor ``config_loader.py`` declared ``ModelConfig``, ``TransformerConfig``,
``AugmentationConfig``, ``AugmentationParams``, ``TrainingConfig``, ``DirectoryConfig``,
``Page2LabelConfig`` and ``ScalesConfig``, then rebuilt all of them by hand from **one flat
JSON dict** using ``self.config['key']`` everywhere. Two consequences followed from that:

* Any key a file omitted raised ``KeyError``, even when the value was irrelevant.
  ``config_train_textline.json`` omits ``adding_rgb_background``, ``shuffle_indexes``,
  ``number_of_backgrounds_per_image``, ``dir_rgb_backgrounds`` and the optimizer keys, so
  ``get_directory_config()`` and ``get_augmentation_config()`` both crashed on it -- the file
  was only loadable through ``load_config()``.
* Nothing was validated: types, ranges and cross-field rules existed only where a hand-written
  ``if`` happened to exist (three of them, in ``validate_configs``).

Here every field has a type, a default and -- where meaningful -- a constraint, and the
relationships between files are declared rather than remembered:

* Flat legacy JSON keeps working -- ``transformer_*`` keys are folded into
  :class:`TransformerConfig`, and inlined page-to-label keys into :class:`Page2Label`.
* ``" "`` for ``dir_of_start_model`` (which the shipped config uses to mean "unset") becomes
  ``None``.
* The shipped ``"blur_k": ["blur", "guass", "median"]`` misspelling is corrected instead of
  crashing, matching :class:`sbb_pytorch.storage.images.BlurKind`.
* ``model_json_schema()`` now *is* the config documentation -- no separate table to keep in sync.

Advisory checks that would reject a shipped-but-imperfect value are exposed through
:meth:`SBBConfig.warnings` rather than raised, so a config can be loaded and inspected without
the CLI refusing to start. See ``docs/migration.md`` for the full list of semantic changes.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sbb_pytorch.storage.images import BlurKind

__all__ = [
    "Page2Label",
    "SBBConfig",
    "ScalesConfig",
    "BackboneType",
    "BlurKind",
    "OptimizerType",
    "TaskType",
    "TransformerConfig",
    "UseCase",
    "TypeOutput",
    "validate_pair",
]

#: Output array shape for generated labels (the legacy ``--type_output`` flag took any string).
TypeOutput = Literal["2d", "3d"]

#: What ``train.py --device`` accepted as a free string is now a validated pattern.
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")

#: The field names :class:`TransformerConfig` declares; the flat config spelling prefixes
#: each one with ``transformer_`` (``transformer_patchsize_x`` -> ``patchsize_x``).
_TRANSFORMER_FIELDS = (
    "num_patches_xy",
    "patchsize_x",
    "patchsize_y",
    "projection_dim",
    "mlp_head_units",
    "layers",
    "num_heads",
    "cnn_first",
)

#: The flat keys that used to spell out :class:`Page2Label` as siblings of the main config.
_PAGE2LABEL_KEYS = (
    "use_case",
    "textregions",
    "imageregion",
    "separatorregion",
    "graphicregions",
    "columns_width",
)

#: Historical spellings accepted for :class:`BlurKind` values before validation.
_BLUR_ALIASES = {"guass": "gauss", "gaussian": "gauss", "mean": "blur", "median_blur": "median"}


class BackboneType(str, Enum):
    """Feature-extractor family. ``RESNET`` and ``NONTRANSFORMER`` are the same CNN backbone
    under two historical names; both were selectable from the config."""

    TRANSFORMER = "transformer"
    RESNET = "resnet"
    NONTRANSFORMER = "nontransformer"


class TaskType(str, Enum):
    """What the model predicts.

    ``BINARIZATION`` and ``READING_ORDER`` were absent from the enum although both are shipped
    Eynollah use cases, so a config naming either one failed to load.
    """

    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ENHANCEMENT = "enhancement"
    BINARIZATION = "binarization"
    READING_ORDER = "reading_order"


class UseCase(str, Enum):
    """Eynollah page layout use cases, used by the page-to-label mapping."""

    TEXTLINE = "textline"
    LAYOUT = "layout"
    PRINTSPACE = "printspace"
    GLYPH = "glyph"
    WORD = "word"


class OptimizerType(str, Enum):
    """``initialization.py`` hard-coded ``torch.optim.Adam`` while the config carried an unused
    ``"optimizer": "adam"`` string; that string is now the field that actually selects one."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADADELTA = "adadelta"


class TransformerConfig(BaseModel):
    """Vision-transformer geometry. Was seven flat ``transformer_*`` keys."""

    model_config = ConfigDict(extra="forbid")

    num_patches_xy: tuple[int, int]
    patchsize_x: int = Field(gt=0)
    patchsize_y: int = Field(gt=0)
    projection_dim: int = Field(gt=0)
    mlp_head_units: tuple[int, ...] = Field(min_length=1)
    layers: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    cnn_first: bool = False

    @model_validator(mode="after")
    def _heads_divide_dim(self) -> TransformerConfig:
        if self.projection_dim % self.num_heads:
            raise ValueError(
                f"transformer_projection_dim={self.projection_dim} is not divisible by "
                f"transformer_num_heads={self.num_heads}"
            )
        return self


class Page2Label(BaseModel):
    """``custom_config_page2label.json`` -- which XML element names map to which class ids.

    The legacy ``validate_configs`` required *exactly one* text region for the textline use
    case; the shipped file defines three (``rest_as_paragraph``, ``header``, ``heading``,
    ``marginalia``) so the rule contradicted the project's own config and is dropped. The
    duplicate value (both ``header`` and ``heading`` map to ``2``) is reported by
    :meth:`SBBConfig.warnings` rather than rejected, because it is present in the shipped file.
    """

    model_config = ConfigDict(extra="forbid")

    use_case: UseCase
    textregions: dict[str, int] = Field(default_factory=dict)
    imageregion: int | None = None
    separatorregion: int | None = None
    graphicregions: dict[str, int] = Field(default_factory=dict)
    columns_width: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _values_are_positive(self) -> Page2Label:
        bad = {name: value for name, value in self.all_regions().items() if value < 0}
        if bad:
            raise ValueError(f"negative class ids in page2label mapping: {bad}")
        return self

    def all_regions(self) -> dict[str, int]:
        """Every region name and its class id, including the two unnamed singletons."""
        regions = {**self.textregions, **self.graphicregions}
        if self.imageregion is not None:
            regions["<imageregion>"] = self.imageregion
        if self.separatorregion is not None:
            regions["<separatorregion>"] = self.separatorregion
        return regions

    def class_ids(self) -> dict[int, list[str]]:
        """Inverted view: ``{class_id: [region names]}`` -- duplicates have more than one name."""
        inverted: dict[int, list[str]] = {}
        for name, value in self.all_regions().items():
            inverted.setdefault(int(value), []).append(name)
        return inverted


class ScalesConfig(BaseModel):
    """``scales_enhancement.json``. The legacy validator required ``0 < scale < 1``; kept."""

    model_config = ConfigDict(extra="forbid")

    scales: Annotated[tuple[float, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _in_range(self) -> ScalesConfig:
        bad = [s for s in self.scales if not 0.0 < s < 1.0]
        if bad:
            raise ValueError(f"scales must be between 0 and 1, got {bad}")
        return self


class SBBConfig(BaseModel):
    """The whole configuration for one run: model, training, augmentation and directories.

    Every field carries a default, so a config may name only what it cares about -- the old
    bracket-indexing made an omitted optional key a ``KeyError``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    # ------------------------------------------------------------------ model

    task: TaskType
    backbone_type: BackboneType
    n_classes: int = Field(gt=0)
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    light_model: bool = False
    patches: bool = False
    pretraining: bool = False
    data_is_provided: bool = False
    transformer: TransformerConfig | None = None
    page2label: Page2Label | None = None
    #: Enhancement-task scales (``scales_enhancement.json``), loaded alongside the main config.
    enhancement_scales: ScalesConfig | None = None

    # -------------------------------------------------------------- training

    n_epochs: int = Field(default=50, ge=0)
    n_batch: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-6, ge=0)
    optimizer: OptimizerType = OptimizerType.ADAM
    momentum: float = Field(default=0.9, ge=0, lt=1)
    device: str = "cuda"
    num_workers: int = Field(default=4, ge=0)
    patience: int = Field(default=10, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    gradient_clip_val: float | None = Field(default=None, ge=0)
    precision: Literal["32", "16", "bf16", "32-true", "16-mixed", "bf16-mixed"] = "32-true"
    continue_training: bool = False
    index_start: int = Field(default=0, ge=0)
    dir_of_start_model: Path | None = None
    weighted_loss: bool = False
    is_loss_soft_dice: bool = False
    seed: int | None = None

    # ----------------------------------------------------------- augmentation

    augmentation: bool = False
    flip_aug: bool = False
    blur_aug: bool = False
    scaling: bool = False
    adding_rgb_background: bool = False
    adding_rgb_foreground: bool = False
    add_red_textlines: bool = False
    channels_shuffling: bool = False
    degrading: bool = False
    brightening: bool = False
    binarization: bool = False
    scaling_bluring: bool = False
    scaling_binarization: bool = False
    scaling_flip: bool = False
    rotation: bool = False
    rotation_not_90: bool = False

    blur_k: list[BlurKind] = Field(
        default_factory=lambda: [BlurKind.BLUR, BlurKind.GAUSS, BlurKind.MEDIAN]
    )
    scales: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    brightness: tuple[float, ...] = (1.3, 1.5, 1.7, 2.0)
    degrade_scales: tuple[float, ...] = (0.2, 0.4)
    flip_index: tuple[int, ...] = (0, 1, -1)
    shuffle_indexes: tuple[tuple[int, ...], ...] = ((0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0))
    #: ``thetha`` is the legacy spelling (sic); ``theta`` is accepted as an alternative.
    thetha: Annotated[
        tuple[float, ...], Field(validation_alias=AliasChoices("thetha", "theta"))
    ] = (
        5.0,
        -5.0,
    )
    number_of_backgrounds_per_image: int = Field(default=2, ge=1)

    # ------------------------------------------------------------ directories

    dir_train: str = ""
    dir_eval: str = ""
    dir_output: str = "runs/default"
    dir_rgb_backgrounds: str = ""
    dir_rgb_foregrounds: str = ""
    dir_img_bin: str = ""

    # ------------------------------------------------------------- inference

    type_output: TypeOutput = "2d"
    save_visualisation: bool = True

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_keys(cls, data: Any) -> Any:
        """Fold the flat ``transformer_*`` / page-to-label keys into their nested models.

        ``config_params.json`` spells the transformer block as eight sibling keys prefixed
        ``transformer_``; the modern shape nests them under ``transformer``. Both are accepted,
        and an explicit nested block always wins over the prefixed keys.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        nested = data.get("transformer")
        if nested is None and data.get("backbone_type") == BackboneType.TRANSFORMER:
            folded = {
                name: data.pop(f"transformer_{name}")
                for name in _TRANSFORMER_FIELDS
                if f"transformer_{name}" in data
            }
            if len(folded) == len(_TRANSFORMER_FIELDS):
                data["transformer"] = folded
        elif nested is None:
            # Drop a non-transformer config's leftover prefixed keys instead of tripping extra.
            for name in _TRANSFORMER_FIELDS:
                data.pop(f"transformer_{name}", None)

        if data.get("page2label") is None:
            folded_labels = {key: data.pop(key) for key in _PAGE2LABEL_KEYS if key in data}
            if "use_case" in folded_labels:
                data["page2label"] = folded_labels

        if isinstance(data.get("scales"), list):
            data["scales"] = tuple(float(s) for s in data["scales"])
        return data

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> str:
        value = str(value)
        if not _DEVICE_RE.match(value):
            raise ValueError(f"device must be one of auto, cpu, mps, cuda, cuda:N -- got {value!r}")
        return value

    @field_validator("dir_of_start_model", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: Any) -> Any:
        """The shipped configs use ``" "`` for "no weights"; that means ``None``, not a path."""
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return Path(stripped) if stripped else None

    @field_validator(
        "dir_train",
        "dir_eval",
        "dir_output",
        "dir_rgb_backgrounds",
        "dir_rgb_foregrounds",
        "dir_img_bin",
        mode="before",
    )
    @classmethod
    def _strip_paths(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("blur_k", mode="before")
    @classmethod
    def _accept_legacy_blur_spelling(cls, value: Any) -> Any:
        """``"guass"`` ships in both configs; the enum spells it ``gauss``. Correct, don't fail."""
        if value is None:
            return None
        items = value if isinstance(value, list | tuple) else [value]
        return [
            _BLUR_ALIASES.get(i.strip().lower(), i.strip().lower()) if isinstance(i, str) else i
            for i in items
        ]

    @field_validator("flip_index", mode="before")
    @classmethod
    def _validate_flip_codes(cls, value: Any) -> Any:
        codes = tuple(value) if isinstance(value, list | tuple) else (value,)
        bad = [c for c in codes if int(c) not in (-1, 0, 1)]
        if bad:
            raise ValueError(f"flip_index entries must be -1, 0 or 1, got {bad}")
        return tuple(int(c) for c in codes)

    @field_validator("shuffle_indexes", mode="before")
    @classmethod
    def _validate_channel_permutations(cls, value: Any) -> Any:
        if value is None:
            return None
        rows = tuple(tuple(int(i) for i in row) for row in value)
        bad = [row for row in rows if sorted(row) != [0, 1, 2]]
        if bad:
            raise ValueError(f"shuffle_indexes entries must permute (0, 1, 2), got {bad}")
        return rows

    @model_validator(mode="after")
    def _cross_field_rules(self) -> SBBConfig:
        if self.task is TaskType.ENHANCEMENT and self.n_classes != 3:
            # Legacy rule: enhancement predicts colour, so it always needs three channels.
            raise ValueError(f"task=enhancement requires n_classes=3, got {self.n_classes}")
        if self.task is TaskType.CLASSIFICATION and self.n_classes < 2:
            raise ValueError(f"task=classification requires n_classes>=2, got {self.n_classes}")
        if self.augmentation and not (self.dir_train or self.data_is_provided):
            raise ValueError(
                "augmentation=True but no training data: set dir_train or data_is_provided"
            )
        if self.backbone_type is BackboneType.TRANSFORMER and self.transformer is None:
            raise ValueError(
                "backbone_type=transformer needs a transformer config "
                "(nested 'transformer' or the eight transformer_* keys)"
            )
        if self.transformer is not None and self.backbone_type is not BackboneType.TRANSFORMER:
            raise ValueError(
                f"transformer config given but backbone_type={self.backbone_type.value}"
            )
        return self

    # ------------------------------------------------------------------ API

    @property
    def image_size(self) -> tuple[int, int]:
        """``(height, width)`` -- the shape every dataset and model consumes."""
        return self.input_height, self.input_width

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SBBConfig:
        """Build from an already-parsed dict (the ``**kwargs`` style ``train.py`` used)."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> SBBConfig:
        """Read and validate a main config file (``config_params.json`` and friends)."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def json_schema(self) -> dict[str, Any]:
        """The generated JSON schema -- what ``docs/config-reference.md`` is rendered from."""
        return type(self).model_json_schema()

    def warnings(self) -> list[str]:
        """Advisory problems worth printing before a run: real, but not worth refusing to start.

        Deliberately *not* ``ValueError``s -- every warning here is exhibited by a config the
        project ships, so raising would break existing runs rather than improve them.
        """
        notes: list[str] = []

        if self.input_height % 32 or self.input_width % 32:
            notes.append(
                f"input size {self.input_height}x{self.input_width} is not a multiple of 32; "
                "a 5-stage U-Net downsamples to rounded sizes and the decoded output will not "
                "align with the input mask"
            )
        if self.transformer is not None:
            cols, rows = self.transformer.num_patches_xy
            expected = (cols * self.transformer.patchsize_x, rows * self.transformer.patchsize_y)
            if expected != (self.input_width, self.input_height):
                notes.append(
                    f"transformer grid {cols}x{rows} with patch size "
                    f"{self.transformer.patchsize_x}x{self.transformer.patchsize_y} covers "
                    f"{expected[0]}x{expected[1]}, but input is "
                    f"{self.input_width}x{self.input_height}"
                )
        if self.pretraining and self.dir_of_start_model is None:
            notes.append(
                "pretraining=True with no dir_of_start_model; training starts from random "
                "initialisation"
            )
        if self.page2label is not None:
            for value, names in sorted(self.page2label.class_ids().items()):
                if len(names) > 1:
                    notes.append(
                        f"page2label class id {value} is shared by several regions: "
                        f"{', '.join(sorted(names))}"
                    )
        if self.task is TaskType.SEGMENTATION and self.patches and self.transformer is None:
            notes.append(
                "legacy validation required a transformer config for segmentation with "
                "patches; none was given"
            )
        if (
            self.page2label is None
            and self.task in (TaskType.SEGMENTATION, TaskType.BINARIZATION)
            and not self.data_is_provided
        ):
            notes.append(
                f"task={self.task.value} without a page2label mapping: labels cannot be "
                "derived from ALTO/PageXML, so only pre-made label images can be used"
            )
        return notes


def validate_pair(config: SBBConfig, page2label: Page2Label | None = None) -> list[str]:
    """Check a main config against a page-to-label config and return the problems as strings.

    Returns:
        Human-readable problems; empty means the pair is consistent. The CLI *prints* them
        rather than raising: the shipped ``config_params.json`` and
        ``custom_config_page2label.json`` are themselves an inconsistent pair (the main config
        declares ``n_classes=2`` while the mapping uses ids up to 6), and refusing to start
        would break a run that exists today. A mapping that exceeds ``n_classes`` is how the
        old code produced a model that trained fine and mislabelled every page.
    """
    label = page2label or config.page2label
    if label is None:
        return []

    highest = max(label.class_ids(), default=0)
    if highest >= config.n_classes:
        return [
            f"page2label maps a region to class id {highest}, but n_classes={config.n_classes} "
            f"only covers ids 0..{config.n_classes - 1}"
        ]
    return []
