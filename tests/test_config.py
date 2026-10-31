"""Tests for the config schema and loaders, including the two configs the repo ships."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sbb_pytorch.config.loaders import (
    apply_overrides,
    dump_json_schema,
    load_config,
    load_page2label,
    load_scales,
)
from sbb_pytorch.config.models import (
    CASE_LABEL_USE_CASES,
    SINGLE_CLASS_USE_CASES,
    Page2Label,
    SBBConfig,
    ScalesConfig,
    UseCase,
    validate_pair,
)
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.gt import generator as gt_generator

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE: dict[str, object] = {
    "task": "segmentation",
    "backbone_type": "nontransformer",
    "n_classes": 3,
    "input_height": 672,
    "input_width": 448,
}

#: Mirrors the transformer half of the shipped ``config_params.json``.
TRANSFORMER: dict[str, object] = {
    "task": "segmentation",
    "backbone_type": "transformer",
    "n_classes": 2,
    "input_height": 448,
    "input_width": 448,
    "pretraining": True,
    "transformer_num_patches_xy": [56, 56],
    "transformer_patchsize_x": 4,
    "transformer_patchsize_y": 4,
    "transformer_projection_dim": 64,
    "transformer_mlp_head_units": [128, 64],
    "transformer_layers": 1,
    "transformer_num_heads": 1,
    "transformer_cnn_first": False,
}


def make(**overrides: object) -> dict[str, object]:
    """``BASE`` plus whatever a given test cares about."""
    return {**BASE, **overrides}


def write(tmp_path: Path, data: dict[str, object], name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------- schema


def test_minimal_config_fills_in_every_default() -> None:
    """Five fields used to be enough only for ``load_config``; now they are enough, period."""
    config = SBBConfig.model_validate(BASE)
    assert config.n_epochs == 50
    assert config.n_batch == 1
    assert config.learning_rate == 1e-4
    assert config.optimizer.value == "adam"
    assert config.device == "cuda"
    assert config.num_workers == 4
    assert config.patience == 10
    assert config.image_size == (672, 448)
    assert config.transformer is None
    assert config.type_output == "2d"


@pytest.mark.parametrize("data", [BASE, TRANSFORMER], ids=["textline", "params"])
def test_shipped_root_configs_still_validate(data: dict[str, object]) -> None:
    """Both shipped configs must keep loading -- the refactor may not break either of them."""
    assert SBBConfig.model_validate(data).n_classes in (2, 3)


def test_every_config_in_the_configs_directory_validates() -> None:
    """Whatever ships under ``configs/`` must parse, or the CLI ships a broken default."""
    if not (REPO_ROOT / "configs").is_dir():
        pytest.skip("configs/ not created yet")
    paths = sorted((REPO_ROOT / "configs").glob("*.json"))
    assert paths, "configs/ exists but holds no JSON files"
    for path in paths:
        SBBConfig.model_validate_json(path.read_text(encoding="utf-8"))


def test_the_shipped_textline_config_is_single_class() -> None:
    """``configs/textline.json`` predicts background plus one class, and now says so.

    It said ``n_classes: 7`` -- ``configs/layout.json``'s count, copied onto a config whose own
    upstream value is 2 -- so the head carried five channels nothing supervised.  The head width
    is a function of this number, so the value is pinned here as loudly as the checkpoint that
    stops loading under it.
    """
    config = load_config(REPO_ROOT / "configs" / "textline.json")
    assert config.n_classes == 2
    assert config.page2label is not None
    assert config.page2label.use_case.value == "textline"
    assert config.page2label.case_label == 1  # the field the model lacked and the file implies
    assert config.page2label.legal_class_ids() == {0, 1}  # not the six ids the file's table lists


def test_legal_class_ids_follows_the_use_case_not_only_the_regions() -> None:
    """``class_ids()`` answers "which ids do the regions use"; the data loader needs "which ids may
    a label image carry", and for a single-class use case those are different questions.

    ``SINGLE_CLASS_USE_CASES`` / ``CASE_LABEL_USE_CASES`` move here from ``gt/generator.py``, which
    had its own copy of the pair -- the reason ``Page2Label`` could not ask the same question the
    generator asks.
    """
    single = frozenset({UseCase.TEXTLINE, UseCase.WORD, UseCase.GLYPH, UseCase.PRINTSPACE})
    assert single == SINGLE_CLASS_USE_CASES
    assert single | {UseCase.BINARIZATION} == CASE_LABEL_USE_CASES
    # The generator re-exports both instead of defining them, so the two cannot drift again.
    assert gt_generator.SINGLE_CLASS_USE_CASES is SINGLE_CLASS_USE_CASES
    assert gt_generator.CASE_LABEL_USE_CASES is CASE_LABEL_USE_CASES

    layout = Page2Label(use_case="layout", textregions={"<a>": 1, "<b>": 2}, imageregion=4)
    assert layout.legal_class_ids() == {0, 1, 2, 4}  # unchanged: the region table, plus background

    for use_case in CASE_LABEL_USE_CASES:
        assert Page2Label(use_case=use_case, case_label=3).legal_class_ids() == {0, 3}
        assert Page2Label(
            use_case=use_case, case_label=3, textregions={"<a>": 1, "<b>": 7}
        ).legal_class_ids() == {0, 3}


def test_page2label_carries_the_five_fields_the_legacy_file_has(tmp_path: Path) -> None:
    """``case_label`` and friends were readable only by ``gt.GroundTruthConfig``."""
    label = Page2Label(
        use_case="textline",
        case_label=1,
        artificial_class_label=8,
        printspace_as_class_in_layout=9,
        tableregion=10,
        noiseregion=11,
    )
    assert (label.case_label, label.tableregion, label.noiseregion) == (1, 10, 11)
    assert label.artificial_class_label == 8
    assert label.printspace_as_class_in_layout == 9
    # The inversion names only the elements it always named, so a layout config's class ids are
    # unchanged by fields that describe the run rather than its regions.
    assert label.all_regions() == {}
    assert label.class_ids() == {}
    # The set the data loader consults is not that inversion: for this use case it is the
    # background plus case_label, which class_ids() leaves out while the field names it.
    assert label.legal_class_ids() == {0, 1}

    # The flat legacy shape carried these keys beside ``use_case``; they fold into ``page2label``
    # instead of being left at the top level, where ``extra="forbid"`` would reject the file.
    config = load_config(
        write(
            tmp_path,
            make(
                use_case="textline",
                case_label=1,
                artificial_class_label=8,
                printspace_as_class_in_layout=9,
                tableregion=10,
                noiseregion=11,
            ),
        )
    )
    assert config.page2label is not None
    assert config.page2label.case_label == 1
    assert config.page2label.tableregion == 10
    assert config.page2label.noiseregion == 11


def test_flat_transformer_keys_fold_into_the_nested_model() -> None:
    config = SBBConfig.model_validate(TRANSFORMER)
    assert config.transformer is not None
    assert config.transformer.num_patches_xy == (56, 56)
    assert config.transformer.projection_dim == 64
    assert config.transformer.mlp_head_units == (128, 64)
    assert config.transformer.cnn_first is False
    # The flat keys must be gone, or extra="forbid" would reject them.
    assert "transformer_num_patches_xy" not in config.model_dump()


def test_nested_transformer_block_is_accepted_too() -> None:
    config = SBBConfig.model_validate(
        {
            **BASE,
            "backbone_type": "transformer",
            "transformer": {
                "num_patches_xy": [112, 112],
                "patchsize_x": 4,
                "patchsize_y": 4,
                "projection_dim": 64,
                "mlp_head_units": [128],
                "layers": 2,
                "num_heads": 4,
            },
        }
    )
    assert config.transformer is not None and config.transformer.num_heads == 4


def test_transformer_is_required_when_named_and_rejected_otherwise() -> None:
    with pytest.raises(Exception, match="needs a transformer config"):
        SBBConfig.model_validate({**BASE, "backbone_type": "transformer"})

    with pytest.raises(Exception, match=r"transformer config given but backbone_type=resnet"):
        SBBConfig.model_validate(
            {
                "task": "segmentation",
                "backbone_type": "resnet",
                "n_classes": 2,
                "input_height": 32,
                "input_width": 32,
                "transformer": {
                    "num_patches_xy": [8, 8],
                    "patchsize_x": 4,
                    "patchsize_y": 4,
                    "projection_dim": 8,
                    "mlp_head_units": [8],
                    "layers": 1,
                    "num_heads": 2,
                },
            }
        )


def test_head_count_must_divide_projection_dim() -> None:
    with pytest.raises(Exception, match="not divisible by"):
        SBBConfig.model_validate(
            {
                **BASE,
                "backbone_type": "transformer",
                "transformer": {
                    "num_patches_xy": [1, 1],
                    "patchsize_x": 4,
                    "patchsize_y": 4,
                    "projection_dim": 64,
                    "mlp_head_units": [8],
                    "layers": 1,
                    "num_heads": 5,
                },
            }
        )


def test_blank_start_model_string_becomes_none() -> None:
    """``config_params.json`` uses ``" "`` for "no weights"; that is not a path."""
    assert SBBConfig.model_validate(make(dir_of_start_model=" ")).dir_of_start_model is None
    assert SBBConfig.model_validate(make(dir_of_start_model="")).dir_of_start_model is None
    assert SBBConfig.model_validate(
        make(dir_of_start_model=" runs/x.pt ")
    ) == SBBConfig.model_validate(make(dir_of_start_model="runs/x.pt"))


def test_legacy_blur_misspelling_is_corrected_not_rejected() -> None:
    """Both shipped configs spell one entry ``"guass"``; the enum spells it ``gauss``."""
    config = SBBConfig.model_validate(make(blur_k=["blur", "guass", "median"]))
    assert [k.value for k in config.blur_k] == ["blur", "gauss", "median"]
    with pytest.raises(Exception, match="gauss"):
        SBBConfig.model_validate(make(blur_k=["sharpen"]))


def test_unknown_key_is_rejected() -> None:
    """``extra="forbid"``: a typo'd key is an error now, and used to be silently ignored."""
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        SBBConfig.model_validate(make(**{"learnign_rate": 1e-3}))


def test_bad_field_type_names_the_field() -> None:
    with pytest.raises(Exception, match="n_classes"):
        SBBConfig.from_dict(make(n_classes="three"))


@pytest.mark.parametrize(
    ("task", "n_classes", "message"),
    [
        ("enhancement", 2, r"task=enhancement requires n_classes=3, got 2"),
        ("classification", 1, r"task=classification requires n_classes>=2, got 1"),
    ],
)
def test_task_specific_class_rules(task: str, n_classes: int, message: str) -> None:
    with pytest.raises(Exception, match=message):
        SBBConfig.model_validate(make(task=task, n_classes=n_classes))


@pytest.mark.parametrize("task", ["binarization", "reading_order"])
def test_tasks_the_old_enum_lacked_now_load(task: str) -> None:
    """``TaskType`` omitted both, so a config naming either failed to load at all."""
    assert SBBConfig.model_validate(make(task=task)).task.value == task


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu", "mps", "auto"])
def test_accepted_devices(device: str) -> None:
    assert SBBConfig.model_validate(make(device=device)).device == device


@pytest.mark.parametrize("device", ["gpu", "CUDA", "cuda:", "tpu"])
def test_rejected_devices(device: str) -> None:
    with pytest.raises(Exception, match="device must be"):
        SBBConfig.model_validate(make(device=device))


def test_augmentation_without_data_is_rejected() -> None:
    with pytest.raises(Exception, match="augmentation=True but no training data"):
        SBBConfig.model_validate(make(augmentation=True))
    assert SBBConfig.model_validate(make(augmentation=True, data_is_provided=True)).augmentation


def test_flip_and_shuffle_indexes_are_validated() -> None:
    with pytest.raises(Exception, match="flip_index entries must be"):
        SBBConfig.model_validate(make(flip_index=[0, 2]))
    with pytest.raises(Exception, match="shuffle_indexes entries must permute"):
        SBBConfig.model_validate(make(shuffle_indexes=[[0, 1, 3]]))


def test_theta_accepts_the_legacy_misspelling_and_the_fix() -> None:
    assert SBBConfig.model_validate(make(thetha=[3, -3])).thetha == (3.0, -3.0)
    assert SBBConfig.model_validate(make(theta=[3, -3])).thetha == (3.0, -3.0)


# -------------------------------------------------------------------- loaders


def test_load_config_from_json_file(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, TRANSFORMER)).n_classes == 2


def test_load_config_reports_a_missing_file_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"main config not found: .*absent\.json"):
        load_config(tmp_path / "absent.json")


def test_load_config_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json:", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path)


def test_load_config_rejects_a_json_array(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        load_config(path)


def test_schema_error_lists_every_bad_field(tmp_path: Path) -> None:
    path = write(tmp_path, {"task": "segmentation", "backbone_type": "nontransformer"})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    # n_classes, input_height and input_width are all required.
    assert "3 validation error" in str(excinfo.value)
    assert "n_classes" in str(excinfo.value) and "input_width" in str(excinfo.value)


def test_page2label_file_loads_and_attaches(tmp_path: Path) -> None:
    label_path = write(
        tmp_path,
        {
            "use_case": "textline",
            "textregions": {"rest_as_paragraph": 1, "header": 2, "heading": 2, "marginalia": 3},
            "imageregion": 4,
            "separatorregion": 5,
            "graphicregions": {"rest_as_decoration": 6},
            "columns_width": {"1": 1000, "2": 1300},
        },
        name="page2label.json",
    )
    label = load_page2label(label_path)
    assert label.use_case.value == "textline"
    assert label.class_ids()[2] == ["header", "heading"]  # the shipped duplicate

    config = load_config(write(tmp_path, BASE), page2label_path=label_path)
    assert config.page2label == label

    # An inlined mapping wins over the separate file.
    inline = load_config(
        write(tmp_path, make(page2label={"use_case": "layout"})), page2label_path=label_path
    )
    assert inline.page2label is not None and inline.page2label.use_case.value == "layout"


def test_page2label_inlined_from_the_flat_shape(tmp_path: Path) -> None:
    """The legacy layout had these keys as siblings of ``task``/``n_classes``."""
    config = load_config(
        write(
            tmp_path,
            make(
                use_case="glyph",
                textregions={"glyph": 1},
                imageregion=0,
                separatorregion=0,
                graphicregions={},
                columns_width={},
            ),
        )
    )
    assert config.page2label is not None and config.page2label.use_case.value == "glyph"


def test_scales_file_loads(tmp_path: Path) -> None:
    path = write(tmp_path, {"scales": [0.5, 0.55, 0.9]}, name="scales.json")
    assert load_scales(path).scales == (0.5, 0.55, 0.9)

    config = load_config(write(tmp_path, make(task="enhancement", n_classes=3)), scales_path=path)
    assert config.enhancement_scales is not None

    with pytest.raises(Exception, match="scales must be between 0 and 1"):
        ScalesConfig.model_validate({"scales": [0.5, 1.4]})
    with pytest.raises(Exception, match="scales must be between 0 and 1"):
        ScalesConfig.model_validate({"scales": [0.0]})
    with pytest.raises(Exception, match="at least 1"):
        ScalesConfig.model_validate({"scales": []})


def test_validate_pair_flags_classes_the_config_cannot_represent() -> None:
    label = Page2Label(use_case="layout", textregions={"para": 1, "head": 7})
    assert validate_pair(SBBConfig.model_validate(make(n_classes=3)), label) == [
        "page2label maps a region to class id 7, but n_classes=3 only covers ids 0..2"
    ]
    assert validate_pair(SBBConfig.model_validate(make(n_classes=8)), label) == []
    # No mapping at all is not a pairing problem; warnings() reports that instead.
    assert validate_pair(SBBConfig.model_validate(make())) == []


def test_negative_class_ids_are_rejected() -> None:
    with pytest.raises(Exception, match="negative class ids"):
        Page2Label(use_case="layout", textregions={"para": -1})


def test_warnings_report_the_shipped_transformer_mismatch() -> None:
    """config_params.json asks for a 56x56 grid of 4px patches (=224) but a 448px input."""
    notes = SBBConfig.model_validate(TRANSFORMER).warnings()
    assert any("covers 224x224, but input is 448x448" in note for note in notes)
    assert any("pretraining=True with no dir_of_start_model" in note for note in notes)


def test_a_consistent_config_produces_no_warnings() -> None:
    assert SBBConfig.model_validate(make(data_is_provided=True)).warnings() == []


def test_missing_page2label_for_segmentation_is_a_warning_not_an_error() -> None:
    notes = SBBConfig.model_validate(make(augmentation=True, dir_train="data")).warnings()
    assert any("without a page2label mapping" in note for note in notes)


# ------------------------------------------------------------------ overrides


def test_cli_flag_names_map_onto_fields() -> None:
    """``--epochs`` / ``--batch-size`` / ``--resume`` had no config home; they do now."""
    config = SBBConfig.model_validate(BASE)
    changed = apply_overrides(
        config, epochs=30, batch_size=4, lr=1e-3, resume="runs/m.pt", device="cpu"
    )
    assert changed.n_epochs == 30
    assert changed.n_batch == 4
    assert changed.learning_rate == 1e-3
    assert changed.dir_of_start_model == Path("runs/m.pt")
    assert changed.device == "cpu"
    assert config.n_epochs == 50  # the original is untouched

    # Canonical field names work as well as the flag aliases.
    assert apply_overrides(config, n_epochs=7).n_epochs == 7


def test_none_overrides_are_ignored() -> None:
    config = SBBConfig.model_validate(BASE)
    assert apply_overrides(config, epochs=None, batch_size=None) is config


def test_override_validation_still_applies() -> None:
    config = SBBConfig.model_validate(BASE)
    with pytest.raises(ConfigError, match="unknown config option 'n_epocs'"):
        apply_overrides(config, n_epocs=3)
    with pytest.raises(ConfigError, match="invalid override"):
        apply_overrides(config, n_batch=0)


def test_json_schema_is_generated_and_dumpable(tmp_path: Path) -> None:
    schema = SBBConfig.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "n_classes" in schema["properties"]
    assert "task" in schema["required"]

    written = dump_json_schema(tmp_path / "docs" / "config-reference.json")
    assert json.loads(written.read_text(encoding="utf-8")) == schema
