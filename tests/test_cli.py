"""The ``sbb`` command line, driven the way the console script drives it.

Every subcommand is invoked through click's runner, because the CLI is the part of the package a
person touches daily and the part the legacy scripts got wrong most often: ``train.py``'s
``--batch-size`` reached ``ConfigLoader`` and stopped, ``inference_pytorch.py`` needed five flags
to say what one checkpoint already knows, and an unreadable config produced a traceback instead of
a message. The two error paths are tested here next to the happy paths for that reason.

The model is built once per session (:func:`sbb_pytorch.models.create_model` for the light
decoder) and reused: the CLI always constructs the architecture the config names, so a test that
wanted a cheaper model would have to change the config it is testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner, Result

from sbb_pytorch import __version__
from sbb_pytorch.cli import main
from sbb_pytorch.config import SBBConfig
from sbb_pytorch.models import create_model
from sbb_pytorch.storage import load_checkpoint, save_checkpoint
from sbb_pytorch.storage.images import imwrite

SIZE = 32
BASE: dict[str, object] = {
    "task": "segmentation",
    "backbone_type": "nontransformer",
    "light_model": True,  # the repaired light decoder, and the cheapest architecture to build
    "n_classes": 3,
    "input_height": SIZE,
    "input_width": SIZE,
    "n_epochs": 1,
    "n_batch": 2,
    "num_workers": 0,
    "device": "cpu",
    "augmentation": False,
    "pretraining": False,
}

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/schema/v2019.07.24">
  <Page imageFilename="page.png" imageWidth="{size}" imageHeight="{size}">
    <TextRegion ID="r1" type="paragraph">
      <Coords points="4,4 {edge},4 {edge},{edge} 4,{edge}"/>
    </TextRegion>
  </Page>
</PcGts>"""


@pytest.fixture(scope="session")
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, *args: str | Path) -> Result:
    """Run ``sbb`` with ``args``; ``catch_exceptions=False`` makes an unhandled error a failure."""
    return runner.invoke(main, [str(a) for a in args], catch_exceptions=False)


@pytest.fixture(scope="session")
def light_model():  # type: ignore[no-untyped-def]
    """The architecture ``BASE`` describes, built once for the checkpoint fixtures."""
    return create_model(SBBConfig.model_validate(BASE), pretrained=False)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(BASE))
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A provided split root: ``images/`` + ``labels/``, four pages, three classes."""
    images, labels = tmp_path / "pages" / "images", tmp_path / "pages" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    codes = np.tile(np.arange(SIZE, dtype=np.int64) % 3, (SIZE, 1)).astype(np.uint8)
    for index in range(4):
        imwrite(images / f"page_{index}.png", np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
        imwrite(labels / f"page_{index}.png", codes)
    return tmp_path / "pages"


@pytest.fixture
def run_dir(tmp_path: Path, light_model) -> Path:  # type: ignore[no-untyped-def]
    """A finished training run: ``model.pt`` plus its ``model_config.json`` sidecar."""
    directory = tmp_path / "run"
    directory.mkdir()
    save_checkpoint(
        directory / "model.pt", light_model, task="segmentation", epoch=2, step=7, config=BASE
    )
    return directory


# --------------------------------------------------------------------- the group itself


def test_version_reports_the_package_version(runner: CliRunner) -> None:
    assert invoke(runner, "--version").output.strip() == f"sbb, version {__version__}"


def test_help_lists_every_subcommand(runner: CliRunner) -> None:
    """Seven commands, one per legacy entry script plus the two generated-document ones."""
    output = invoke(runner, "--help").output
    for command in ("train", "infer", "gt", "export", "validate", "config-schema"):
        assert command in output


def test_unknown_command_is_a_usage_error(runner: CliRunner) -> None:
    result = invoke(runner, "trains", "config.json")
    assert result.exit_code == 2
    assert "No such command" in result.output


# ----------------------------------------------------------------------------- sbb train


def test_train_dry_run_validates_the_run_without_writing(
    runner: CliRunner, tmp_path: Path, data_root: Path
) -> None:
    """``--dry-run`` replaces the `python train.py && wait an hour to find a typo` workflow."""
    config = Path(tmp_path / "cfg.json")
    config.write_text(
        json.dumps(
            {
                **BASE,
                "dir_train": str(data_root),
                "dir_eval": str(data_root),
                "dir_output": str(tmp_path / "out"),
                "data_is_provided": True,
            }
        )
    )
    result = invoke(runner, "train", config, "--dry-run", "--no-logger")
    assert result.exit_code == 0, result.output
    assert "dry run finished" in result.output
    assert not (tmp_path / "out" / "model.pt").exists()


def test_train_writes_model_pt_and_prints_the_metrics(
    runner: CliRunner, tmp_path: Path, data_root: Path
) -> None:
    config = Path(tmp_path / "cfg.json")
    config.write_text(
        json.dumps(
            {
                **BASE,
                "dir_train": str(data_root),
                "dir_eval": str(data_root),
                "dir_output": str(tmp_path / "out"),
                "data_is_provided": True,
            }
        )
    )
    result = invoke(runner, "train", config, "--no-logger", "--no-progress-bar")
    assert result.exit_code == 0, result.output
    assert f"wrote {tmp_path / 'out' / 'model.pt'}" in result.output
    assert (tmp_path / "out" / "model.pt").exists()
    # the metrics the legacy script lost as KeyError('val_acc') are now printed from the run
    assert "val_loss" in result.output and "train_loss" in result.output


def test_device_flag_is_validated_before_any_work(runner: CliRunner, config_file: Path) -> None:
    """``--device cuda:7`` on a 1-GPU host used to fail inside ``Tensor.to``; click stops it."""
    result = invoke(runner, "train", config_file, "--device", "cuda:7", "--dry-run")
    if not torch.cuda.is_available():
        assert result.exit_code == 2
        assert "cuda" in result.output


# ----------------------------------------------------------------------------- sbb infer


def test_infer_writes_the_prediction_and_the_layout(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    """``sbb infer run_dir -i page.png`` -- no ``--config``, no ``--model``, no ``--input`` chain."""
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    result = invoke(runner, "infer", run_dir, str(page), "--save-layout")
    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in (run_dir / "predictions").glob("*.png"))
    assert written == ["page_layout.png", "page_prediction.png"]


def test_infer_without_save_writes_nothing_and_reports_it(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    result = invoke(runner, "infer", run_dir, str(page), "--no-save")
    assert result.exit_code == 0, result.output
    assert "nothing saved" in result.output


def test_infer_writes_a_json_answer_for_a_label_only_task(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A classification checkpoint produces no image; the legacy script saved nothing, we write JSON."""
    config = {**BASE, "task": "classification", "n_classes": 2}
    directory = tmp_path / "run"
    directory.mkdir()
    save_checkpoint(
        directory / "model.pt",
        create_model(SBBConfig.model_validate(config), pretrained=False),
        task="classification",
        config=config,
    )
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))

    result = invoke(runner, "infer", directory, page)
    assert result.exit_code == 0, result.output
    answer = json.loads((directory / "predictions" / "page_prediction.json").read_text())
    assert answer["task"] == "classification" and answer["label"].startswith("class_")
    assert len(answer["probabilities"]) == 2


def test_infer_needs_something_to_predict_on(runner: CliRunner, run_dir: Path) -> None:
    result = invoke(runner, "infer", run_dir)
    assert result.exit_code == 2
    assert "no images to predict on" in result.output


def test_infer_reads_the_architecture_from_the_checkpoint_unless_overridden(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    """``--architecture`` is the legacy script's other ``--model``; the path is the argument now."""
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    good = invoke(runner, "infer", run_dir, str(page), "--architecture", "resnet50_unet_light")
    assert good.exit_code == 0, good.output
    bad = invoke(runner, "infer", run_dir, str(page), "--architecture", "resnet50_classifier")
    assert bad.exit_code == 1  # a Conv/Linear state dict cannot load into the light UNet
    assert "Error:" in bad.output


# ----------------------------------------------------------------------------- sbb export


def test_export_normalises_a_legacy_pth_into_model_pt(
    runner: CliRunner, tmp_path: Path, light_model
) -> None:
    """The job ``build_model_load_pretrained_weights_and_save.py`` did by editing constants."""
    legacy = tmp_path / "old" / "weights.pth"
    legacy.parent.mkdir()
    torch.save({"model_state_dict": light_model.state_dict()}, legacy)

    result = invoke(runner, "export", legacy, "-o", legacy.parent / "model.pt")
    assert result.exit_code == 0, result.output
    exported = load_checkpoint(legacy.parent / "model.pt")
    # the source was pre-schema (0 / "unknown"); the file written declares what it actually is
    assert exported.task is None
    assert exported.schema_version == 1 and exported.package_version == __version__
    assert exported.model_state.keys() == light_model.state_dict().keys()
    assert "parameters" in result.output and "config embedded" in result.output


def test_export_reports_a_directory_with_no_checkpoint(runner: CliRunner, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = invoke(runner, "export", empty)
    assert result.exit_code == 1
    assert "no checkpoint found" in result.output


# --------------------------------------------------------------------------- sbb validate


def test_validate_reports_the_config_and_its_advisories(
    runner: CliRunner, config_file: Path
) -> None:
    """What ``sbb validate`` replaces: reading ``config_params.json`` and hoping it is consistent."""
    assert "is a valid SBBConfig (segmentation)" in invoke(runner, "validate", config_file).output


def test_validate_strict_turns_advisories_into_a_failure(
    runner: CliRunner, config_file: Path
) -> None:
    """BASE has no page2label, which is an advisory: printed, and fatal only with ``--strict``."""
    assert invoke(runner, "validate", config_file).exit_code == 0
    strict = invoke(runner, "validate", config_file, "--strict")
    assert strict.exit_code == 1
    assert "warning:" in strict.output


def test_validate_turns_a_config_error_into_one_line(runner: CliRunner, tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    result = invoke(runner, "validate", broken)
    assert result.exit_code == 1
    assert result.output.startswith("Error:")
    assert "not valid JSON" in result.output


# ----------------------------------------------------------------------- sbb config-schema


def test_config_schema_writes_the_generated_reference(runner: CliRunner, tmp_path: Path) -> None:
    destination = tmp_path / "reference.json"
    result = invoke(runner, "config-schema", "-o", destination)
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["title"] == "SBBConfig"


# ------------------------------------------------------------------------------ sbb gt


@pytest.fixture
def xml_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "xml"
    directory.mkdir()
    (directory / "page.xml").write_bytes(
        PAGE_XML.format(size=SIZE, edge=SIZE - 4).encode("iso-8859-5")
    )
    return directory


def test_gt_visualize_renders_one_page(runner: CliRunner, tmp_path: Path, xml_dir: Path) -> None:
    destination = tmp_path / "rendered.png"
    result = invoke(runner, "gt", "visualize", xml_dir / "page.xml", "-o", destination)
    assert result.exit_code == 0, result.output
    assert destination.exists()


def test_gt_visualize_defaults_to_a_type_named_file(runner: CliRunner, xml_dir: Path) -> None:
    result = invoke(runner, "gt", "visualize", xml_dir / "page.xml", "--type-output", "2d")
    assert result.exit_code == 0, result.output
    assert (xml_dir / "page_2d.png").exists()


def test_gt_generate_dry_run_lists_the_files_it_would_read(
    runner: CliRunner, tmp_path: Path, xml_dir: Path
) -> None:
    out = tmp_path / "gt"
    result = invoke(
        runner, "gt", "generate", "--xml", xml_dir, "-o", out, "--use-case", "layout", "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert "page.xml" in result.output
    assert not out.exists()


def test_gt_generate_needs_a_use_case_or_a_config(
    runner: CliRunner, tmp_path: Path, xml_dir: Path, config_file: Path
) -> None:
    """The old script's two ways to say what to build: a config file, or the flags around it."""
    out = tmp_path / "gt"
    missing = invoke(runner, "gt", "generate", "--xml", xml_dir, "-o", out)
    assert missing.exit_code == 2 and "give --config or --use-case" in missing.output
    both = invoke(
        runner,
        "gt",
        "generate",
        "--xml",
        xml_dir,
        "-o",
        out,
        "--use-case",
        "layout",
        "--config",
        config_file,
    )
    assert both.exit_code == 2 and "not both" in both.output
