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
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
from click.testing import CliRunner, Result
from pytorch_lightning.callbacks import Callback, EarlyStopping

from sbb_pytorch import __version__, cli
from sbb_pytorch.cli import main
from sbb_pytorch.config import SBBConfig, TaskType
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.gt import contour_bbox, parse_pagexml
from sbb_pytorch.inference import Prediction
from sbb_pytorch.models import create_model
from sbb_pytorch.storage import Checkpoint, load_checkpoint, safetensors_available, save_checkpoint
from sbb_pytorch.storage.images import imwrite, read_mask
from sbb_pytorch.training.callbacks import build_callbacks, load_callbacks_file

#: The safetensors half of ``infer``/``export`` is skipped, not failed, where the package is
#: missing -- the same marker ``tests/test_checkpoint.py`` and ``tests/test_inference.py`` define.
needs_safetensors = pytest.mark.skipif(
    not safetensors_available(), reason="needs the safetensors package"
)

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

# Two regions, because a reading-order sample needs a pair to order.
PAGE_XML_TWO = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/schema/v2019.07.24">
  <Page imageFilename="page.png" imageWidth="{size}" imageHeight="{size}">
    <TextRegion ID="r1" type="paragraph">
      <Coords points="4,4 14,4 14,14 4,14"/>
    </TextRegion>
    <TextRegion ID="r2" type="paragraph">
      <Coords points="18,4 28,4 28,14 18,14"/>
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


def test_model_flag_is_documented_as_a_registry_key(runner: CliRunner) -> None:
    """``--model`` takes an architecture name, not a ``backbone_type``; the help said backbone.

    The value goes to :func:`~sbb_pytorch.models.create_model` as its ``model=`` registry key
    (``resnet50_unet_light``, ``vit_resnet50_unet``, ...), while ``backbone`` names a config field
    -- ``resnet50`` or ``vit`` -- that several architectures share.
    """
    output = " ".join(invoke(runner, "train", "--help").output.split())  # undo click's wrapping
    assert "--model TEXT Registry key overriding the architecture" in output
    assert "backbone" not in output


# The two entry spellings the format allows, both naming a class that imports without a GPU.
CALLBACKS_FILE = json.dumps(
    [
        "pytorch_lightning.callbacks.Callback",
        {
            "class": "pytorch_lightning.callbacks.EarlyStopping",
            "init": {"monitor": "val_loss", "patience": 2},
        },
    ]
)


def test_a_callbacks_file_adds_to_the_run_it_is_trained_on(tmp_path: Path) -> None:
    """``--callbacks-file`` names extra callbacks; it never replaces the config's own.

    The flag exists as ``build_trainer``'s ``callbacks=`` (``training/trainer.py:111``, "Callbacks
    to use as-is instead of the built set") -- which is exactly the danger: handing the trainer the
    file's list alone would drop the ``ModelCheckpoint``\\ s and the pair of ``EarlyStopping``\\ s
    :func:`~sbb_pytorch.training.callbacks.build_callbacks` derives from the config, so
    :func:`build_callbacks` appends and the run keeps what it already had.
    """
    path = tmp_path / "callbacks.json"
    path.write_text(CALLBACKS_FILE)
    loaded = load_callbacks_file(path)
    # Both spellings resolve, and the object form hands its ``init`` to the class's __init__.
    assert [type(callback) for callback in loaded] == [Callback, EarlyStopping]
    assert (loaded[1].monitor, loaded[1].patience) == ("val_loss", 2)

    # Appended, in order: the config's own callbacks first, then everything the file named. The
    # comparison is by type because the second call builds fresh instances of the file's classes.
    config = SBBConfig.model_validate(BASE)
    directory = tmp_path / "out"
    directory.mkdir()
    own = build_callbacks(config, output_dir=directory)
    with_file = build_callbacks(config, output_dir=directory, callbacks_file=path)
    assert [type(callback) for callback in with_file] == [type(callback) for callback in own] + [
        Callback,
        EarlyStopping,
    ]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("{not json", "is not valid JSON"),
        ('{"class": "pytorch_lightning.callbacks.Callback"}', "must contain a JSON list"),
        ("3", "must contain a JSON list, got int"),
        ('[["x"]]', "entry 0 must be a dotted class path or an object"),
        (
            '["pytorch_lightning.callbacks.Callback", 3]',
            "entry 1 must be a dotted class path or an object",
        ),
        (
            '["pytorch_lightning.callbacks.NoSuchCallback"]',
            "entry 0: module 'pytorch_lightning.callbacks' has no attribute 'NoSuchCallback'",
        ),
        (
            '["pytorch_lightning.Trainer"]',
            "entry 0 'pytorch_lightning.Trainer' is a Trainer, not a pytorch_lightning Callback",
        ),
        (
            '[{"class": "pytorch_lightning.callbacks.Callback", "init": {"nope": 1}}]',
            "entry 0 'pytorch_lightning.callbacks.Callback' cannot be built",
        ),
        ('[{"class": "x.y", "monitor": "val_loss"}]', "entry 0 has unknown key(s)"),
        ('[{"class": "x.y", "init": 3}]', "entry 0 'init' must be a JSON object"),
        ('["Callback"]', "entry 0 'Callback' is not a dotted import path"),
    ],
)
def test_every_unusable_callbacks_file_names_the_entry_that_is_wrong(
    tmp_path: Path, document: str, message: str
) -> None:
    """Eleven ways the file can be wrong, and the one line each of them becomes.

    Every message carries the file *and* the entry index -- the second row above is why: the failure
    mode this replaces was a ``ModuleNotFoundError`` traceback from inside ``pl.Trainer`` that named
    neither, the same late-erroring the module docstring lists for the legacy callbacks. The wording
    mirrors :func:`~sbb_pytorch.config.loaders.load_config` so both config-shaped files refuse a
    bad file in the same voice.
    """
    path = tmp_path / "callbacks.json"
    path.write_text(document)
    with pytest.raises(ConfigError) as error:
        load_callbacks_file(path)
    assert str(path) in str(error.value)
    assert message in str(error.value)


def test_train_runs_with_a_callbacks_file(
    runner: CliRunner, tmp_path: Path, data_root: Path
) -> None:
    """The flag reaches ``build_trainer``'s ``callbacks=``: the dry run still trains and says so.

    The file names a bare ``Callback``, not ``CALLBACKS_FILE``'s ``EarlyStopping``: Lightning gives
    every stateful callback a key of ``ClassName-monitor-mode``
    (``callbacks/early_stopping.py:154-155``) and refuses two of the same key on one Trainer, so a
    second ``monitor="val_loss"`` stopper -- whatever its patience -- is the trainer's error rather
    than this flag's, and the run's own pair already covers that metric.
    """
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
    callbacks = tmp_path / "callbacks.json"
    callbacks.write_text(json.dumps(["pytorch_lightning.callbacks.Callback"]))
    result = invoke(
        runner, "train", config, "--dry-run", "--no-logger", "--callbacks-file", callbacks
    )
    assert result.exit_code == 0, result.output
    assert "dry run finished" in result.output


def test_train_reports_a_callbacks_file_it_cannot_use(
    runner: CliRunner, config_file: Path, tmp_path: Path
) -> None:
    """The file is read before the run starts, so a typo costs seconds instead of an hour."""
    broken = tmp_path / "callbacks.json"
    broken.write_text("{not json")
    result = invoke(
        runner, "train", config_file, "--dry-run", "--no-logger", "--callbacks-file", broken
    )
    assert result.exit_code == 1
    assert "Error: " in result.output
    assert "callbacks file is not valid JSON" in result.output


def test_start_weights_and_resume_conflict_in_the_cli(
    runner: CliRunner, config_file: Path, run_dir: Path
) -> None:
    """``--resume`` continues a run, ``--start-weights`` starts a new one from weights; Lightning
    would load the start file and then overwrite it with the resume file, so the CLI refuses it."""
    weights = run_dir / "model.pt"
    result = invoke(
        runner,
        "train",
        config_file,
        "--dry-run",
        "--no-logger",
        "--resume",
        weights,
        "--start-weights",
        weights,
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_start_weights_and_explicit_pretrained_conflict(
    runner: CliRunner, config_file: Path, run_dir: Path
) -> None:
    """``--pretrained`` picks which weights to *download*; giving it beside a file that already
    holds the weights asks two questions about the same slot, so it is refused rather than ignored."""
    result = invoke(
        runner,
        "train",
        config_file,
        "--dry-run",
        "--no-logger",
        "--start-weights",
        run_dir / "model.pt",
        "--no-pretrained",
    )
    assert result.exit_code == 2
    assert "--no-pretrained is redundant" in result.output


def test_start_weights_and_a_config_that_continues_conflict(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    """``continue_training`` is the config-file spelling of ``--resume``, so it is refused too."""
    config = tmp_path / "continue.json"
    config.write_text(
        json.dumps({**BASE, "continue_training": True, "dir_of_start_model": str(run_dir)})
    )
    result = invoke(
        runner,
        "train",
        config,
        "--dry-run",
        "--no-logger",
        "--start-weights",
        run_dir / "model.pt",
    )
    assert result.exit_code == 2
    assert "contradicts this config's continue_training=true" in result.output


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


def test_infer_save_default_is_the_configs_save_visualisation(
    runner: CliRunner, tmp_path: Path, light_model: torch.nn.Module
) -> None:
    """``save_visualisation`` was a config field nothing read; it is now ``--save``'s default.

    The flag still wins in both directions -- ``--save`` on a run whose config says false writes the
    image, exactly as ``--patches`` overrides the config's ``patches``.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    quiet = tmp_path / "quiet-run"
    save_checkpoint(
        quiet / "model.pt",
        light_model,
        task="segmentation",
        epoch=1,
        step=1,
        config={**BASE, "save_visualisation": False},
    )

    result = invoke(runner, "infer", str(quiet), str(page))
    assert result.exit_code == 0, result.output
    assert "nothing saved" in result.output
    assert not list((quiet / "predictions").glob("*.png"))

    forced = invoke(runner, "infer", str(quiet), str(page), "--save")
    assert forced.exit_code == 0, forced.output
    assert (quiet / "predictions" / "page_prediction.png").exists()


def test_infer_save_logits_writes_the_tensor_the_prediction_keeps(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    """``Prediction.logits`` was documented as kept "for ``--save-logits`` style debugging".

    The flag did not exist, so nothing could reach the tensor. It is now ``torch.save``d next to the
    images and survives the round trip at the model's own output size, not the resized image's.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    result = invoke(runner, "infer", run_dir, str(page), "--save-logits")
    assert result.exit_code == 0, result.output
    saved = torch.load(run_dir / "predictions" / "page_logits.pt", map_location="cpu")
    assert saved.ndim == 3, "(n_classes, H, W) -- the logits, not a rendered image"


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


@needs_safetensors
def test_infer_accepts_a_safetensors_checkpoint(
    runner: CliRunner, tmp_path: Path, light_model
) -> None:
    """``sbb infer run/model.safetensors page.png`` -- a file with no ``model_config.json`` beside
    it, because its config is in the file's own ``__metadata__``, and that is enough to rebuild."""
    import safetensors.torch

    exported = tmp_path / "model.safetensors"
    Checkpoint(
        model_state=light_model.state_dict(),
        config=SBBConfig.model_validate(BASE).model_dump(mode="json"),
        task="segmentation",
        epoch=3,
    ).save_safetensors(exported)
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))

    result = invoke(runner, "infer", exported, str(page), "--no-save")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "model_config.json").exists()  # the sidecar is inside the file now
    with safetensors.torch.safe_open(exported, framework="pt") as handle:
        assert handle.metadata()["sbb.task"] == "segmentation"


def test_infer_format_flag_picks_the_extension(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--format`` says which checkpoint a directory means; left alone, ``model.pt`` and then the
    other ``.pt``/``.pth`` outrank a ``.safetensors``. The stub records the path it was handed, so
    the assertion is about the CLI's resolution -- and the two files are one byte each on purpose."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "model.ckpt").write_bytes(b"x")
    (run / "model.safetensors").write_bytes(b"x")
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    stub = _stub_predictor(
        monkeypatch,
        Prediction(task=TaskType.SEGMENTATION, mask=np.zeros((SIZE, SIZE), dtype=np.uint8)),
    )

    invoke(runner, "infer", run, str(page), "--no-save")
    assert stub.seen[-1] == str(run / "model.ckpt")  # the ranking, which the flag then overrides
    invoke(runner, "infer", run, str(page), "--no-save", "--format", "safetensors")
    assert stub.seen[-1] == str(run / "model.safetensors")


def test_infer_overlap_reaches_the_sliding_window(
    runner: CliRunner, tmp_path: Path, run_dir: Path
) -> None:
    """``--overlap`` was only settable in Python; it is now a flag, checked before the run.

    ``--patches`` asked for and ``--overlap`` given is a coherent pair, while an overlap without
    ``--patches`` windows nothing -- so it is refused rather than quietly ignored.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    assert invoke(runner, "infer", run_dir, page, "--patches", "--overlap", "0.25").exit_code == 0

    outside = invoke(runner, "infer", run_dir, page, "--patches", "--overlap", "1.5")
    assert outside.exit_code == 2 and "overlap must be in [0, 1)" in outside.output
    unused = invoke(runner, "infer", run_dir, page, "--no-patches", "--overlap", "0.25")
    assert unused.exit_code == 2 and "--overlap tunes the --patches window" in unused.output


def test_infer_ground_truth_prints_the_numbers_it_used_to_throw_away(
    runner: CliRunner, tmp_path: Path, run_dir: Path, data_root: Path
) -> None:
    """``--ground_truth`` computed an IoU in the legacy script and never showed it; now it prints."""
    result = invoke(
        runner,
        "infer",
        run_dir,
        data_root / "images" / "page_0.png",
        "--ground-truth",
        data_root / "labels",
    )
    assert result.exit_code == 0, result.output
    assert "page_0.png" in result.output
    assert "miou=" in result.output and "accuracy=" in result.output and "dice=" in result.output


def test_infer_ground_truth_refuses_a_task_with_no_mask(runner: CliRunner, tmp_path: Path) -> None:
    """A classification page answers with a label, and a confusion matrix cannot compare one."""
    config = {**BASE, "task": "classification", "n_classes": 2}
    directory = tmp_path / "run"
    directory.mkdir()
    save_checkpoint(
        directory / "model.pt",
        create_model(SBBConfig.model_validate(config), pretrained=False),
        task="classification",
        config=config,
    )
    pages, labels = tmp_path / "pages", tmp_path / "labels"
    pages.mkdir()
    labels.mkdir()
    imwrite(pages / "page.png", np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    imwrite(labels / "page.png", np.zeros((SIZE, SIZE), dtype=np.uint8))

    result = invoke(runner, "infer", directory, pages / "page.png", "--ground-truth", labels)
    assert result.exit_code == 1
    assert "gave no mask for task=classification" in result.output


def _stub_predictor(
    monkeypatch: pytest.MonkeyPatch,
    prediction: Prediction,
    *,
    config: dict[str, object] | None = None,
) -> type:
    """Answer every page with *prediction*, so the assertion is about the XML and not the weights.

    The ``--xml-file`` tests need to know the mask to know the regions a run should write, and the
    session's model is untrained -- its mask would say nothing about the writer. The stub mirrors
    the two methods the command calls (:meth:`~sbb_pytorch.inference.Predictor.from_checkpoint` and
    :meth:`~sbb_pytorch.inference.Predictor.predict_file`) and nothing else, so a signature change
    on the real predictor breaks these exactly as it breaks the other ``infer`` tests. The class is
    returned, and it remembers the checkpoint each call was handed -- which is how the ``--format``
    test checks the CLI's answer rather than the model's. ``config`` replaces ``BASE`` where a test's
    assertion depends on the mapping the run was trained with, since a mask's class ids mean nothing
    without the ``page2label`` that named them.
    """

    # Built outside the class body: a class namespace is not an enclosing scope, so `config` would
    # be unbound inside a class that also assigns a name by that name.
    resolved = SBBConfig.model_validate(config if config is not None else BASE)

    class Stub:
        config = resolved
        seen: ClassVar[list[str]] = []

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @classmethod
        def from_checkpoint(cls, checkpoint: object, **_kwargs: object) -> Stub:
            cls.seen.append(str(checkpoint))
            return cls()

        def predict_file(self, path: str | Path, **_kwargs: object) -> Prediction:
            return replace(prediction, path=Path(path))

    monkeypatch.setattr(cli, "Predictor", Stub)
    return Stub


def test_infer_xml_file_writes_the_regions_the_prediction_found(
    runner: CliRunner, tmp_path: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One page, one file: the answer replaces that page's XML, which is what upstream promised.

    ``inference.py:731-735``: "xml file with layout coordinates that reading order detection will
    be implemented on. The result will be written in the same xml file." The rewrite keeps what the
    file knew -- ``@imageFilename``, the declared size -- and gains one region per predicted contour
    (:func:`sbb_pytorch.cli._regions_for_mask`), which is what a *layout* answer is: the config has
    no ``page2label``, so its task decides the shape and the shape is one flat region per class.

    The class name is the pair the assertion checks. ``class_1`` is the placeholder ``n_classes``
    stands in for a name a table would give, not a name, so ``@type`` stays empty (the enumeration is
    closed) and ``@custom`` says the element's own word, ``text_zone`` -- the word escriptorium uses
    for a block it has no class for, and the one Aletheia accepts.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    source = tmp_path / "labels" / "page.xml"
    source.parent.mkdir()
    source.write_text(PAGE_XML.format(size=SIZE, edge=28))

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[4:12, 4:28], mask[16:24, 4:28] = 1, 2
    _stub_predictor(monkeypatch, Prediction(task=TaskType.SEGMENTATION, mask=mask))

    result = invoke(runner, "infer", run_dir, page, "--no-save", "--xml-file", source)
    assert result.exit_code == 0, result.output
    assert "1 PageXML file(s) written" in result.output

    written = parse_pagexml(source)
    assert (written.width, written.height, written.file_name) == (SIZE, SIZE, "page")
    assert [
        (
            region.region_id,
            region.type_name,
            region.custom,
            contour_bbox(region.contour, (SIZE, SIZE)),
        )
        for region in written.regions
    ] == [
        ("page_1_0", "", "structure {type:text_zone;}", (4, 4, 24, 8)),
        ("page_2_0", "", "structure {type:text_zone;}", (4, 16, 24, 8)),
    ]


def test_infer_xml_writes_no_region_for_a_predicted_boundary_class(
    runner: CliRunner, tmp_path: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A textline run predicts three classes, and only one of them is something to write a region as.

    The boundary ring the label generator carves between lines is a class of its own, so the model
    predicts it as contours of its own -- and :func:`sbb_pytorch.cli._textline_regions` turned every
    class the mask held into a block of lines, writing each ring as a ``TextRegion`` of thin
    fragments beside the lines it rings. The writer now asks the config which ids name page content
    (:meth:`~sbb_pytorch.config.models.Page2Label.region_class_ids`), so the ring is predicted,
    trained on and never written: one block, one line, and no ``class_2`` region anywhere.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    source = tmp_path / "labels" / "page.xml"
    source.parent.mkdir()
    source.write_text(PAGE_XML.format(size=SIZE, edge=28))

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[4:12, 4:28] = 1  # a line the model found
    mask[16:24, 4:28] = 2  # a ring around a second line, which it never found as class 1
    _stub_predictor(
        monkeypatch,
        Prediction(task=TaskType.SEGMENTATION, mask=mask),
        config={
            **BASE,
            "page2label": {"use_case": "textline", "case_label": 1, "artificial_class_label": 2},
        },
    )

    result = invoke(runner, "infer", run_dir, page, "--no-save", "--xml-file", source)
    assert result.exit_code == 0, result.output
    assert [(region.region_id, region.custom) for region in parse_pagexml(source).regions] == [
        ("page_1_0", "structure {type:text_zone;}"),  # the block the line forms
        ("page_1_0_0", "structure {type:text_line;}"),  # and the line itself
    ]


def test_infer_xml_file_names_one_file_per_page_when_the_run_has_several(
    runner: CliRunner, tmp_path: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several pages cannot share the one file the flag names, so the directory plus each stem wins.

    That pairing rule is the one the port already uses for every per-page artefact
    (:func:`sbb_pytorch.data.dataset.label_path_for`), and it is the half of upstream's
    ``--page-xml`` (``inference.py:736-740``) that ``--xml_file`` never had to answer.
    """
    pages = tmp_path / "pages"
    pages.mkdir()
    for name in ("a", "b"):
        imwrite(pages / f"{name}.png", np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[4:12, 4:28] = 1
    _stub_predictor(monkeypatch, Prediction(task=TaskType.SEGMENTATION, mask=mask))

    result = invoke(
        runner, "infer", run_dir, pages, "--no-save", "--xml-file", tmp_path / "out/run.xml"
    )
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (tmp_path / "out").glob("*.xml")) == ["a.xml", "b.xml"]
    assert "2 PageXML file(s) written" in result.output


def test_infer_xml_file_reports_a_page_it_cannot_describe(
    runner: CliRunner, tmp_path: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing file for a label answer, and a junk file for a mask answer: each is one line.

    ``--xml-file`` takes ``exists=False`` because the file is usually the one being created -- that
    is what upstream's ``--page-xml`` did -- so the cases where nothing *can* be written have to say
    so instead of writing a stub over a page. A label answers with an order stamped onto an existing
    page, so a page with no XML has nothing to order; a file that is not PageXML raises
    :class:`~sbb_pytorch.exceptions.PageXMLError` from the read, printed once and left unread, in the
    same voice ``--ground-truth`` uses above.
    """
    page = tmp_path / "page.png"
    imwrite(page, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))

    _stub_predictor(monkeypatch, Prediction(task=TaskType.CLASSIFICATION, label="class_1"))
    missing = invoke(runner, "infer", run_dir, page, "--no-save", "--xml-file", tmp_path / "x.xml")
    assert missing.exit_code == 1
    assert "--xml-file has no page to order: page.png" in missing.output
    assert not (tmp_path / "x.xml").exists()  # reported, not silently written over

    junk = tmp_path / "junk.xml"
    junk.write_text("<not-a-page/>")
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[4:12, 4:28] = 1
    _stub_predictor(monkeypatch, Prediction(task=TaskType.SEGMENTATION, mask=mask))
    unreadable = invoke(runner, "infer", run_dir, page, "--no-save", "--xml-file", junk)
    assert unreadable.exit_code == 1
    assert "expected a <PcGts> root" in unreadable.output
    assert junk.read_text() == "<not-a-page/>"


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


@needs_safetensors
def test_export_writes_safetensors_when_asked(
    runner: CliRunner, tmp_path: Path, light_model
) -> None:
    """``--format safetensors`` writes ``model.safetensors``: the same tensors and the same fields,
    in the format with no pickle in it -- so the re-stamped schema and package versions are read
    from the file's metadata, which is the only place they can live there."""
    import safetensors.torch

    legacy = tmp_path / "old" / "weights.pth"
    legacy.parent.mkdir()
    torch.save({"model_state_dict": light_model.state_dict()}, legacy)

    result = invoke(runner, "export", legacy, "--format", "safetensors")
    assert result.exit_code == 0, result.output
    target = legacy.parent / "model.safetensors"
    assert target.exists()
    with safetensors.torch.safe_open(target, framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata is not None
    assert metadata["format"] == "pt"
    assert metadata["sbb.schema_version"] == "1"
    assert metadata["sbb.package_version"] == __version__  # the re-stamp travels into the file
    assert not (legacy.parent / "model_config.json").exists()  # no sidecar for this format


def test_export_dash_o_extension_wins_over_format(
    runner: CliRunner, tmp_path: Path, light_model
) -> None:
    """``-o model.pt --format safetensors``: a named file's own extension is the answer, and
    ``--format`` only names the file when ``-o`` names a directory or nothing at all."""
    legacy = tmp_path / "old" / "weights.pth"
    legacy.parent.mkdir()
    torch.save({"model_state_dict": light_model.state_dict()}, legacy)
    named = tmp_path / "out" / "model.pt"
    named.parent.mkdir()

    result = invoke(runner, "export", legacy, "-o", named, "--format", "safetensors")
    assert result.exit_code == 0, result.output
    assert named.exists()
    assert not (tmp_path / "out" / "model.safetensors").exists()
    assert load_checkpoint(named).schema_version == 1


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


def test_config_schema_default_is_absolute_not_the_current_directory(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``-o`` the schema used to land in ``./docs``, wherever the caller happened to stand.

    The default is now derived from the package (``DEFAULT_SCHEMA_PATH``), so an omitted ``-o``
    still means "the reference beside the code it documents".
    """
    destination = tmp_path / "beside-the-package.json"
    assert cli.DEFAULT_SCHEMA_PATH.is_absolute()
    monkeypatch.setattr(cli, "DEFAULT_SCHEMA_PATH", destination)
    result = invoke(runner, "config-schema")
    assert result.exit_code == 0, result.output
    assert destination.exists() and "SBBConfig" in destination.read_text()


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


def test_gt_generate_reads_a_parent_folder_of_child_datasets(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--xml`` on a folder whose children are datasets reads every one of them.

    A dataset folder holding nothing but child dataset folders -- ``smp-private/`` with one export
    per subfolder -- has no XML files of its own, so the one-level listing answered it with
    ``no ALTO/PageXML files in ...`` and a command that looked like it covered three datasets had
    read none. The listing is recursive now, a dry run names each file relative to ``--xml`` so the
    child it came from is visible, and ``--no-recursive`` is the old single-folder answer.
    """
    parent = tmp_path / "smp-private"
    child = parent / "sam_44_mss_variants"
    child.mkdir(parents=True)
    (child / "000123456-1800.xml").write_bytes(PAGE_XML.format(size=SIZE, edge=SIZE - 4).encode())
    gt_config = tmp_path / "gt_config.json"
    gt_config.write_text(
        json.dumps(
            {
                "use_case": "layout",
                "type_output": "2d",
                "page2label": {"use_case": "layout", "textregions": {"paragraph": 1}},
            }
        )
    )
    out = tmp_path / "gt"

    dry = invoke(
        runner,
        "gt",
        "generate",
        "--xml",
        parent,
        "-o",
        out,
        "--config",
        gt_config,
        "--dry-run",
    )
    assert dry.exit_code == 0, dry.output
    assert "sam_44_mss_variants/000123456-1800.xml" in dry.output  # named through its child folder
    assert not out.exists()

    result = invoke(runner, "gt", "generate", "--xml", parent, "-o", out, "--config", gt_config)
    assert result.exit_code == 0, result.output
    assert [p.name for p in out.glob("*.png")] == ["1800.png"]

    flat = invoke(
        runner,
        "gt",
        "generate",
        "--xml",
        parent,
        "-o",
        tmp_path / "flat",
        "--config",
        gt_config,
        "--no-recursive",
        "--dry-run",
    )
    assert flat.exit_code == 2 and f"no ALTO/PageXML files in {parent}" in flat.output


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


def test_gt_generate_min_area_overrides_the_config_value(
    runner: CliRunner, tmp_path: Path, xml_dir: Path
) -> None:
    """``--min-area`` next to ``--config`` used to be refused; the flag now overrides the field."""
    gt_config = tmp_path / "gt_config.json"
    gt_config.write_text(json.dumps({"use_case": "textline", "min_area": 0.25}))
    result = invoke(
        runner,
        "gt",
        "generate",
        "--xml",
        xml_dir,
        "-o",
        tmp_path / "gt",
        "--config",
        gt_config,
        "--min-area",
        "0.01",
        "--dry-run",
    )
    assert result.exit_code == 0, result.output
    assert "page.xml" in result.output


def test_gt_generate_honours_the_configs_own_type_output(
    runner: CliRunner, tmp_path: Path, xml_dir: Path
) -> None:
    """The first test to actually run ``sbb gt generate`` rather than ``--dry-run`` it, because the
    bug was in a line dry runs never reach: the command passed its own ``--type-output`` flag,
    ``3d`` by default, to the generator and never consulted ``GroundTruthConfig``, which had no
    ``type_output`` field to consult. A ``"2d"`` config therefore produced the palette rendering --
    class id 1 as the colour ``(255, 0, 0)`` -- and ``sbb train`` on that folder failed with
    ``no class id configured for label values [255]``.
    """
    gt_config = tmp_path / "gt_config.json"
    gt_config.write_text(
        json.dumps(
            {
                "use_case": "layout",
                "type_output": "2d",
                "page2label": {"use_case": "layout", "textregions": {"paragraph": 1}},
            }
        )
    )
    out = tmp_path / "gt"
    result = invoke(runner, "gt", "generate", "--xml", xml_dir, "-o", out, "--config", gt_config)
    assert result.exit_code == 0, result.output
    label = read_mask(out / "page.png")
    assert label.shape == (SIZE, SIZE)
    assert set(np.unique(label).tolist()) == {0, 1}  # class ids, not the palette's 255


def test_gt_generate_type_output_flag_beats_the_config(
    runner: CliRunner, tmp_path: Path, xml_dir: Path
) -> None:
    """Naming the flag still wins over the file -- the precedence ``sbb infer --save`` uses -- and
    ``3d`` here writes the palette colour ``(255, 0, 0)`` for class 1, whose channel 0 the label
    reader reads back as ``255``."""
    gt_config = tmp_path / "gt_config.json"
    gt_config.write_text(
        json.dumps(
            {
                "use_case": "layout",
                "type_output": "2d",
                "page2label": {"use_case": "layout", "textregions": {"paragraph": 1}},
            }
        )
    )
    out = tmp_path / "gt"
    result = invoke(
        runner,
        "gt",
        "generate",
        "--xml",
        xml_dir,
        "-o",
        out,
        "--config",
        gt_config,
        "--type-output",
        "3d",
    )
    assert result.exit_code == 0, result.output
    assert set(np.unique(read_mask(out / "page.png")).tolist()) == {0, 255}


def test_gt_enhance_writes_a_degraded_page_and_its_label_per_scale(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``image_enhancement`` from ``generate_gt_for_training.py``, and its pairing by filename."""
    images = tmp_path / "images"
    images.mkdir()
    imwrite(images / "page.png", np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps({"scales": [0.5, 0.25]}))
    out = tmp_path / "enhanced"

    dry = invoke(runner, "gt", "enhance", "-i", images, "-o", out, "--scales", scales, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "2 page(s) x 2 scale(s)" in dry.output or "1 page(s) x 2 scale(s)" in dry.output
    assert not out.exists()

    result = invoke(runner, "gt", "enhance", "-i", images, "-o", out, "--scales", scales)
    assert result.exit_code == 0, result.output
    names = ["page_0.png", "page_1.png"]
    assert sorted(p.name for p in out.glob("*.png")) == names
    # the label half: the untouched original, under the same name as each degraded page
    assert sorted(p.name for p in (tmp_path / "enhanced_labels").glob("*.png")) == names


def test_gt_reading_order_writes_the_pairwise_samples(runner: CliRunner, tmp_path: Path) -> None:
    """``machine_based_reading_order`` became ``sbb gt reading-order``: a pair per region order."""
    xml = tmp_path / "xml"
    xml.mkdir()
    (xml / "page.xml").write_bytes(PAGE_XML_TWO.format(size=SIZE).encode("utf-8"))
    out = tmp_path / "reading_order"

    dry = invoke(runner, "gt", "reading-order", "--xml", xml, "-o", out, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "2 pair(s) would be written" in dry.output
    assert not (out / "classes").exists()

    result = invoke(runner, "gt", "reading-order", "--xml", xml, "-o", out)
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.png"))) == 2  # the two ordered pairs of one page
    assert len(list((out / "classes").glob("*.npy"))) == 2
