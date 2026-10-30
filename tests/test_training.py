"""The Lightning module, the callbacks and the trainer -- replacing ``train.py``/``training.py``.

One ``pl.Trainer`` run over a four-page dataset covers the module, the callbacks and the final
checkpoint together, because that is how ``sbb train`` uses them; the rest is checked piece by
piece so that a failure names the piece rather than the fit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.exceptions import ConfigError, TrainingError
from sbb_pytorch.losses import CombinedLoss, SoftDiceLoss
from sbb_pytorch.storage.checkpoint import Checkpoint
from sbb_pytorch.storage.images import imwrite
from sbb_pytorch.training import (
    SBBLightningModule,
    SBBModelSummary,
    SBBProgressBar,
    StepCheckpoint,
    build_callbacks,
    build_trainer,
    checkpoint_for,
    default_criterion,
    early_stopping_fields,
    early_stopping_for,
    fit,
    metric_mode,
    metric_snapshot,
    monitor_metrics,
    resolve_accelerator,
    resolve_logger,
    unpack_batch,
)
from sbb_pytorch.training.factory import class_weights_from_directory

SIZE = 32
_IMAGE = torch.zeros(2, 3, 4, 4)
_MASK = torch.zeros(2, 4, 4)


def make(**overrides: object) -> SBBConfig:
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": SIZE,
        "input_width": SIZE,
        "n_epochs": 1,
        "n_batch": 2,
        "num_workers": 0,
        "device": "cpu",
        "augmentation": False,
        **overrides,
    }
    return SBBConfig.model_validate(data)


@pytest.fixture
def stub() -> nn.Module:
    """A 1x1 conv: ``(B,3,H,W) -> (B,3,H,W)``, so loss and metrics agree on the shape."""
    return nn.Conv2d(3, 3, 1)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A split root: ``images/`` plus ``labels/``, four pages, three classes."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    gradient = np.tile(np.linspace(0, 255, SIZE, dtype=np.uint8), (SIZE, 1))
    codes = np.tile(np.arange(SIZE, dtype=np.int64) % 3, (SIZE, 1)).astype(np.uint8)
    for index in range(4):
        imwrite(images / f"page_{index}.png", np.stack([gradient] * 3, axis=-1))
        imwrite(labels / f"page_{index}.png", codes)
    return tmp_path


# --------------------------------------------------------------------- the module


def test_module_wraps_the_given_model(stub: nn.Module) -> None:
    """``SegmentationModel`` took a model and hid it behind ``self.model``; so does this."""
    module = SBBLightningModule(make(), stub)
    x = torch.zeros(2, 3, SIZE, SIZE)
    assert module.model is stub
    assert torch.equal(module(x), stub(x))


def test_a_given_model_keeps_its_weights(stub: nn.Module) -> None:
    """``train.py`` re-initialised every model it was handed, pretrained or not."""
    before = stub.weight.detach().clone()
    SBBLightningModule(make(), stub)
    assert torch.equal(stub.weight, before)


def test_an_explicit_scheme_is_applied_to_a_given_model(stub: nn.Module) -> None:
    before = stub.weight.detach().clone()
    SBBLightningModule(make(), stub, init_method="xavier")
    assert not torch.equal(stub.weight, before)


def test_default_criterion_follows_the_loss_fields() -> None:
    """``weighted_loss`` and ``is_loss_soft_dice`` were read by only one of the three copies."""
    assert isinstance(default_criterion(make()), CombinedLoss)
    assert isinstance(default_criterion(make(is_loss_soft_dice=True)), SoftDiceLoss)


def test_default_criterion_reads_weights_from_the_label_folder(data_root: Path) -> None:
    """``dir_train`` is a split root here, so the labels come from ``dir_train/labels``."""
    criterion = default_criterion(make(dir_train=str(data_root), weighted_loss=True))
    assert isinstance(criterion, CombinedLoss)
    assert class_weights_from_directory(data_root, 3).shape == (3,)


@pytest.mark.parametrize(
    ("batch", "expected_mask"),
    [((_IMAGE, _MASK), _MASK), ({"image": _IMAGE, "mask": _MASK}, _MASK), (_IMAGE, None)],
    ids=["tuple", "dict", "tensor"],
)
def test_unpack_batch_accepts_every_batch_shape(
    batch: object, expected_mask: torch.Tensor | None
) -> None:
    images, mask = unpack_batch(batch)
    assert torch.equal(images, _IMAGE)
    assert torch.equal(mask, expected_mask) if expected_mask is not None else mask is None


def test_unpack_batch_rejects_junk() -> None:
    with pytest.raises(TrainingError, match="batch type"):
        unpack_batch([torch.zeros(2, 3, 4, 4)])
    with pytest.raises(TrainingError, match="no 'mask'"):
        unpack_batch({"image": torch.zeros(2, 3, 4, 4)})


def test_configure_optimizers_returns_optimizer_and_scheduler(stub: nn.Module) -> None:
    """The ``ReduceLROnPlateau`` ``train.py`` imported but never constructed is now returned."""
    module = SBBLightningModule(make(optimizer="sgd", learning_rate=1e-3), stub)
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.SGD)
    assert config["optimizer"].param_groups[0]["lr"] == pytest.approx(1e-3)
    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert config["lr_scheduler"]["monitor"] == "val_loss"


def test_schedule_can_be_turned_off(stub: nn.Module) -> None:
    """The legacy return value -- an optimizer alone -- is still reachable."""
    config = SBBLightningModule(make(), stub, schedule=False).configure_optimizers()
    assert set(config) == {"optimizer"}


def test_logged_val_keys_names_the_two_metrics_a_run_produces(stub: nn.Module) -> None:
    module = SBBLightningModule(make(), stub)
    assert module.logged_val_keys == ("val_loss", "val_accuracy")


def test_logged_val_keys_without_metrics_has_no_best(stub: nn.Module) -> None:
    """Enhancement and reading order log a loss only; the second checkpoint has to notice."""
    module = SBBLightningModule(make(), stub, val_metrics={})
    assert module.logged_val_keys == ("val_loss", None)


# ------------------------------------------------------------------- the callbacks


def test_metric_mode_is_min_only_for_losses() -> None:
    assert metric_mode("val_loss") == "min"
    assert metric_mode("val_accuracy") == "max"
    assert metric_mode("train_loss") == "min"


def test_monitor_metrics_is_the_legacy_pair(stub: nn.Module) -> None:
    """``train.py`` registered two checkpoints and two early-stopping callbacks, always."""
    assert monitor_metrics(SBBLightningModule(make(), stub)) == (
        ("val_loss", "min"),
        ("val_accuracy", "max"),
    )


def test_monitor_metrics_drops_a_metric_the_task_never_logs(stub: nn.Module) -> None:
    assert monitor_metrics(SBBLightningModule(make(), stub, val_metrics={})) == (
        ("val_loss", "min"),
    )


def test_checkpoint_for_writes_a_readable_name(stub: nn.Module, tmp_path: Path) -> None:
    checkpoint = checkpoint_for(make(), "val_loss", output_dir=tmp_path)
    assert checkpoint.filename == "model-{epoch:02d}-{val_loss:.4f}"
    assert checkpoint.monitor == "val_loss" and checkpoint.mode == "min"
    assert checkpoint.best_model_path == ""
    best = checkpoint_for(make(), "val_accuracy", output_dir=tmp_path)
    assert best.filename == "best-{epoch:02d}-{val_accuracy:.4f}"


def test_early_stopping_uses_the_configured_patience(tmp_path: Path) -> None:
    stopping = early_stopping_for(make(patience=4), "val_loss")
    assert stopping.patience == 4 and stopping.min_delta == 0.0
    assert early_stopping_for(make(patience=4), "val_accuracy", patience=2).patience == 2


def test_both_callbacks_are_quiet_unless_asked(stub: nn.Module, tmp_path: Path) -> None:
    """``EarlyStopping(verbose=True)`` was hard-coded; the pair's prints are now opt-in.

    The two lines that used to follow the bar every epoch -- ``Metric val_loss improved. New best
    score: 0.449`` and ``Metric val_accuracy improved by 0.063 >= min_delta = 0.0. ...`` -- come
    from these two ``verbose`` flags, so both default to ``False`` and one flag turns them on.
    """
    assert checkpoint_for(make(), "val_loss", output_dir=tmp_path).verbose is False
    assert early_stopping_for(make(), "val_loss").verbose is False
    assert checkpoint_for(make(), "val_loss", output_dir=tmp_path, verbose=True).verbose is True
    assert early_stopping_for(make(), "val_loss", verbose=True).verbose is True


def test_build_callbacks_turns_both_verbases_on_together(stub: nn.Module, tmp_path: Path) -> None:
    """``--verbose-metrics`` sets the flag on the checkpoint *and* the stopping rule of each pair."""
    module = SBBLightningModule(make(), stub)
    quiet = build_callbacks(make(), module, output_dir=tmp_path)
    loud = build_callbacks(make(), module, output_dir=tmp_path, verbose=True)
    assert [callback.verbose for callback in quiet] == [False] * 4
    assert [callback.verbose for callback in loud] == [True] * 4


def test_early_stopping_fields_names_the_wait_of_every_attached_callback(tmp_path: Path) -> None:
    """The bar's ``early_stopping/val_loss: 3/10``: epochs waited over the configured patience."""
    config = make(patience=4)
    stopping = early_stopping_for(config, "val_loss")
    stopping.wait_count = 3
    assert early_stopping_fields(SimpleNamespace(callbacks=[stopping])) == {
        "early_stopping/val_loss": "3/4"
    }


def test_progress_bar_carries_the_counters_alongside_the_losses(
    stub: nn.Module, tmp_path: Path
) -> None:
    """``SBBProgressBar.get_metrics`` is what puts them on the bar next to ``val_loss``."""
    config = make()
    module = SBBLightningModule(config, stub)
    trainer = build_trainer(config, module, output_dir=tmp_path, logger=False)
    bar = next(callback for callback in trainer.callbacks if isinstance(callback, SBBProgressBar))
    # the bar reads ``trainer.progress_bar_metrics`` (what ``self.log(..., prog_bar=True)`` fills),
    # not ``callback_metrics``, and RichProgressBar turns each tensor into a float
    trainer.progress_bar_metrics["val_loss"] = torch.tensor(0.5)
    items = bar.get_metrics(trainer, module)
    assert items["val_loss"] == pytest.approx(0.5)
    assert items["early_stopping/val_loss"] == "0/10"
    assert items["early_stopping/val_accuracy"] == "0/10"


def test_build_callbacks_pairs_each_metric_with_its_stopping(
    stub: nn.Module, tmp_path: Path
) -> None:
    callbacks = build_callbacks(make(), SBBLightningModule(make(), stub), output_dir=tmp_path)
    assert [type(c).__name__ for c in callbacks] == [
        "ModelCheckpoint",
        "EarlyStopping",
        "ModelCheckpoint",
        "EarlyStopping",
    ]


def test_build_callbacks_without_a_module_tracks_only_the_loss(tmp_path: Path) -> None:
    callbacks = build_callbacks(make(), None, output_dir=tmp_path, early_stopping=False)
    assert [type(c).__name__ for c in callbacks] == ["ModelCheckpoint"]


def test_build_callbacks_adds_the_step_saver(tmp_path: Path) -> None:
    """``--save-interval`` became ``every_n_steps``; the callback it built never fired."""
    callbacks = build_callbacks(
        make(),
        SBBLightningModule(make(), nn.Conv2d(3, 3, 1)),
        output_dir=tmp_path,
        every_n_steps=1,
        early_stopping=False,
    )
    assert [type(c).__name__ for c in callbacks] == [
        "ModelCheckpoint",
        "ModelCheckpoint",  # the second monitored metric's checkpoint, without its stopping rule
        "StepCheckpoint",
    ]
    with pytest.raises(ValueError, match="every_n_steps"):
        StepCheckpoint(make(), every_n_steps=0, output_dir=tmp_path)


def test_metric_snapshot_skips_non_scalars() -> None:
    """``train.py`` indexed ``callback_metrics['val_acc']`` and raised after a whole run."""

    class FakeTrainer:  # duck-typed: metric_snapshot reads one attribute
        def __init__(self) -> None:
            self.callback_metrics = {
                "val_loss": torch.tensor(0.5),
                "val_accuracy": torch.tensor([0.4, 0.6]),  # multi-element: skipped
                "epoch": 3,
                "note": "ignored",
            }

    assert metric_snapshot(FakeTrainer()) == {"val_loss": 0.5, "epoch": 3.0}  # type: ignore[arg-type]


# ---------------------------------------------------------------------- the trainer


def test_resolve_accelerator_splits_a_device_spec() -> None:
    """``accelerator=device, devices=1`` could not express ``cuda:1``; this can."""
    assert resolve_accelerator("cpu") == {"accelerator": "cpu", "devices": 1}
    assert resolve_accelerator("auto")["accelerator"] in {"cpu", "cuda", "mps"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine with no GPU")
def test_resolve_accelerator_rejects_an_absent_gpu() -> None:
    with pytest.raises(ValueError, match="cuda"):
        resolve_accelerator("cuda:7")


def test_resolve_logger_falls_back_to_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sbb_pytorch.training.trainer as trainer_module

    if not trainer_module.tensorboard_available():  # pragma: no cover - the CI box has tensorboard
        assert isinstance(resolve_logger(tmp_path), CSVLogger)
    monkeypatch.setattr(trainer_module, "tensorboard_available", lambda: False)
    assert isinstance(resolve_logger(tmp_path), CSVLogger)
    monkeypatch.setattr(trainer_module, "tensorboard_available", lambda: True)
    assert isinstance(resolve_logger(tmp_path), TensorBoardLogger)


def test_build_trainer_applies_the_config_fields(stub: nn.Module, tmp_path: Path) -> None:
    """``precision``, ``accumulate_grad_batches`` and ``gradient_clip_val`` were ignored before."""
    config = make(n_epochs=3, precision="32-true", accumulate_grad_batches=2, gradient_clip_val=0.5)
    module = SBBLightningModule(config, stub)
    trainer = build_trainer(
        config,
        module,
        output_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    assert trainer.max_epochs == 3
    assert trainer.accumulate_grad_batches == 2
    assert trainer.gradient_clip_val == 0.5
    assert trainer.precision == "32-true"
    assert trainer.log_every_n_steps == 1
    assert Path(trainer.default_root_dir) == tmp_path


@pytest.mark.skipif(
    not importlib.util.find_spec("rich"), reason="needs rich to build the rich callbacks"
)
def test_build_trainer_hands_over_its_own_bar_and_summary(stub: nn.Module, tmp_path: Path) -> None:
    """The bar and the summary come from this package, and Lightning must not add a second one.

    ``Trainer`` appends a ``RichModelSummary`` itself whenever ``enable_model_summary`` is true and
    rich is installed, and warns when a summary callback is already in the list -- so handing over
    :class:`SBBModelSummary` goes together with ``enable_model_summary=False``.
    """
    module = SBBLightningModule(make(), stub)
    trainer = build_trainer(make(), module, output_dir=tmp_path, logger=False)
    types = {type(callback) for callback in trainer.callbacks}
    assert SBBProgressBar in types and SBBModelSummary in types
    # ``RichModelSummary`` derives from ``ModelSummary``, so this count is how a second summary --
    # Lightning's own, which the ``enable_model_summary=False`` in ``build_trainer`` prevents --
    # would show up. ``Trainer`` keeps the flag private in 2.6, so the list is the observable.
    assert sum(1 for callback in trainer.callbacks if isinstance(callback, ModelSummary)) == 1


@pytest.mark.parametrize("verbose", [False, True])
def test_build_trainer_passes_verbosity_to_the_callbacks_it_builds(
    stub: nn.Module, tmp_path: Path, verbose: bool
) -> None:
    trainer = build_trainer(
        make(),
        SBBLightningModule(make(), stub),
        output_dir=tmp_path,
        logger=False,
        verbose=verbose,
        enable_progress_bar=False,
    )
    built = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint | EarlyStopping)]
    assert len(built) == 4
    assert [c.verbose for c in built] == [verbose] * 4


def test_build_trainer_rejects_an_unknown_logger_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="logger"):
        build_trainer(make(), output_dir=tmp_path, logger="mlflow")


def test_fit_rejects_trainer_kwargs_alongside_a_trainer(stub: nn.Module, tmp_path: Path) -> None:
    trainer = build_trainer(make(), output_dir=tmp_path, logger=False, enable_progress_bar=False)
    with pytest.raises(TrainingError, match="ignored when trainer"):
        # build_trainer-only kwargs make no sense once a trainer already exists: say so, don't ignore
        fit(make(), model=stub, trainer=trainer, enable_progress_bar=False)


def test_fit_trains_one_epoch_and_writes_model_pt(
    stub: nn.Module, data_root: Path, tmp_path: Path
) -> None:
    """The end-to-end path ``sbb train`` takes: module, trainer, fit, then ``model.pt``."""
    config = make(
        dir_train=str(data_root),
        dir_eval=str(data_root),
        dir_output=str(tmp_path),
        data_is_provided=True,  # the fixture *is* the dataset, so nothing has to be generated
    )
    trainer = fit(
        config,
        model=stub,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    checkpoint_path = tmp_path / "model.pt"
    assert checkpoint_path.exists()
    saved = Checkpoint.load(checkpoint_path)
    assert saved.task == "segmentation"
    assert saved.config is not None and saved.config["n_classes"] == 3
    assert saved.epoch == trainer.current_epoch and saved.step == 2  # four pages, batch size 2
    assert saved.metrics["val_loss"] == pytest.approx(float(trainer.callback_metrics["val_loss"]))
    assert saved.model_state.keys() == stub.state_dict().keys()
    # epoch-end logging is what produces these; the legacy code logged them per batch instead
    assert "train_accuracy" in trainer.callback_metrics
    # the sidecar the checkpoint schema promises, so tooling can read the config unpickled
    assert (tmp_path / "model_config.json").exists()


def test_fit_writes_the_epoch_checkpoints_next_to_it(
    stub: nn.Module, data_root: Path, tmp_path: Path
) -> None:
    config = make(
        dir_train=str(data_root),
        dir_eval=str(data_root),
        dir_output=str(tmp_path),
        data_is_provided=True,
    )
    fit(config, model=stub, logger=False, enable_progress_bar=False, enable_model_summary=False)
    names = sorted(p.name for p in tmp_path.glob("*.ckpt"))
    assert len(names) == 2  # model-* and best-*, one per monitored metric
    assert any(n.startswith("model-") for n in names) and any(n.startswith("best-") for n in names)


def test_fit_can_skip_the_final_save(stub: nn.Module, data_root: Path, tmp_path: Path) -> None:
    config = make(
        dir_train=str(data_root),
        dir_eval=str(data_root),
        dir_output=str(tmp_path),
        data_is_provided=True,
    )
    fit(
        config,
        model=stub,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        save_final=False,
    )
    assert not (tmp_path / "model.pt").exists()


def test_fit_needs_data_or_a_directory(tmp_path: Path) -> None:
    """``train.py`` found out mid-epoch; the config is now checked before a trainer exists."""
    with pytest.raises(ConfigError, match="no data configured"):
        fit(make(dir_output=str(tmp_path)), model=nn.Conv2d(3, 3, 1), logger=False)
