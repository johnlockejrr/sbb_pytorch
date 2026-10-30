"""``pl.Trainer`` built from an ``SBBConfig``, and the ``fit`` the CLI calls.

Two things about ``train.py``'s trainer were wrong independent of everything around them:

* ``pl.Trainer(accelerator=device, devices=1)`` -- ``accelerator`` accepts
  ``auto|cpu|gpu|cuda|mps|tpu``, so a config ``device`` of ``"cuda:1"`` (a perfectly valid value
  for ``torch.device``) raised ``MisconfigurationException``, and ``devices`` was hard-coded to
  ``1``, so the second GPU of a machine was unreachable however it was configured.
  :func:`resolve_accelerator` splits a device spec into the pair pytorch-lightning wants, on top of
  :func:`sbb_pytorch.device.resolve_device`.
* ``torch.save({"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...,
  "val_loss": ..., "val_acc": ...}, "model.pt")`` -- and it read ``trainer.callback_metrics["val_acc"]``
  to build it, so any task that does not log ``val_acc`` raised ``KeyError`` *after* a full
  training run, throwing away the result. The final save now goes through
  :class:`~sbb_pytorch.storage.checkpoint.Checkpoint`, whose metrics field tolerates whatever the
  run logged and whose schema every reader accepts.

Everything else the legacy ``Trainer(...)`` call passed is preserved: ``max_epochs``, the
TensorBoard logger, ``log_every_n_steps=1``, the model summary, and ``ckpt_path`` for a resume --
with ``config.precision``, ``config.accumulate_grad_batches`` and ``config.gradient_clip_val``
finally honoured instead of being ignored.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from sbb_pytorch import __version__
from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.data import SBBDataModule
from sbb_pytorch.device import resolve_device
from sbb_pytorch.exceptions import TrainingError
from sbb_pytorch.storage.checkpoint import Checkpoint, safetensors_available
from sbb_pytorch.training.callbacks import (
    SBBModelSummary,
    SBBProgressBar,
    build_callbacks,
    metric_snapshot,
)
from sbb_pytorch.training.module import SBBLightningModule

__all__ = [
    "apply_start_weights",
    "best_checkpoint",
    "best_checkpoint_path",
    "build_trainer",
    "final_checkpoint",
    "fit",
    "resolve_accelerator",
    "resolve_logger",
    "tensorboard_available",
]

DataArgument = (
    pl.LightningDataModule | DataLoader | tuple[DataLoader, DataLoader] | list[DataLoader] | None
)


def resolve_accelerator(device: str | torch.device = "auto") -> dict[str, Any]:
    """The ``{accelerator, devices}`` pair pytorch-lightning wants for a device spec.

    ``cuda:1`` becomes ``{"accelerator": "cuda", "devices": [1]}`` -- the case the legacy
    ``accelerator=device, devices=1`` could not express -- and anything else keeps its own type with
    a single device. Raises whatever :func:`sbb_pytorch.device.resolve_device` raises for an
    unusable spec (an index beyond the installed GPU count, an unknown string).
    """
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        return {"accelerator": "cuda", "devices": [resolved.index if resolved.index else 0]}
    return {"accelerator": resolved.type, "devices": 1}


def tensorboard_available() -> bool:
    """Whether ``tensorboard`` or ``tensorboardX`` is importable -- what ``TensorBoardLogger`` needs.

    ``train.py`` constructed ``TensorBoardLogger`` unconditionally and crashed on machines without
    it, even though pytorch-lightning ships a CSV logger that costs nothing.
    """
    return bool(importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"))


def _rich_available() -> bool:
    return importlib.util.find_spec("rich") is not None


def resolve_logger(output_dir: str | Path, *, name: str = "lightning_logs") -> Any:
    """A TensorBoard logger when tensorboard is installed, otherwise a CSV logger.

    Both come from ``pytorch_lightning.loggers`` -- ``pl.CSVLogger`` / ``pl.TensorBoardLogger`` do
    not exist in pytorch-lightning 2.x, which is why ``train.py`` imported the class directly.
    """
    if tensorboard_available():
        return TensorBoardLogger(save_dir=str(output_dir), name=name)
    return CSVLogger(save_dir=str(output_dir), name=name)


def build_trainer(
    config: SBBConfig,
    module: SBBLightningModule | None = None,
    *,
    device: str | torch.device | None = None,
    callbacks: list[pl.Callback] | None = None,
    logger: Any = "auto",
    output_dir: str | Path | None = None,
    enable_progress_bar: bool | None = None,
    use_rich: bool | None = None,
    every_n_steps: int | None = None,
    enable_model_summary: bool = True,
    verbose: bool = False,
    **overrides: Any,
) -> pl.Trainer:
    """Assemble the ``pl.Trainer`` for ``config`` without touching a dataset.

    Args:
        config: The resolved configuration -- supplies epochs, precision, gradient accumulation,
            gradient clipping, the output directory and the early-stopping patience.
        module: The module being trained. Its logged metrics decide which checkpoints and
            early-stopping callbacks are created (see :func:`~sbb_pytorch.training.callbacks.build_callbacks`).
        device: Overrides ``config.device``; ``"auto"`` picks the best available device.
        callbacks: Callbacks to use as-is instead of the built set (``sbb train --callbacks-file``).
        logger: ``"auto"``, ``False``, or a ready ``pl`` logger.
        output_dir: Where logs and checkpoints go; defaults to ``config.dir_output``.
        enable_progress_bar: ``None`` follows ``use_rich``; ``False`` disables the bar entirely
            (pytorch-lightning refuses a bar instance together with ``enable_progress_bar=False``,
            so the two are set consistently here rather than passed separately).
        use_rich: Force the ``rich`` bar on or off; default is whenever ``rich`` is importable,
            which the package requires anyway.
        every_n_steps: Also add the mid-epoch :class:`~sbb_pytorch.training.callbacks.StepCheckpoint`.
        enable_model_summary: The summary table ``train.py`` printed on every run start, built here
            as :class:`~sbb_pytorch.training.callbacks.SBBModelSummary` (same table, no
            ``Total FLOPs: 0`` line).
        verbose: ``--verbose-metrics`` -- the ``Metric ... improved. New best score`` prints of the
            checkpoint and early-stopping callbacks this function builds. Ignored when
            ``callbacks=`` is given, since that list brings its own settings.
        **overrides: Anything else ``pl.Trainer`` takes (``fast_dev_run``, ``max_steps``,
            ``limit_train_batches``, ``strategy``, ...), applied last.

    Raises:
        ValueError: ``logger`` is neither ``"auto"``, ``False`` nor a logger instance.
    """
    directory = Path(output_dir if output_dir is not None else config.dir_output)
    directory.mkdir(parents=True, exist_ok=True)

    show_bar = True if enable_progress_bar is None else bool(enable_progress_bar)
    rich_bar = show_bar and (_rich_available() if use_rich is None else bool(use_rich))

    if callbacks is None:
        callbacks = build_callbacks(
            config, module, output_dir=directory, every_n_steps=every_n_steps, verbose=verbose
        )
    else:
        callbacks = list(callbacks)
    if rich_bar:
        # RichProgressBar, plus the early-stopping counters on its metrics line.
        callbacks.append(SBBProgressBar())

    # ``pl.Trainer`` builds the summary itself -- ``RichModelSummary`` whenever rich is installed,
    # which the package requires -- and that render ends with ``Total FLOPs: 0``: the count needs
    # ``LightningModule.example_input_array``, which this package's module has never had, so the
    # line has always reported 0 for a model with real convolutions. Handing over our own subclass
    # and turning the bool off keeps the table and drops the number; a summary callback *and*
    # ``enable_model_summary=True`` would make Lightning warn about two summaries. Without rich the
    # subclass cannot be constructed, so the bool goes through unchanged.
    own_summary = SBBModelSummary() if enable_model_summary and _rich_available() else None
    if own_summary is not None:
        callbacks.append(own_summary)

    resolved_logger: Any = False
    if isinstance(logger, str):
        if logger != "auto":
            raise ValueError(f"logger must be 'auto', False, or a logger instance, got {logger!r}")
        resolved_logger = resolve_logger(directory)
    elif logger is not None and logger is not False:
        resolved_logger = logger

    kwargs: dict[str, Any] = {
        **resolve_accelerator(device if device is not None else config.device),
        "max_epochs": config.n_epochs,
        "precision": config.precision,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "gradient_clip_val": config.gradient_clip_val,
        "logger": resolved_logger,
        "callbacks": callbacks,
        "enable_progress_bar": show_bar,
        "enable_model_summary": bool(enable_model_summary) and own_summary is None,
        "log_every_n_steps": 1,
        "default_root_dir": str(directory),
    }
    kwargs.update(overrides)
    return pl.Trainer(**kwargs)


def _dataloader_kwargs(config: SBBConfig, data: DataArgument) -> dict[str, Any]:
    """Map ``data`` onto ``trainer.fit``'s dataloader arguments.

    ``None`` builds an :class:`~sbb_pytorch.data.datamodule.SBBDataModule` from the config: that
    class owns directory resolution, the train/val split, patch generation and the loader
    settings, so ``fit`` never re-derives them. The legacy ``train.py`` built its ``DataModule``
    and then called ``train_dataloader()`` / ``val_dataloader()`` by hand for no reason.

    Raises:
        ConfigError: ``data`` is ``None`` and the config describes no data anywhere.
        TrainingError: any other shape that is not a datamodule, a loader, or a (train, val) pair.
    """
    if data is None:
        return {"datamodule": SBBDataModule(config)}
    if isinstance(data, pl.LightningDataModule):
        return {"datamodule": data}
    if isinstance(data, DataLoader):
        return {"train_dataloaders": data}
    if isinstance(data, tuple | list) and len(data) == 2:
        return {"train_dataloaders": data[0], "val_dataloaders": data[1]}
    raise TrainingError(
        f"data must be a LightningDataModule, a DataLoader, or a (train, val) pair; got {data!r}"
    )


def final_checkpoint(
    config: SBBConfig, module: SBBLightningModule, trainer: pl.Trainer
) -> Checkpoint:
    """The ``Checkpoint`` ``fit`` writes as ``model.pt``: the trainer's **last** weights.

    Extracted from ``fit``, which built it inline; the fields, the ``metric_snapshot`` and the use of
    ``module.model`` rather than the LightningModule's own state (which carries the ``model.``
    prefix) are unchanged, so ``model.pt`` holds exactly what it held before this existed.
    """
    return Checkpoint(
        model_state=dict(module.model.state_dict()),
        epoch=int(trainer.current_epoch),
        step=int(trainer.global_step),
        task=config.task.value,
        config=config.model_dump(mode="json"),
        metrics=metric_snapshot(trainer),
    )


def _best_path(callback: object) -> Path | None:
    """The one file a ``ModelCheckpoint`` says is best, or ``None`` while it has written nothing."""
    path = getattr(callback, "best_model_path", None)
    return Path(path) if isinstance(path, str) and path else None


def best_checkpoint_path(trainer: pl.Trainer) -> Path | None:
    """The ``.ckpt`` a run's own callbacks call best, or ``None`` when none was written.

    ``build_trainer`` attaches one ``ModelCheckpoint`` per monitored metric --
    ``model-07-0.3141.ckpt`` for ``val_loss`` and ``best-07-0.9812.ckpt`` for ``val_accuracy`` --
    and each exposes a public ``best_model_path`` naming a single file even at ``save_top_k=3``.
    The accuracy checkpoint is the answer, because that is the metric ``sbb train`` optimises and the
    file ``sbb export`` is pointed at by name; a task with no accuracy metric has only the loss
    callback, and its file is then the best there is. ``--dry-run`` writes no checkpoint at all,
    which is the ``None`` every caller has to handle.
    """
    callbacks = sorted(
        trainer.checkpoint_callbacks,
        key=lambda callback: 0
        if "accuracy" in str(getattr(callback, "monitor", "")).casefold()
        else 1,
    )
    for callback in callbacks:
        path = _best_path(callback)
        if path is not None and path.is_file():
            return path
    return None


def best_checkpoint(
    config: SBBConfig, module: SBBLightningModule, trainer: pl.Trainer
) -> Checkpoint:
    """The **best** weights a run produced, as the file ``model.pt``'s safetensors twin.

    Read from the ``.ckpt`` the callbacks name (:func:`best_checkpoint_path`), because that file is
    the answer to "which weights scored best" and ``model.pt`` is the answer to "which weights ran
    last". The two disagree on purpose: folding the best weights into ``model.pt`` would silently
    move a file every existing tool reads, and a reader could not tell which of the two it got -- so
    the best weights are published beside it, in the format that has no pickle in it.

    The ``.ckpt`` is loaded through :meth:`Checkpoint.load`, which knows pytorch-lightning's layout
    and records the Lightning version in ``pytorch_lightning_version`` while re-stamping
    ``package_version`` (that field answers for *this* package, and a Lightning file was not produced
    by it). A Lightning checkpoint carries no metrics, so the numbers come from
    ``metric_snapshot(trainer)``; with no checkpoint at all (``--dry-run``, ``save_final=False``) the
    export falls back to the final state, which is the same object ``model.pt`` was written from.
    """
    path = best_checkpoint_path(trainer)
    if path is None:
        return final_checkpoint(config, module, trainer)
    checkpoint = Checkpoint.load(path)
    return replace(
        checkpoint,
        task=checkpoint.task or config.task.value,
        config=checkpoint.config or config.model_dump(mode="json"),
        metrics=checkpoint.metrics or metric_snapshot(trainer),
        package_version=checkpoint.package_version or __version__,
    )


def apply_start_weights(module: SBBLightningModule, path: str | Path) -> Checkpoint:
    """Load ``path``'s weights into ``module`` before training starts, and return what was read.

    This is the finetuning entry. ``--resume`` needs a real pytorch-lightning ``.ckpt`` -- ``fit``
    hands that path to ``trainer.fit(ckpt_path=...)``, which restores the optimiser, the epoch
    counter and the scheduler along with the weights -- so a ``model.pt``, an exported
    ``model.safetensors`` or a best ``.ckpt`` from another run had nowhere to go. Only the
    architecture's weights move here: the optimiser and the epoch start fresh, which is the point of
    finetuning, and why ``fit`` consults neither ``init_method`` nor ImageNet when they are given.

    Raises whatever :meth:`Checkpoint.load`/``apply_to`` raise -- a missing file, an unreadable
    format, or weights that do not fit the architecture the config builds.
    """
    checkpoint = Checkpoint.load(path)
    checkpoint.apply_to(module)
    return checkpoint


def fit(
    config: SBBConfig,
    *,
    model: nn.Module | None = None,
    module: SBBLightningModule | None = None,
    data: DataArgument = None,
    trainer: pl.Trainer | None = None,
    device: str | torch.device | None = None,
    resume: str | Path | None = None,
    start_weights: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_final: bool = True,
    save_safetensors: bool | None = None,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """Train ``config`` end to end and return the finished ``pl.Trainer``.

    This is what ``sbb train`` calls: build the module (unless given), load ``start_weights`` into it
    (if given), build the trainer (unless given), run ``trainer.fit``, then write ``model.pt`` -- the
    file ``sbb infer`` reads -- and beside it ``model.safetensors``, the best weights in the format
    that has no pickle in it.

    Args:
        config: The resolved configuration.
        model: Model to train; omitted means ``create_model(config)`` (see
            :class:`~sbb_pytorch.training.module.SBBLightningModule` for when its weights are
            re-initialised).
        module: A ready LightningModule, e.g. one carrying a custom loss or metric set.
        data: Datamodule, loader, or ``(train, val)`` loaders; default built from the config.
        trainer: An already-built trainer (from :func:`build_trainer`); ``trainer_kwargs`` are then
            rejected rather than silently ignored.
        device: Overrides ``config.device``.
        resume: Checkpoint to resume from. Defaults to ``config.dir_of_start_model`` when
            ``config.continue_training`` is set -- the two config fields ``train.py`` read but never
            passed to ``fit(ckpt_path=...)``, which only ever took the CLI flag. Mutually exclusive
            with ``start_weights``: one restores the run, the other starts a new one from weights.
        start_weights: Weights to build the model from before training -- a ``model.pt``, a
            ``model.safetensors`` or a ``.ckpt``, read by :func:`apply_start_weights`. The
            optimiser and the epoch counter start fresh, so this is the finetuning path; it replaces
            the config's ``init_method``/ImageNet choice, which is never consulted when given.
        output_dir: Where the run lives; defaults to ``config.dir_output``.
        save_final: Write ``model.pt`` (and, if asked for, ``model.safetensors``) afterwards.
            ``False`` for a dry run or a ``fast_dev_run``-style smoke test.
        save_safetensors: Also write ``model.safetensors`` beside ``model.pt``. ``None`` -- the
            default and what ``sbb train`` passes -- follows
            :func:`~sbb_pytorch.storage.checkpoint.safetensors_available`, so a machine without the
            package loses the extra file instead of the run; ``True`` makes the missing package an
            error, and the re-export in this module is the patch point a test uses for both.
        **trainer_kwargs: Forwarded to :func:`build_trainer`.

    Returns:
        The trainer, with ``trainer.callback_metrics`` holding the final epoch's numbers.
    """
    if trainer is not None and trainer_kwargs:
        raise TrainingError(
            f"trainer_kwargs {sorted(trainer_kwargs)} are ignored when trainer= is given"
        )
    if start_weights is not None and resume is not None:
        raise TrainingError(
            "start_weights and resume are mutually exclusive: resume restores a run (weights, "
            "optimiser, epoch), start_weights starts a new run from weights"
        )
    module = module if module is not None else SBBLightningModule(config, model)
    if start_weights is not None:
        # Before the trainer, so the run starts from these weights rather than adopting a .ckpt's
        # optimiser state afterwards; the init_method/ImageNet choice is not consulted at all here.
        apply_start_weights(module, start_weights)
    data_kwargs = _dataloader_kwargs(config, data)
    trainer = (
        trainer
        if trainer is not None
        else build_trainer(config, module, device=device, output_dir=output_dir, **trainer_kwargs)
    )

    checkpoint = resume
    if checkpoint is None and config.continue_training and config.dir_of_start_model:
        checkpoint = config.dir_of_start_model
    trainer.fit(module, ckpt_path=str(checkpoint) if checkpoint else None, **data_kwargs)

    if save_final:
        directory = Path(output_dir if output_dir is not None else config.dir_output)
        final = final_checkpoint(config, module, trainer)
        final.save(directory / "model.pt")
        # The best weights, in the format a Hugging Face tool can open. model.pt keeps holding the
        # final weights so publishing the best never moves the file every existing reader expects;
        # the two files disagreeing is the honest version of "best" and "last".
        write_safetensors = (
            safetensors_available() if save_safetensors is None else save_safetensors
        )
        if write_safetensors:
            best_checkpoint(config, module, trainer).save_safetensors(
                directory / "model.safetensors"
            )
    return trainer
