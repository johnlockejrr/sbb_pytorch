"""Checkpointing and early stopping: the callbacks that work, replacing the two that never ran.

Three legacy implementations are folded into this one:

``train.py``'s ``SaveWeightsAfterSteps``
    Declared ``on_train_batch_end(self, model, optimizer, epoch)`` while pytorch-lightning calls
    ``on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx)`` -- so its first
    argument was the trainer and its "epoch" was the loss dict. :class:`StepCheckpoint` has the
    signature pl actually calls (the same one ``ModelCheckpoint`` uses) and writes a real
    :class:`~sbb_pytorch.storage.checkpoint.Checkpoint`.

``training.py``'s ``ModelCheckpoint``
    Its ``save_model`` read ``self.model`` and ``self.optimizer``, attributes its own ``__init__``
    never assigned, so the first save raised ``AttributeError`` and no file was ever written.

``train.py``'s callback list
    Two ``ModelCheckpoint``\\ s (best by ``val_loss``, best by ``val_acc``) and two
    ``EarlyStopping``\\ s, registered unconditionally -- so any task that never logs ``val_acc``
    (enhancement, reading order) raised at its first validation epoch end. :func:`build_callbacks`
    derives the monitored keys from what the module really logs via
    :attr:`~sbb_pytorch.training.module.SBBLightningModule.logged_val_keys`.

Pruning is handled twice, correctly, in both places that need it: pytorch-lightning keeps the
``top_k`` best by metric for its own files, and
:func:`~sbb_pytorch.storage.checkpoint.prune_checkpoints` orders the step files numerically (the
legacy ``sorted(glob("..._epoch_*.pth"))`` put ``epoch_10`` before ``epoch_2`` and deleted the
newest files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch import Tensor

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.storage.checkpoint import Checkpoint, prune_checkpoints
from sbb_pytorch.training.module import SBBLightningModule

__all__ = [
    "DEFAULT_TOP_K",
    "StepCheckpoint",
    "build_callbacks",
    "checkpoint_for",
    "early_stopping_for",
    "metric_mode",
    "metric_snapshot",
    "monitor_metrics",
]

#: ``train.py`` kept the three newest checkpoints per monitored metric.
DEFAULT_TOP_K = 3


def metric_mode(metric: str) -> str:
    """``min`` for a loss, ``max`` for anything else -- the rule ``train.py`` spelled out twice."""
    return "min" if "loss" in metric else "max"


def monitor_metrics(module: SBBLightningModule) -> tuple[tuple[str, str], ...]:
    """The ``(metric, mode)`` pairs worth tracking on ``module``: the legacy pair, minus dead keys.

    Returns ``(("val_loss", "min"), ("val_accuracy", "max"))`` for a task with metrics, and the
    loss alone for one without -- which is what lets a run skip the ``monitor="val_acc"`` callback
    instead of crashing the way the unconditional legacy pair did.
    """
    loss_key, best_key = module.logged_val_keys
    pairs = [(loss_key, metric_mode(loss_key))]
    if best_key is not None:
        pairs.append((best_key, metric_mode(best_key)))
    return tuple(pairs)


def checkpoint_for(
    config: SBBConfig,
    monitor: str,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> ModelCheckpoint:
    """One ``ModelCheckpoint`` writing ``{stem}-{epoch:02d}-{metric:.4f}.ckpt``.

    The filename pattern is the readable half of what ``train.py`` had
    (``model-07-0.3141.ckpt``); pytorch-lightning then keeps the ``top_k`` best by metric, which
    also retires the lexicographic prune described in :mod:`sbb_pytorch.storage.checkpoint`.

    The ``<stem>_config.json`` sidecar is *not* written here: pytorch-lightning names and rotates
    these files, and the config travels inside ``model.pt`` (see
    :func:`sbb_pytorch.training.trainer.fit`) and inside every ``Checkpoint`` this module writes.
    """
    directory = Path(output_dir if output_dir is not None else config.dir_output)
    mode = mode or metric_mode(monitor)
    stem = stem or ("model" if mode == "min" else "best")
    return ModelCheckpoint(
        dirpath=str(directory),
        filename=f"{stem}-{{epoch:02d}}-{{{monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=int(top_k),
    )


def early_stopping_for(
    config: SBBConfig,
    monitor: str,
    *,
    mode: str | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
) -> EarlyStopping:
    """``EarlyStopping`` on ``monitor`` with the config's patience (``train.py``'s ``min_delta=0``)."""
    return EarlyStopping(
        monitor=monitor,
        mode=mode or metric_mode(monitor),
        patience=int(config.patience if patience is None else patience),
        min_delta=float(min_delta),
        verbose=True,
    )


def metric_snapshot(trainer: pl.Trainer) -> dict[str, float]:
    """``trainer.callback_metrics`` as ``{name: float}`` -- what ``Checkpoint.metrics`` wants.

    ``train.py`` indexed ``trainer.callback_metrics['val_acc']`` directly when saving the final
    model, so a task without that key raised ``KeyError`` after a full training run. Everything
    non-scalar is skipped instead.
    """
    snapshot: dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                continue
            snapshot[key] = float(value.detach().cpu().item())
        elif isinstance(value, int | float):
            snapshot[key] = float(value)
    return snapshot


def optimizer_state(trainer: pl.Trainer) -> dict[str, Any] | None:
    """The first optimizer's state, or ``None`` -- ``train.py`` assumed exactly one optimizer."""
    optimizers = trainer.optimizers
    if isinstance(optimizers, list | tuple):
        if not optimizers:
            return None
        optimizers = optimizers[0]
    state = getattr(optimizers, "state_dict", None)
    return None if state is None else state()


class StepCheckpoint(Callback):
    """Save a :class:`~sbb_pytorch.storage.checkpoint.Checkpoint` every ``every_n_steps`` batches.

    Args:
        config: Config stored inside each saved checkpoint (the schema's ``config`` field).
        every_n_steps: Batches between saves; ``0``/``None`` behaviour is pytorch-lightning's
            ``every_n_train_steps`` equivalent -- here a value below 1 disables the callback.
        keep_last_n: How many step files to keep; older ones are deleted (numerically ordered).
        output_dir: Run directory, defaulting to ``config.dir_output``.
        dirname: Subdirectory for step files, so ``latest_checkpoint(dir_output)`` still returns an
            epoch checkpoint rather than a mid-epoch one. The legacy code created one directory per
            step (``model_step_1200/model.pt``), which no reader could glob.
    """

    def __init__(
        self,
        config: SBBConfig,
        *,
        every_n_steps: int = 1000,
        keep_last_n: int = 5,
        output_dir: str | Path | None = None,
        dirname: str = "steps",
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {every_n_steps}")
        self.config = config
        self.every_n_steps = int(every_n_steps)
        self.keep_last_n = int(keep_last_n)
        self.directory = Path(output_dir if output_dir is not None else config.dir_output) / dirname

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Write one checkpoint on every ``every_n_steps``-th optimiser step."""
        step = int(trainer.global_step)
        if step < 1 or step % self.every_n_steps:
            return
        model = getattr(pl_module, "model", pl_module)
        Checkpoint(
            model_state=dict(model.state_dict()),
            epoch=int(trainer.current_epoch),
            step=step,
            task=self.config.task.value,
            config=self.config.model_dump(mode="json"),
            optimizer_state=optimizer_state(trainer),
            metrics=metric_snapshot(trainer),
        ).save(self.directory / f"model_step_{step}.pt", write_sidecar=False)
        prune_checkpoints(self.directory, keep_last_n=self.keep_last_n)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prune once more at the end, so a short run leaves no more than ``keep_last_n`` files."""
        prune_checkpoints(self.directory, keep_last_n=self.keep_last_n)


def build_callbacks(
    config: SBBConfig,
    module: SBBLightningModule | None = None,
    *,
    output_dir: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    every_n_steps: int | None = None,
    early_stopping: bool = True,
) -> list[Callback]:
    """The callback list for a run: the checkpoint pair, the early-stopping pair, optional steps.

    Args:
        config: The resolved configuration (``dir_output``, ``patience``).
        module: The module being trained; its :attr:`~sbb_pytorch.training.module.SBBLightningModule.logged_val_keys`
            decides which metrics exist. Without one, only ``val_loss`` is tracked -- the safest
            subset, since a checkpoint cannot monitor a metric nobody logs.
        output_dir: Where to write; defaults to ``config.dir_output``.
        top_k: Checkpoints kept per monitored metric.
        every_n_steps: Add a :class:`StepCheckpoint` saving every N batches, mirroring the
            ``--save-interval`` the legacy CLI passed to ``SaveWeightsAfterSteps``.
        early_stopping: ``False`` to train without the early-stopping pair (the legacy
            ``--no-early-stop`` equivalent).
    """
    callbacks: list[Callback] = []
    if module is None:
        metrics: tuple[tuple[str, str], ...] = (("val_loss", "min"),)
    else:
        metrics = monitor_metrics(module)

    for metric, mode in metrics:
        callbacks.append(
            checkpoint_for(config, metric, mode=mode, output_dir=output_dir, top_k=top_k)
        )
        if early_stopping:
            callbacks.append(early_stopping_for(config, metric, mode=mode))
    if every_n_steps is not None:
        callbacks.append(StepCheckpoint(config, every_n_steps=every_n_steps, output_dir=output_dir))
    return callbacks
