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

:command:`sbb train --callbacks-file` -- the flag ``build_trainer`` names at
``training/trainer.py:111`` ("Callbacks to use as-is instead of the built set") -- is
:func:`load_callbacks_file`. Neither legacy repository had anything like it: ``train.py`` and
``training.py`` both hard-coded their callback lists in Python (the pair of ``ModelCheckpoint``\\s
and pair of ``EarlyStopping``\\s described above), so this package defines the format -- and
:func:`build_callbacks` *appends* what the file names to the set above instead of standing in for
it, which is what the flag has always been for: adding the one callback a run needs, not
re-deriving the checkpointing the config already describes.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.utilities.model_summary import (
    get_formatted_model_size,
    get_human_readable_count,
)
from torch import Tensor

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.storage.checkpoint import Checkpoint, prune_checkpoints
from sbb_pytorch.training.module import SBBLightningModule

__all__ = [
    "DEFAULT_TOP_K",
    "SBBModelSummary",
    "SBBProgressBar",
    "StepCheckpoint",
    "build_callbacks",
    "checkpoint_for",
    "early_stopping_fields",
    "early_stopping_for",
    "load_callbacks_file",
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
    verbose: bool = False,
) -> ModelCheckpoint:
    """One ``ModelCheckpoint`` writing ``{stem}-{epoch:02d}-{metric:.4f}.ckpt``.

    The filename pattern is the readable half of what ``train.py`` had
    (``model-07-0.3141.ckpt``); pytorch-lightning then keeps the ``top_k`` best by metric, which
    also retires the lexicographic prune described in :mod:`sbb_pytorch.storage.checkpoint`.

    The ``<stem>_config.json`` sidecar is *not* written here: pytorch-lightning names and rotates
    these files, and the config travels inside ``model.pt`` (see
    :func:`sbb_pytorch.training.trainer.fit`) and inside every ``Checkpoint`` this module writes.

    ``verbose`` is ``False`` (the ``ModelCheckpoint`` default) -- the one line it prints on an
    improvement is the pair's ``Metric val_loss improved. New best score: 0.449`` output, and
    :command:`sbb train --verbose-metrics` switches both this and :func:`early_stopping_for` on
    together.
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
        verbose=bool(verbose),
    )


def early_stopping_for(
    config: SBBConfig,
    monitor: str,
    *,
    mode: str | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    verbose: bool = False,
) -> EarlyStopping:
    """``EarlyStopping`` on ``monitor`` with the config's patience (``train.py``'s ``min_delta=0``).

    ``verbose`` used to be hard-coded ``True``: every improvement printed a line -- ``Metric
    val_loss improved. New best score: 0.449`` the first time, ``Metric val_loss improved by 0.047
    >= min_delta = 0.0. New best score: 0.402`` every time after, so two lines per epoch for the
    pair this package registers. It is ``False`` now, because the same numbers sit on the progress
    bar as ``early_stopping/val_loss: 3/10`` (see :class:`SBBProgressBar`) and
    :command:`sbb train --verbose-metrics` brings the prints back for anyone who wants them in the
    log instead of the bar.
    """
    return EarlyStopping(
        monitor=monitor,
        mode=mode or metric_mode(monitor),
        patience=int(config.patience if patience is None else patience),
        min_delta=float(min_delta),
        verbose=bool(verbose),
    )


def early_stopping_fields(trainer: Any) -> dict[str, str]:
    """``{"early_stopping": "3/10"}`` -- one field for every ``EarlyStopping`` on *trainer*.

    The bar's version of what the ``verbose=True`` prints said: epochs a monitor has gone without
    improving, over the patience that ends the run. The two callbacks this package registers (one per
    monitored metric) used to each add a field -- ``early_stopping/val_loss: 3/10`` and
    ``early_stopping/val_accuracy: 3/10`` -- which on the bar read as two more copies of the metric
    *names* with the values left out, right next to the same numbers as ``val_loss: 0.449``. One
    field now, carrying the callback closest to stopping (largest ``wait_count``, then the shortest
    ``patience``); ``verbose=True`` still prints a line per callback for whoever wants both.
    """
    counts = [
        (callback.wait_count, callback.patience)
        for callback in getattr(trainer, "callbacks", ())
        if isinstance(callback, EarlyStopping) and callback.monitor is not None
    ]
    if not counts:
        return {}
    wait, patience = max(counts, key=lambda pair: (pair[0], -pair[1]))
    return {"early_stopping": f"{wait}/{patience}"}


class SBBProgressBar(RichProgressBar):
    """``RichProgressBar`` that leaves every finished epoch's line on screen, counters included.

    Two changes to the parent, both about what a bar is for:

    ``leave=True`` (the parent defaults to ``False``)
        Rich paints one *live* panel and repaints it in place, so with the default an epoch's bar is
        erased the moment the next epoch starts and a finished run leaves one line behind -- the last
        epoch, not the run. The parent stops and rebuilds the live display at each epoch start when
        ``leave`` is on, which prints the finished frame first: ``Epoch 0/199 ...`` with the numbers
        that epoch ended on, then a fresh bar. A ``sbb train`` of 200 epochs ends with 200 lines.
    :func:`early_stopping_fields` on the metrics line
        The parent lists ``trainer.callback_metrics`` and says nothing about the two
        ``EarlyStopping``\\ s registered beside it, so a run's remaining patience was only ever
        visible as the ``Metric ...`` lines this package silenced. ``early_stopping: 3/10`` is now
        part of the line, beside the losses.
    """

    def __init__(self, *args: Any, leave: bool = True, **kwargs: Any) -> None:
        """``leave`` defaults to ``True``; the parent's own arguments (``refresh_rate``, ``theme``,
        ``console_kwargs``) pass through unchanged."""
        super().__init__(*args, leave=leave, **kwargs)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict[str, Any]:
        """The parent's metrics plus the early-stopping counters, as one dict."""
        items = dict(super().get_metrics(trainer, pl_module))
        items.update(early_stopping_fields(trainer))
        return items


class SBBModelSummary(RichModelSummary):
    """``RichModelSummary`` minus the ``FLOPs`` column and the ``Total FLOPs: 0`` line under it.

    Lightning can only count FLOPs when the module exposes ``example_input_array``, and
    :class:`~sbb_pytorch.training.module.SBBLightningModule` has never had one -- so every run
    printed ``Total FLOPs: 0`` under a model with real convolutions, a number that was never a
    measurement. The table and the parameter grid are rendered here without it: everything
    ``train.py`` showed stays, the one field that never meant anything goes.
    :func:`~sbb_pytorch.training.trainer.build_trainer` still decides whether a summary is shown.
    """

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int | None = None,
        **summarize_kwargs: Any,
    ) -> None:
        """The rich table and parameter grid of ``RichModelSummary``, without the FLOPs entries.

        ``total_flops`` is accepted and ignored: the point of this subclass is that the value is
        unmeasured, and the signature has to stay the one ``ModelCheckpoint``-era
        ``ModelSummary.on_fit_start`` calls it with.
        """
        from rich import get_console
        from rich.table import Table

        def count(value: float) -> str:
            return f"{get_human_readable_count(int(value)):<10}"

        columns = [(name, values) for name, values in summary_data if name != "FLOPs"]
        table = Table(header_style=summarize_kwargs.get("header_style", "bold magenta"))
        for name, _ in columns:
            table.add_column(
                name,
                justify="left" if name in (" ", "Name", "Type") else "right",
                no_wrap=name == "Name",
            )
        if columns:
            for row in zip(*(values for _, values in columns), strict=False):
                table.add_row(*(str(cell) for cell in row))
        get_console().print(table)

        size = f"{get_formatted_model_size(model_size):<10}"
        rows = [
            ("Trainable params", count(trainable_parameters)),
            ("Non-trainable params", count(total_parameters - trainable_parameters)),
            ("Total params", count(total_parameters)),
            ("Total estimated model params size (MB)", size),
            ("Modules in train mode", str(total_training_modes.get("train", 0))),
            ("Modules in eval mode", str(total_training_modes.get("eval", 0))),
        ]
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()
        for label, value in rows:
            grid.add_row(f"[bold]{label}[/]: {value}")
        get_console().print(grid)


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


def _build_entry(entry: object, *, path: Path, index: int) -> Callback:
    """One entry of a callbacks file, imported and instantiated.

    An entry is either a dotted import path (``"pytorch_lightning.callbacks.Callback"``) or an
    object naming the class and, optionally, the kwargs its ``__init__`` takes::

        {"class": "pytorch_lightning.callbacks.ModelCheckpoint", "init": {"monitor": "val_loss"}}

    Anything else is refused *by position* -- a typo inside an object used to survive parsing and
    fail inside ``pl.Trainer``, which is the same late-erroring the module docstring lists for the
    legacy callbacks. Every message names both the file and the entry, so a seven-entry file with
    one bad line says which one.

    Raises:
        ConfigError: the entry is neither spelling, ``init`` is not an object, the module or class
            cannot be imported, the attribute is not a ``pl.Callback`` subclass, or ``init`` does
            not fit the class's ``__init__``.
    """
    if isinstance(entry, str):
        dotted, init = entry, {}
    elif isinstance(entry, dict) and isinstance(entry.get("class"), str):
        dotted = str(entry["class"])
        unknown = sorted(set(entry) - {"class", "init"})
        if unknown:
            raise ConfigError(
                f"{path}: entry {index} has unknown key(s) {unknown}; expected 'class' and 'init'"
            )
        raw_init = entry.get("init", {})
        if not isinstance(raw_init, dict):
            raise ConfigError(
                f"{path}: entry {index} 'init' must be a JSON object, got {type(raw_init).__name__}"
            )
        init = dict(raw_init)
    else:
        raise ConfigError(
            f"{path}: entry {index} must be a dotted class path or an object naming "
            f"'class' and 'init', got {entry!r}"
        )

    module_name, _, attribute = dotted.rpartition(".")
    if not module_name:
        raise ConfigError(
            f"{path}: entry {index} {dotted!r} is not a dotted import path (name the module too)"
        )
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except ImportError as exc:
        raise ConfigError(f"{path}: entry {index} cannot import {module_name!r}: {exc}") from exc
    except AttributeError as exc:
        raise ConfigError(
            f"{path}: entry {index}: module {module_name!r} has no attribute {attribute!r}"
        ) from exc
    if not isinstance(candidate, type):
        raise ConfigError(
            f"{path}: entry {index} {dotted!r} is a {type(candidate).__name__}, not a class"
        )
    if not issubclass(candidate, Callback):
        raise ConfigError(
            f"{path}: entry {index} {dotted!r} is a {candidate.__name__}, not a "
            f"pytorch_lightning Callback"
        )
    try:
        return candidate(**init)
    except TypeError as exc:  # the kwargs the class does not take -- a config error, not a crash
        raise ConfigError(f"{path}: entry {index} {dotted!r} cannot be built: {exc}") from exc


def load_callbacks_file(path: str | Path) -> list[Callback]:
    """Build the callbacks ``sbb train --callbacks-file`` points at.

    The file is a JSON **list**; each entry is a dotted import path or a ``{"class": ..., "init":
    {...}}`` object (see :func:`_build_entry`), and an empty list is legal -- it adds nothing, which
    is how a config-controlled run turns the feature off without deleting the file. Everything that
    can be wrong with the file itself -- unreadable, malformed, a non-list, an unusable entry --
    comes back as :class:`~sbb_pytorch.exceptions.ConfigError`, the type the CLI prints as one
    ``Error:`` line instead of a traceback, worded like
    :func:`sbb_pytorch.config.loaders.load_config` so both config-shaped files say it the same way.

    Raises:
        ConfigError: the file cannot be read, is not valid JSON, does not hold a list, or names a
            class that cannot be built (see :func:`_build_entry`).
    """
    file = Path(path)
    try:
        raw = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"callbacks file could not be read ({file}): {exc}") from exc
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"callbacks file is not valid JSON ({file}): {exc}") from exc
    if not isinstance(entries, list):
        raise ConfigError(
            f"callbacks file must contain a JSON list, got {type(entries).__name__} ({file})"
        )
    return [_build_entry(entry, path=file, index=index) for index, entry in enumerate(entries)]


def build_callbacks(
    config: SBBConfig,
    module: SBBLightningModule | None = None,
    *,
    output_dir: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    every_n_steps: int | None = None,
    early_stopping: bool = True,
    callbacks_file: str | Path | None = None,
    verbose: bool = False,
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
        callbacks_file: ``--callbacks-file``: a JSON list of extra callback classes
            (:func:`load_callbacks_file`), *appended* to the set this function builds. It never
            replaces it -- ``build_trainer``'s ``callbacks=`` argument is what replaces the built
            set, and :command:`sbb train` passes the finished list through that argument so the
            config's own callbacks survive the file.
        verbose: ``--verbose-metrics``, forwarded to both callbacks built here -- the ``Metric
            val_loss improved. New best score: 0.449`` prints ``train.py`` ran with. ``False`` by
            default, because the same numbers reach the bar as ``early_stopping/val_loss: 3/10``
            through :class:`SBBProgressBar`.
    """
    callbacks: list[Callback] = []
    if module is None:
        metrics: tuple[tuple[str, str], ...] = (("val_loss", "min"),)
    else:
        metrics = monitor_metrics(module)

    for metric, mode in metrics:
        callbacks.append(
            checkpoint_for(
                config, metric, mode=mode, output_dir=output_dir, top_k=top_k, verbose=verbose
            )
        )
        if early_stopping:
            callbacks.append(early_stopping_for(config, metric, mode=mode, verbose=verbose))
    if every_n_steps is not None:
        callbacks.append(StepCheckpoint(config, every_n_steps=every_n_steps, output_dir=output_dir))
    if callbacks_file is not None:
        callbacks.extend(load_callbacks_file(callbacks_file))
    return callbacks
