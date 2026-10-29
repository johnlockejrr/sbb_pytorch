"""One Lightning module for every task, replacing ``train.py``'s ``SegmentationModel``.

``train.py`` had exactly one ``pl.LightningModule``, named ``SegmentationModel``, and used it for
all six tasks. What that cost, and what this class does instead:

* **Metrics per task.** It built ``Accuracy``, ``IoU`` and ``DiceScore`` for every task --
  including enhancement, whose target is an image, not a label map, so ``IoU.update`` scattered an
  image into a one-hot tensor and produced nonsense. Metrics now come from
  :func:`sbb_pytorch.metrics.get_metrics`, which knows which metrics exist for which task; a task
  with no pixel-wise metric (``reading_order``) trains with an empty metric set instead of raising.
* **Log names.** ``train_acc`` / ``train_iou`` / ``train_dice`` and their ``val_`` twins came from
  three separate objects, each recomputing the same counts. One confusion matrix now supplies
  ``val_accuracy``, ``val_miou``, ``val_dice_macro``, ``val_dice_micro`` and the per-class keys.
  The ``val_acc`` -> ``val_accuracy`` rename is listed in ``docs/migration.md``.
* **When metrics are computed.** ``self.log(...)`` was called with a fresh ``compute()`` on *every
  batch*, so the accumulated state was recomputed from scratch per batch and the logged value was a
  running average of every intermediate state. Metrics are updated per batch and computed once per
  epoch, which is what an accumulator is for.
* **Optimizer and scheduler.** ``configure_optimizers`` was the second of the three duplicated
  ``get_optimizer`` bodies (see :mod:`sbb_pytorch.training.factory`) and returned no scheduler --
  the ``ReduceLROnPlateau`` imported at the top of ``train.py`` was never constructed. It now
  delegates to the factory and returns the optimizer *and* the scheduler.
* **Hyperparameters.** ``save_hyperparameters(ignore=["model", "criterion"])`` worked because
  ``ConfigLoader`` handed out a plain dict; a pydantic ``SBBConfig`` is not YAML-representable, so
  it would break every checkpoint write. The config lives on ``self.config`` and inside every
  checkpoint instead (see :class:`sbb_pytorch.storage.checkpoint.Checkpoint`).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch import Tensor, nn

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.exceptions import TrainingError
from sbb_pytorch.losses import get_loss_function
from sbb_pytorch.metrics import Metric, get_metrics
from sbb_pytorch.models import create_model
from sbb_pytorch.training.factory import (
    InitMethod,
    build_optimizer,
    class_weights_from_directory,
    init_weights,
    scheduler_from_config,
)

__all__ = ["SBBLightningModule", "default_criterion", "unpack_batch"]

#: Metric keys checked first when choosing the "best by" checkpoint, so the choice is stable
#: regardless of the order :class:`~sbb_pytorch.metrics.Metric` returns its keys in.
_PREFERRED_BEST_KEYS = ("val_accuracy", "val_miou", "val_dice_macro", "val_dice_micro")


def default_criterion(config: SBBConfig, *, weights: Tensor | None = None) -> nn.Module:
    """The loss ``config`` asks for, assembled the way ``train.py`` assembled it.

    ``config.weighted_loss`` pulls class weights out of ``config.dir_train`` (the legacy
    ``calculate_class_weights`` call, now :func:`~sbb_pytorch.training.factory.class_weights_from_directory`)
    and ``config.is_loss_soft_dice`` selects ``soft_dice`` over the task default -- two config
    fields the legacy code had but only ever honoured in the ``segmentation`` branch.

    Raises:
        NotImplementedError: ``task=reading_order``, which has no trainable loss in this port.
        ValueError: an unknown task, or an empty ``dir_train`` when ``weighted_loss`` is set.
    """
    if config.weighted_loss and weights is None:
        weights = class_weights_from_directory(config.dir_train, config.n_classes)
    return get_loss_function(
        config.task,
        weights,
        loss="soft_dice" if config.is_loss_soft_dice else None,
        n_classes=config.n_classes,
    )


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor | None]:
    """``(image, mask)`` from the tuple :class:`~sbb_pytorch.data.dataset.SegmentationDataset` yields.

    Also accepts the dict form an ``albudataloading``-style pipeline produces and a bare image
    tensor (``mask`` is then ``None``), because ``predict`` batches have no labels.

    Raises:
        TrainingError: a shape none of those describes.
    """
    if isinstance(batch, Tensor):
        return batch, None
    if isinstance(batch, dict):
        try:
            return batch["image"], batch["mask"]
        except KeyError as exc:
            raise TrainingError(
                f"batch dict has no {exc.args[0]!r} key; keys are {sorted(batch)}"
            ) from exc
    if isinstance(batch, list | tuple) and len(batch) == 2:
        return batch[0], batch[1]
    raise TrainingError(f"unsupported batch type {type(batch).__name__}")


def _metrics_for(config: SBBConfig) -> dict[str, Metric]:
    """Metrics for ``config.task``; an empty set where the task has no pixel-wise metric."""
    try:
        return dict(get_metrics(config.task, config.n_classes))
    except NotImplementedError:
        # reading_order: the order is validated through sbb_pytorch.gt, not per pixel.
        return {}


class SBBLightningModule(pl.LightningModule):
    """Model + loss + metrics for ``pl.Trainer.fit``, for any task in :class:`SBBConfig`.

    Args:
        config: The resolved configuration. Everything below defaults out of it.
        model: A ready model. When omitted, :func:`~sbb_pytorch.models.registry.create_model`
            builds the one ``config`` describes.
        criterion: Loss override; defaults to :func:`default_criterion`.
        weights: Class weights for a weighted loss, so ``dir_train`` need not be on disk (a test,
            or a resume where recomputing them from the training folder would be wrong).
        train_metrics / val_metrics: Metric sets per phase; default to freshly built sets from
            :func:`sbb_pytorch.metrics.get_metrics`. Pass ``{}`` to switch a phase off -- train and
            validation need separate accumulators, so one dict is never shared between them.
        init_method: ``"auto"`` (default) reproduces the intended legacy policy -- kaiming for a
            model built from scratch, no re-initialisation when pretrained weights were loaded.
            ``train.py`` always re-initialised, which erased the ImageNet weights it had just
            downloaded; pass a method to force one, or ``None`` to keep the model's own init.
        model_name: Registry key overriding the config-derived architecture (``sbb train --model``).
        pretrained: Override ``config.pretraining``; ``False`` on machines without weight access.
        monitor: Metric the plateau scheduler and the checkpoint callbacks reduce on.
        schedule: ``False`` returns the optimizer alone, reproducing what ``train.py`` returned.

    Example:
        >>> import torch
        >>> from sbb_pytorch.config.models import SBBConfig
        >>> config = SBBConfig.model_validate(
        ...     {"task": "segmentation", "n_classes": 2, "input_height": 32, "input_width": 32}
        ... )
        >>> module = SBBLightningModule(config, pretrained=False)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: SBBConfig,
        model: nn.Module | None = None,
        *,
        criterion: nn.Module | None = None,
        weights: Tensor | None = None,
        train_metrics: dict[str, Metric] | None = None,
        val_metrics: dict[str, Metric] | None = None,
        init_method: InitMethod | str | None = "auto",
        model_name: str | None = None,
        pretrained: bool | None = None,
        monitor: str = "val_loss",
        schedule: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.monitor = monitor
        self.schedule = schedule

        load_pretrained = config.pretraining if pretrained is None else bool(pretrained)
        self.model = (
            create_model(config, model=model_name, pretrained=pretrained)
            if model is None
            else model
        )
        method = init_method
        if isinstance(method, str) and method == "auto":
            # "auto" == kaiming for a from-scratch model, nothing for a pretrained one. A model
            # handed in is left alone: we cannot tell whether its weights are someone's training.
            method = None if (load_pretrained or model is not None) else InitMethod.KAIMING
        if method is not None:
            init_weights(self.model, method)

        self.criterion = (
            criterion if criterion is not None else default_criterion(config, weights=weights)
        )
        self.train_metrics = _metrics_for(config) if train_metrics is None else dict(train_metrics)
        self.val_metrics = _metrics_for(config) if val_metrics is None else dict(val_metrics)

    # ------------------------------------------------------------------ forward

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @staticmethod
    def unpack_batch(batch: Any) -> tuple[Tensor, Tensor | None]:
        """See :func:`unpack_batch` -- exposed on the class so subclasses can specialise it."""
        return unpack_batch(batch)

    # -------------------------------------------------------------------- steps

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """One optimiser step: loss on the batch, metrics accumulated for the epoch summary."""
        images, targets = self.unpack_batch(batch)
        if targets is None:
            raise TrainingError("training_step needs a target; got an image-only batch")
        logits = self(images)
        loss = self.criterion(logits, targets)

        # `train_loss` is the epoch value (the key legacy code had), `train_loss_step` keeps the
        # progress bar live during the epoch.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False)
        for metric in self.train_metrics.values():
            metric.update(logits, targets)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Same forward pass in eval mode; ``val_loss`` plus the accumulated metrics."""
        images, targets = self.unpack_batch(batch)
        if targets is None:
            raise TrainingError("validation_step needs a target; got an image-only batch")
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss, prog_bar=True)
        for metric in self.val_metrics.values():
            metric.update(logits, targets)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int | None = None) -> Tensor:
        """Logits only, which is what ``sbb infer`` consumes."""
        return self(self.unpack_batch(batch)[0])

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self.val_metrics)

    def _log_metrics(self, phase: str, metrics: dict[str, Metric]) -> None:
        """Compute every accumulated metric once, log the flat dict, then clear the accumulators.

        ``compute_and_reset`` is the metrics API's own pair (:class:`~sbb_pytorch.metrics.Metric`),
        so a phase's numbers come from one pass over its counts instead of one pass per batch.
        """
        if not metrics:
            return
        values: dict[str, float] = {}
        for metric in metrics.values():
            values.update(metric.compute_and_reset())
        self.log_dict({f"{phase}_{key}": value for key, value in values.items()})

    # ------------------------------------------------------------- optimization

    def configure_optimizers(self) -> dict[str, Any]:
        """The factory's optimizer, plus the ``ReduceLROnPlateau`` ``train.py`` never built.

        ``schedule=False`` returns the optimizer alone, reproducing the legacy return value for a
        run that wants it.
        """
        optimizer = build_optimizer(self.model, self.config)
        if not self.schedule:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_from_config(self.config, optimizer),
                "monitor": self.monitor,
                "interval": "epoch",
            },
        }

    # ------------------------------------------------------------------ naming

    @property
    def logged_val_keys(self) -> tuple[str, str | None]:
        """``(loss_key, best_key)`` -- the two metrics ``train.py`` put in its checkpoint names.

        ``val_loss`` always exists; the second key is the best-not-by-loss metric the module logs,
        or ``None`` when the task has no metrics -- which is what lets ``callbacks.py`` skip the
        ``monitor="val_acc"`` checkpoint instead of crashing the way the legacy one did on a task
        that never logs it.
        """
        if not self.val_metrics:
            return self.monitor, None
        keys = [
            f"val_{key}"
            for metric in self.val_metrics.values()
            for key in metric.compute()
            # A slash would nest the checkpoint file (`iou/textline` -> `best-07-.../`), and a
            # second loss key would duplicate the loss checkpoint.
            if "/" not in key and "loss" not in key
        ]
        preferred = [key for key in _PREFERRED_BEST_KEYS if key in keys]
        return "val_loss", (preferred[0] if preferred else (keys[0] if keys else None))
