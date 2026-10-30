"""Training: the pieces ``train.py`` + ``training.py`` + ``initialization.py`` did separately.

=========================== =====================================================================
Module                      Replaces
=========================== =====================================================================
:mod:`~sbb_pytorch.training.factory`
                            ``train.py``'s ``get_optimizer`` / ``get_loss_function`` /
                            ``calculate_class_weights``, the unused ``ReduceLROnPlateau`` import,
                            and all of ``initialization.py`` (weight init, gradient clipping,
                            ``WeightRegularizer``)
:mod:`~sbb_pytorch.training.module`
                            ``train.py``'s ``SegmentationModel`` (one LightningModule for all six
                            tasks) and ``training.py``'s ``Metric``/``Accuracy``/``IoU``/``DiceScore``
                            wiring
:mod:`~sbb_pytorch.training.callbacks`
                            ``train.py``'s ``SaveWeightsAfterSteps`` (wrong hook signature, never
                            ran), ``training.py``'s ``ModelCheckpoint`` (unread ``self.model``,
                            crashed on first save), ``train.py``'s unconditional checkpoint and
                            early-stopping pairs, and ``load_callbacks_file`` behind
                            ``sbb train --callbacks-file`` (nothing legacy could add a callback
                            without editing the file)
:mod:`~sbb_pytorch.training.trainer`
                            ``train.py``'s ``pl.Trainer(...)`` call, its ``torch.save`` of the final
                            model, and ``training.py``'s hand-written ``Trainer`` class with its
                            ``train_step``/``validate_step``/``train_epoch`` loops
=========================== =====================================================================

Typical use, which is what ``sbb train`` runs:

    >>> from sbb_pytorch.config.models import SBBConfig
    >>> from sbb_pytorch.training import SBBLightningModule, build_trainer  # doctest: +SKIP
    >>> config = SBBConfig.model_validate(  # doctest: +SKIP
    ...     {"task": "segmentation", "n_classes": 2, "input_height": 32, "input_width": 32}
    ... )
    >>> module = SBBLightningModule(config, pretrained=False)  # doctest: +SKIP
    >>> trainer = build_trainer(config, module, logger=False, enable_progress_bar=False)  # doctest: +SKIP
"""

from __future__ import annotations

from sbb_pytorch.training.callbacks import (
    DEFAULT_TOP_K,
    SBBModelSummary,
    SBBProgressBar,
    StepCheckpoint,
    build_callbacks,
    checkpoint_for,
    early_stopping_fields,
    early_stopping_for,
    load_callbacks_file,
    metric_mode,
    metric_snapshot,
    monitor_metrics,
)
from sbb_pytorch.training.factory import (
    InitMethod,
    RegularizationKind,
    build_optimizer,
    build_scheduler,
    class_weights,
    class_weights_from_directory,
    clip_gradients,
    init_weights,
    kaiming_init,
    orthogonal_init,
    regularization_loss,
    scheduler_from_config,
    xavier_init,
)
from sbb_pytorch.training.module import SBBLightningModule, default_criterion, unpack_batch
from sbb_pytorch.training.trainer import (
    apply_start_weights,
    best_checkpoint,
    best_checkpoint_path,
    build_trainer,
    final_checkpoint,
    fit,
    resolve_accelerator,
    resolve_logger,
    tensorboard_available,
)

__all__ = [
    "DEFAULT_TOP_K",
    "InitMethod",
    "RegularizationKind",
    "SBBLightningModule",
    "SBBModelSummary",
    "SBBProgressBar",
    "StepCheckpoint",
    "apply_start_weights",
    "best_checkpoint",
    "best_checkpoint_path",
    "build_callbacks",
    "build_optimizer",
    "build_scheduler",
    "build_trainer",
    "checkpoint_for",
    "class_weights",
    "class_weights_from_directory",
    "clip_gradients",
    "default_criterion",
    "early_stopping_fields",
    "early_stopping_for",
    "final_checkpoint",
    "fit",
    "init_weights",
    "kaiming_init",
    "load_callbacks_file",
    "metric_mode",
    "metric_snapshot",
    "monitor_metrics",
    "orthogonal_init",
    "regularization_loss",
    "resolve_accelerator",
    "resolve_logger",
    "scheduler_from_config",
    "tensorboard_available",
    "unpack_batch",
    "xavier_init",
]
