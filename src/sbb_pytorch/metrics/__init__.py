"""Metrics for the package: one interface, one implementation per formula, one set of numbers.

The pre-refactor repo had ``metrics.py`` (384 lines) plus ``losses.py`` (112 lines), and between
them computed mIoU four different ways, Dice three ways, and shipped an enhancement loss that
raised ``NameError`` the moment it was constructed. This package keeps one definition per
formula, in one place, and lists every place its numbers differ from the old ones below.

Modules
-------
* :mod:`sbb_pytorch.metrics.functional` -- the analytic torch math (coefficients + loss
  functions), all on the canonical ``(B, C, N)`` shape.
* :mod:`sbb_pytorch.metrics.segmentation` -- :class:`~sbb_pytorch.metrics.segmentation.Metric`
  and :class:`~sbb_pytorch.metrics.segmentation.ConfusionMatrixMetric`, plus the one-shot
  ``mean_iou`` / ``dice_score`` / ``accuracy`` helpers.
* :mod:`sbb_pytorch.metrics.enhance` -- PSNR and SSIM for the enhancement task.

Use :func:`get_metrics` to build the right set for a task, as ``metrics.get_metrics`` did.

SEMANTIC CHANGES -- every number this changes, for sign-off
-----------------------------------------------------------
Behaviour preservation was the rule for this refactor, so anything that moves a number is listed
here rather than buried in a docstring. Items marked **(breaking)** change reported values; the
rest only change which value is reported, in exchange for being correct.

1. **(breaking) Tversky ``alpha``/``beta`` were swapped.** ``metrics.class_tversky`` computed
   ``TP / (TP + alpha*FN + beta*FP)``; the definition is ``TP / (TP + alpha*FP + beta*FN)``. Any
   run that used the ``alpha=0.2, beta=0.8`` pairing was penalising false positives lightly and
   false negatives heavily -- the opposite of what it asked for.
2. **(breaking) Focal-Tversky used the wrong formula.** Old: ``sum((1 - tversky) ** 1.3)``.
   New: ``mean((1 + DT) ** 3)`` with ``DT = (FP + TP) / (TP + alpha*FP + beta*FN)``, i.e. the
   published loss. Loss values shift by a large factor, so a focal-Tversky run must be re-baseline.
3. **(breaking) Focal-loss ``alpha`` is now ``0.25`` everywhere.** ``metrics.focal_loss`` defaulted
   to ``alpha=4.0``, ``losses.FocalLoss`` to ``0.25``; ``4.0`` is outside ``[0, 1]`` and scales
   the loss by 16x relative to Lin et al. The single constant is
   ``sbb_pytorch.metrics.functional.DEFAULT_FOCAL_ALPHA``.
4. **(breaking) ``metrics.focal_loss`` no longer dilutes the loss by ``1 / n_classes``.** It
   multiplied the per-class loss vector by the one-hot target and then took ``.mean()`` over all
   ``C`` classes, so ``C-1`` zeros were averaged in. The new implementation gathers the true-class
   term, which is the definition.
5. **Focal loss now takes logits, not probabilities.** ``metrics.focal_loss`` clamped its input to
   ``(1e-9, 1-1e-9)`` expecting probabilities while ``losses.FocalLoss`` applied
   ``binary_cross_entropy_with_logits`` to the same tensor. One definition: logits in
   (:func:`~sbb_pytorch.metrics.functional.focal_loss`), matching every other loss here.
6. **(breaking) Jaccard loss is no longer multiplied by its smoothing constant.** Old
   ``jaccard_distance_loss`` returned ``(1 - jac) * 100``; new returns ``1 - jac`` in ``[0, 1]``.
7. **Generalized Dice is computed per image, not per batch.** ``generalized_dice_coeff`` summed
   over ``axes = range(1, ndim - 1)``, which on a ``(B, C, H, W)`` tensor is ``(C, H)``: it folded
   the class axis into the spatial one, weighted by per-image sums, and merged all images into one
   number. Per-image coefficients are now averaged. The ``1 / count**2`` weighting is unchanged.
8. **``ignore_index`` means exactly one thing: a target label whose pixels are excluded.**
   ``metrics.IoU``/``DiceScore`` also filtered the computed vector with ``scores != ignore_index``,
   comparing a ratio against a class id -- so with ``ignore_index=0`` every class scored exactly
   ``0`` disappeared from the mean and mIoU read high. It also skipped the *class*
   ``cls == ignore_index``, which the new code does not (that is a different feature; use
   ``drop_last`` or ``class_names`` on
   :class:`~sbb_pytorch.metrics.segmentation.ConfusionMatrixMetric`).
9. **(breaking) Metrics pool counts instead of averaging per-batch ratios.** ``MetricTracker``
   averaged batch means, so a batch of one image weighed the same as a batch of sixty-four, and
   ``seg_metrics`` averaged per-image per-class scores only over images where the class appeared.
   The new numbers are batch-order independent; epoch curves will shift slightly.
10. **Smoothing constants are consolidated.** ``1.0`` (``losses.DiceLoss``), ``0.001``
    (``metrics.seg_metrics``), ``1e-8`` (``metrics.IoU``) and ``100.0``
    (``jaccard_distance_loss``) become ``DEFAULT_SMOOTHING = 1.0`` and ``DEFAULT_EPSILON = 1e-8``,
    both overridable per function. ``DiceLoss(reduction="micro")`` preserves the old value exactly.
11. **``losses.SSIMLoss`` / ``get_loss_function("enhancement")`` used to raise ``NameError``**
    (``np.exp`` with numpy unimported), so no enhancement loss could be built. Now constructible.
12. **SSIM uses a real Gaussian and reflect padding, and derives ``C1``/``C2`` from
    ``data_range``.** The legacy window exponent was ``-x**2 / sigma**2`` (missing the factor 2, so
    an effective sigma of ``1.06`` for the claimed ``1.5``); ``F.conv2d(padding=k//2)`` zero-padded,
    averaging black into every border window and depressing the score; ``C1 = 0.01 ** 2`` silently
    assumed images in ``[0, 1]`` while the pipeline decodes ``0..255``. Matches
    ``skimage.metrics.structural_similarity`` with ``use_sample_covariance=False``.
13. **``metrics.weighted_categorical_crossentropy`` / ``image_categorical_cross_entropy`` are
    folded into** :func:`~sbb_pytorch.metrics.functional.cross_entropy_loss`. They fed
    *probabilities* into ``binary_cross_entropy_with_logits`` (a sigmoid on top of a softmax) and
    built their weight mask with ``torch.maximum(..., ones)``, which floored every class weight at
    ``1.0`` -- a weight below 1 could not be expressed at all.
14. **Prediction layout is always stated, never guessed.** ``class_tversky`` and
    ``generalized_dice_coeff`` did ``permute(3, 1, 2, 0)`` (channels-last, i.e. TensorFlow) on
    PyTorch tensors; ``metrics.IoU`` used ``argmax(dim=1)`` (channels-first) and
    ``seg_metrics`` used ``shape[-1]`` as the class count (channels-last). Every function here takes
    ``layout="channels_first"|"channels_last"`` and validates the channel count against
    ``n_classes``.
15. **Tasks ``binarization`` and ``reading_order`` are handled** by
    :func:`get_metrics`; the old function raised ``ValueError`` for any task other than
    ``segmentation`` / ``classification`` / ``enhancement``.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from sbb_pytorch.exceptions import UnsupportedTaskError
from sbb_pytorch.metrics import enhance, functional, segmentation
from sbb_pytorch.metrics.enhance import (
    PSNRMetric,
    SSIMMetric,
    gaussian_window,
    psnr,
    psnr_from_mse,
    ssim,
    ssim_map,
)
from sbb_pytorch.metrics.functional import (
    DEFAULT_EPSILON,
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_FOCAL_TVERSKY_POWER,
    DEFAULT_SMOOTHING,
    Layout,
    accuracy_from_confusion_matrix,
    as_class_pair,
    class_probabilities,
    confusion_matrix,
    cross_entropy_loss,
    dice_coefficient,
    dice_from_confusion_matrix,
    dice_loss,
    focal_loss,
    focal_tversky_distance,
    focal_tversky_loss,
    generalized_dice_coefficient,
    generalized_dice_loss,
    iou_from_confusion_matrix,
    jaccard_coefficient,
    jaccard_loss,
    micro_dice,
    one_hot,
    per_class_counts,
    precision_recall_f1_from_confusion_matrix,
    soft_dice_loss,
    tversky_coefficient,
    tversky_loss,
)
from sbb_pytorch.metrics.segmentation import (
    ConfusionMatrixMetric,
    Metric,
    accuracy,
    dice_macro,
    dice_micro,
    dice_score,
    mean_iou,
    micro_dice_from_matrix,
)

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "DEFAULT_FOCAL_TVERSKY_POWER",
    "DEFAULT_SMOOTHING",
    "PSNRMetric",
    "SSIMMetric",
    "ConfusionMatrixMetric",
    "Layout",
    "Metric",
    "accuracy",
    "accuracy_from_confusion_matrix",
    "as_class_pair",
    "class_probabilities",
    "confusion_matrix",
    "compute_metrics",
    "cross_entropy_loss",
    "dice_coefficient",
    "dice_from_confusion_matrix",
    "dice_loss",
    "dice_macro",
    "dice_micro",
    "dice_score",
    "enhance",
    "focal_loss",
    "focal_tversky_distance",
    "focal_tversky_loss",
    "functional",
    "gaussian_window",
    "generalized_dice_coefficient",
    "generalized_dice_loss",
    "get_metrics",
    "iou_from_confusion_matrix",
    "jaccard_coefficient",
    "jaccard_loss",
    "mean_iou",
    "micro_dice",
    "micro_dice_from_matrix",
    "one_hot",
    "per_class_counts",
    "precision_recall_f1_from_confusion_matrix",
    "psnr",
    "psnr_from_mse",
    "segmentation",
    "soft_dice_loss",
    "ssim",
    "ssim_map",
    "tversky_coefficient",
    "tversky_loss",
]

#: Tasks whose metrics are pixel-wise confusion-matrix metrics.
_SEGMENTATION_LIKE = ("segmentation", "binarization", "classification")


def get_metrics(task: str, n_classes: int | None = None, **kwargs: Any) -> dict[str, Metric]:
    """Build the metrics for a task -- the replacement for ``metrics.get_metrics``.

    Args:
        task: ``segmentation`` / ``binarization`` / ``classification`` (confusion-matrix metrics),
            or ``enhancement`` (PSNR + SSIM).
        n_classes: Required for the pixel-wise tasks. ``binarization`` defaults to 2 and
            ``classification`` may pass its own count.
        **kwargs: Forwarded to the metric constructors (``class_names``, ``layout``,
            ``ignore_index``, ``data_range``, ...).

    Returns:
        ``{"iou": ...}``-style mapping. One :class:`ConfusionMatrixMetric` supplies iou, dice,
        precision, recall and F1 at once, instead of the legacy separate ``IoU``/``DiceScore``
        objects that each recomputed the same counts.

    Raises:
        ValueError: an unknown task, or a missing ``n_classes`` where one is required.
        UnsupportedTaskError: ``reading_order``, which has no pixel-wise metric -- machine reading
            order is validated through :mod:`sbb_pytorch.gt` instead. It subclasses
            ``NotImplementedError``, which is what ``training.module._metrics_for`` catches.
    """
    # str() of an Enum member is "TaskType.SEGMENTATION"; the value is what we mean.
    name = str(getattr(task, "value", task)).lower()

    if name == "reading_order":
        raise UnsupportedTaskError(
            "reading_order has no pixel-wise metric; order is validated via sbb_pytorch.gt"
        )
    if name == "enhancement":
        return {
            "psnr": PSNRMetric(**kwargs),
            "ssim": SSIMMetric(**kwargs),
        }
    if name in _SEGMENTATION_LIKE:
        resolved = n_classes if n_classes is not None else (2 if name == "binarization" else None)
        if resolved is None:
            raise ValueError(f"n_classes is required for task={name}")
        return {"segmentation": ConfusionMatrixMetric(n_classes=resolved, **kwargs)}

    raise ValueError(
        f"unknown task {task!r}; expected segmentation, binarization, classification, "
        "enhancement or reading_order"
    )


def compute_metrics(
    metrics: dict[str, Metric],
    predictions: Tensor,
    targets: Tensor,
    reset: bool = False,
) -> dict[str, float]:
    """Feed one batch to every metric and return their flattened values.

    Replaces ``metrics.compute_metrics``, which called ``update`` then ``compute`` on every metric
    and so recomputed from scratch on every batch (``MetricTracker`` accumulated a list each time).
    With ``reset=True`` this is also the end-of-epoch variant.
    """
    results: dict[str, float] = {}
    for metric_name, metric in metrics.items():
        metric.update(predictions, targets)
        for key, value in (metric.compute_and_reset() if reset else metric.compute()).items():
            results[f"{metric_name}/{key}"] = value
    return results
