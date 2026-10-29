"""Segmentation metrics: one ``Metric`` interface, one confusion matrix, one answer.

``metrics.py`` exposed ``MetricTracker`` (a list of floats plus ``np.mean``), ``Accuracy``,
``IoU``, ``DiceScore``, and separately ``seg_metrics()`` / ``mean_iou()`` / ``jaccard_distance_loss``
-- five ways to compute essentially two numbers, each with its own smoothing constant and its own
idea of how to average:

* ``MetricTracker.compute`` averaged the per-batch results, so a batch of one image counted as
  much as a batch of sixty-four.
* ``IoU.update``/``DiceScore.update`` looped over classes in python, then applied
  ``iou[iou != self.ignore_index]`` -- comparing a *ratio* against a *class id*, which silently
  deleted every class scored exactly ``0`` and pushed the mean up.
* ``seg_metrics`` averaged per-image per-class scores, optionally dropping the last class, and
  averaged only over images where the class was present (``mask``), which makes the reported
  number depend on batch composition.

All of that collapses into :class:`ConfusionMatrixMetric`: one accumulated ``C x C`` matrix from
:func:`sbb_pytorch.metrics.functional.confusion_matrix`, from which accuracy, per-class IoU,
mIoU, macro/micro Dice and precision/recall/F1 are all derived by formula. Pooling counts instead
of averaging ratios is what makes the epoch number independent of batch size and order.

The one-shot :func:`mean_iou` / :func:`dice_score` / :func:`accuracy` helpers are thin wrappers
over a throwaway instance, replacing ``metrics.mean_iou`` and friends for evaluation scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from sbb_pytorch.metrics.functional import (
    DEFAULT_EPSILON,
    Layout,
    accuracy_from_confusion_matrix,
    confusion_matrix,
    dice_from_confusion_matrix,
    iou_from_confusion_matrix,
    precision_recall_f1_from_confusion_matrix,
)

__all__ = [
    "ConfusionMatrixMetric",
    "Metric",
    "accuracy",
    "dice_macro",
    "dice_micro",
    "dice_score",
    "mean_iou",
    "micro_dice_from_matrix",
]


class Metric(ABC):
    """Accumulate-then-compute interface shared by every metric in the package.

    The three-method shape (``update`` / ``compute`` / ``reset``) is what lets
    :mod:`sbb_pytorch.training` log any metric without knowing which one it has, replacing the
    legacy ``MetricTracker`` subclass hierarchy that existed only to give ``Accuracy``, ``IoU``,
    ``DiceScore``, ``PSNR`` and ``SSIM`` a shared ``values`` list.
    """

    #: Human-readable name, used as the prefix for logged keys.
    name: str = "metric"

    @abstractmethod
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Accumulate one batch. ``predictions``/``targets`` shapes follow the subclass."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """All metrics accumulated so far, as flat ``{key: value}`` for logging."""

    @abstractmethod
    def reset(self) -> None:
        """Forget everything accumulated so far."""

    def compute_and_reset(self) -> dict[str, float]:
        """``compute()`` then ``reset()`` -- the end-of-epoch convenience."""
        result = self.compute()
        self.reset()
        return result


def micro_dice_from_matrix(cm: Tensor, epsilon: float = DEFAULT_EPSILON) -> float:
    """Micro Dice from a confusion matrix: ``2*TP / (2*TP + FP + FN)``, pooled over all classes.

    ``dice_from_confusion_matrix`` averages per class (macro); this pools every class first, which
    is what ``losses.DiceLoss`` optimised and what an unbalanced label set needs reported alongside
    the macro number. For single-label targets this equals the accuracy -- the expected
    cross-check, since ``2*TP + FP + FN == 2 * cm.sum()`` there.

    **Fixed:** the denominator used to be ``sum(cm)``, half of the correct one, so Dice could come
    out above 1 (a flawless two-class prediction scored ``2.0``). ``functional.micro_dice`` -- the
    soft-target version -- has always used the full ``sum(p) + sum(t)`` and is unchanged.
    """
    total = cm.sum().to(torch.float64)
    if total == 0:
        return float("nan")
    return float(2 * torch.diagonal(cm).to(torch.float64).sum() / (2 * total + epsilon))


class ConfusionMatrixMetric(Metric):
    """All hard-label segmentation metrics from one accumulated ``C x C`` confusion matrix.

    Args:
        n_classes: Number of classes. Required, never inferred.
        class_names: Optional names, so logged keys read ``iou/textline`` instead of ``iou/1``.
        layout: Which axis of a 4-D prediction holds the classes.
        ignore_index: Target label whose pixels are excluded (see
            :func:`sbb_pytorch.metrics.functional.one_hot`). This is the *only* meaning
            ``ignore_index`` has here -- the legacy ``IoU`` class also filtered the result vector
            by value, which dropped classes scored exactly ``ignore_index``.
        drop_last: Exclude the final class from the means, reproducing
            ``metrics.seg_metrics(drop_last=True)``, which treated the last class as background.
            Defaults to ``False`` because ``metrics.IoU`` and ``DiceScore`` did not do it.
        exclude_absent_classes: Skip classes with zero target pixels when averaging. Mirrors the
            legacy ``mask = (union != 0)`` and prevents a page with no headings from being scored
            as a failed heading.

    Example:
        >>> import torch
        >>> from sbb_pytorch.metrics import ConfusionMatrixMetric
        >>> metric = ConfusionMatrixMetric(n_classes=2, class_names=["background", "text"])
        >>> metric.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]))
        >>> round(metric.compute()["miou"], 3)
        1.0
    """

    name = "segmentation"

    def __init__(
        self,
        n_classes: int,
        class_names: Sequence[str] | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        drop_last: bool = False,
        exclude_absent_classes: bool = True,
    ) -> None:
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        if class_names is not None and len(class_names) != n_classes:
            raise ValueError(f"{len(class_names)} class_names for n_classes={n_classes}")
        self.n_classes = int(n_classes)
        self.class_names = list(class_names) if class_names is not None else None
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.drop_last = bool(drop_last)
        self.exclude_absent_classes = bool(exclude_absent_classes)
        self._matrix: Tensor | None = None

    # -------------------------------------------------------------- accumulate

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Add one batch's counts to the matrix.

        Raises:
            ValueError: a target label is outside ``[0, n_classes)`` and is not ``ignore_index``.
        """
        batch = confusion_matrix(
            predictions, targets, self.n_classes, self.layout, self.ignore_index
        )
        self._matrix = batch if self._matrix is None else self._matrix + batch

    def reset(self) -> None:
        """Forget every accumulated count."""
        self._matrix = None

    @property
    def confusion_matrix(self) -> Tensor:
        """The accumulated matrix, or a zero ``(C, C)`` matrix if nothing has been seen."""
        if self._matrix is None:
            return torch.zeros((self.n_classes, self.n_classes), dtype=torch.long)
        return self._matrix

    @property
    def num_pixels(self) -> int:
        """Counted pixels (excluding ``ignore_index``) so far."""
        return int(self.confusion_matrix.sum())

    # ---------------------------------------------------------------- compute

    def _labels(self) -> list[str]:
        if self.class_names is not None:
            return list(self.class_names)
        return [str(i) for i in range(self.n_classes)]

    def _evaluated(self, supports: Tensor) -> Tensor:
        """Boolean mask of classes to include in the means, honouring ``drop_last``."""
        keep = torch.ones(self.n_classes, dtype=torch.bool, device=supports.device)
        if self.drop_last and self.n_classes > 1:
            keep[-1] = False
        if self.exclude_absent_classes:
            keep &= supports > 0
        return keep

    def compute(self) -> dict[str, float]:
        """Every derived metric as a flat dict: ``miou``, ``dice_macro``, ``dice_micro``,
        ``accuracy``, plus ``iou/<name>``, ``dice/<name>``, ``precision/<name>``,
        ``recall/<name>`` and ``support/<name>`` per class.

        A class with no pixels anywhere gets ``nan`` rather than a fabricated ``0``, and is left
        out of the means (unless ``exclude_absent_classes=False``) -- see the class docstring for
        why the legacy ``+1e-8`` + filter pattern reported a higher mIoU than the truth.
        """
        cm = self.confusion_matrix
        if int(cm.sum()) == 0:
            return {"accuracy": float("nan"), "miou": float("nan"), "dice_macro": float("nan")}

        iou = iou_from_confusion_matrix(cm)
        dice = dice_from_confusion_matrix(cm)
        precision, recall, f1 = precision_recall_f1_from_confusion_matrix(cm)
        supports = cm.sum(dim=1).to(torch.float64)
        predicted = cm.sum(dim=0).to(torch.float64)
        evaluated = self._evaluated(supports)

        def _mean(values: Tensor) -> float:
            selected = values[evaluated]
            if selected.numel() == 0:
                return float("nan")
            return float(selected.mean())

        results: dict[str, float] = {
            "accuracy": accuracy_from_confusion_matrix(cm),
            "miou": _mean(iou),
            "dice_macro": _mean(dice),
            "dice_micro": micro_dice_from_matrix(cm),
            "f1_macro": _mean(f1),
            "num_pixels": float(self.num_pixels),
        }
        # A class is measurable when it has pixels as a target or a prediction, or when
        # `exclude_absent_classes=False` said to count absent ones anyway; otherwise it is
        # undefined and reported as nan rather than as a fabricated 0. `drop_last` only removes a
        # class from the means, never a real number from the report.
        measurable_by_class = [
            bool(supports[i] > 0 or predicted[i] > 0) or not self.exclude_absent_classes
            for i in range(self.n_classes)
        ]

        def _value(values: Tensor, index: int) -> float:
            return float(values[index]) if measurable_by_class[index] else float("nan")

        for index, label in enumerate(self._labels()):
            results[f"iou/{label}"] = _value(iou, index)
            results[f"dice/{label}"] = _value(dice, index)
            results[f"precision/{label}"] = _value(precision, index)
            results[f"recall/{label}"] = _value(recall, index)
            results[f"support/{label}"] = float(supports[index])
        return results

    def per_class(self) -> dict[str, Tensor]:
        """The raw ``(C,)`` tensors behind :meth:`compute`, for tests and custom reporting."""
        cm = self.confusion_matrix
        precision, recall, f1 = precision_recall_f1_from_confusion_matrix(cm)
        return {
            "iou": iou_from_confusion_matrix(cm),
            "dice": dice_from_confusion_matrix(cm),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": cm.sum(dim=1).to(torch.float64),
            "confusion_matrix": cm.to(torch.float64),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(n_classes={self.n_classes}, pixels={self.num_pixels}, "
            f"miou={self.compute().get('miou')})"
        )


# --------------------------------------------------------------- one-shot API


def _one_shot(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None,
    layout: Layout,
    ignore_index: int | None,
    drop_last: bool,
    key: str,
) -> Any:
    """Build a throwaway metric, feed it one batch, return one number."""
    if n_classes is None and torch.is_floating_point(targets):
        raise ValueError("n_classes is required when the targets are not integer labels")
    count = n_classes or int(torch.as_tensor(targets).max().item()) + 1
    metric = ConfusionMatrixMetric(
        n_classes=count,
        layout=layout,
        ignore_index=ignore_index,
        drop_last=drop_last,
    )
    metric.update(predictions, targets)
    return metric.compute()[key]


def mean_iou(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    drop_last: bool = False,
) -> float:
    """Mean IoU for one batch -- the replacement for ``metrics.mean_iou`` / ``metrics.IoU``.

    ``n_classes`` may be omitted when the targets are integer labels (it becomes
    ``max(target) + 1``), exactly as ``metrics.IoU`` required it to be stated. ``drop_last``
    reproduces ``metrics.seg_metrics(drop_last=True)``.
    """
    return _one_shot(predictions, targets, n_classes, layout, ignore_index, drop_last, "miou")


def dice_score(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    drop_last: bool = False,
    macro: bool = True,
) -> float:
    """Mean Dice for one batch -- the replacement for ``metrics.DiceScore``.

    ``macro=True`` averages per class (what ``DiceScore`` did); ``macro=False`` pools counts
    first, matching what ``losses.DiceLoss`` optimises.
    """
    return _one_shot(
        predictions,
        targets,
        n_classes,
        layout,
        ignore_index,
        drop_last,
        "dice_macro" if macro else "dice_micro",
    )


def dice_macro(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
) -> float:
    """Per-class-averaged Dice. Explicit spelling of ``dice_score(..., macro=True)``."""
    return dice_score(predictions, targets, n_classes, layout, ignore_index, macro=True)


def dice_micro(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
) -> float:
    """Pooled Dice. Explicit spelling of ``dice_score(..., macro=False)``."""
    return dice_score(predictions, targets, n_classes, layout, ignore_index, macro=False)


def accuracy(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
) -> float:
    """Accuracy for one batch -- replaces ``metrics.Accuracy``, which was classification-only.

    Works for ``(B, C)`` classification logits against ``(B,)`` labels and for ``(B, C, H, W)``
    segmentation logits against ``(B, H, W)`` masks, because both reduce to the same matrix.
    """
    return _one_shot(predictions, targets, n_classes, layout, ignore_index, False, "accuracy")
