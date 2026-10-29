"""Analytic segmentation math -- one implementation of every coefficient and loss formula.

Why this module exists
----------------------
The pre-refactor repo computed segmentation numbers in **four** places that disagreed with each
other: ``metrics.seg_metrics`` (intersection/union with ``smooth=0.001`` and ``drop_last=True``),
``metrics.IoU`` (a python loop over classes, ``+1e-8``), ``utils.py`` (numpy, its own smoothing)
and ``training.py`` (Lightning's ``torchmetrics``-style IoU). ``metrics.DiceScore``,
``metrics.class_tversky``, ``metrics.generalized_dice_coeff`` and ``losses.DiceLoss`` each
recomputed the same ``sum(p*t)`` / ``sum(p)`` / ``sum(t)`` triplet with different smoothing
constants, and each independently guessed the tensor layout -- ``class_tversky`` and
``generalized_dice_coeff`` both did ``permute(3, 1, 2, 0)``, a **channels-last** (TensorFlow)
operation on **channels-first** (PyTorch) tensors, so on a ``(B, C, H, W)`` input they summarised
over the class axis and *one* spatial axis instead of over the pixels of each class.

Everything here works on one canonical shape, ``(B, C, N)``, produced by the only two entry
points: :func:`one_hot` (integer labels) and :func:`class_probabilities` (logits or
probabilities, layout given explicitly). Nothing is guessed from shape.

Bugs fixed here, each restated at its own definition
----------------------------------------------------
* **Tversky alpha/beta were swapped.** ``class_tversky`` computed
  ``tp / (tp + alpha*false_neg + beta*false_pos)``; the definition is
  ``tp / (TP + alpha*FP + beta*FN)``. With the shipped ``alpha=0.2, beta=0.8`` the code
  penalised false positives at 0.2 and false negatives at 0.8 -- the opposite of what an
  ``alpha=0.2`` (false-negative-emphasising) configuration asks for.
* **Focal-Tversky applied no dice distance.** ``focal_tversky_loss`` returned
  ``sum((1 - tversky) ** 1.3)``; the published loss is ``sum((1 + DT) ** p)`` with
  ``DT = (FP + TP) / (TP + alpha*FP + beta*FN)`` and ``p = 3``.
* **Generalized Dice summarised the wrong axes** (see above), which also merged the batch: it
  summed all images together and returned one scalar instead of one coefficient per image. The
  ``1 / count**2`` weighting itself was correct and is kept.
* **Focal ``alpha`` had two sources of truth** (``4.0`` in ``metrics.focal_loss``, ``0.25`` in
  ``losses.FocalLoss``). :data:`DEFAULT_FOCAL_ALPHA` is the only one now. ``metrics.focal_loss``
  additionally multiplied the per-class loss vector by the one-hot target and then averaged over
  *all* ``C`` classes, so the ``C-1`` zero entries were included in the mean and the loss
  shrank by a factor of ``C``; :func:`focal_loss` gathers the true-class logit instead, which is
  the definition.
* **``ignore_index`` meant two different things** in ``metrics.IoU`` -- it skipped the *class*
  ``cls == ignore_index`` *and* filtered the result with ``iou[iou != ignore_index]``,
  discarding classes whose IoU happened to equal that integer (a class scored exactly ``0`` with
  ``ignore_index=0`` vanished and mIoU rose). One meaning here: pixels whose *target label* is
  ``ignore_index``.
* **Jaccard loss was multiplied by its own smoothing constant** (``(1 - jac) * smooth``,
  ``smooth=100``), so it returned values near 100 instead of a value in ``[0, 1]``.

Smoothing constants used to be ``1.0`` (``losses.DiceLoss``), ``0.001`` (``metrics.seg_metrics``),
``1e-8`` (``metrics.IoU``) and ``100.0`` (``jaccard_distance_loss``). They are now
:data:`DEFAULT_SMOOTHING` and :data:`DEFAULT_EPSILON`, and each function takes them explicitly.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "DEFAULT_FOCAL_TVERSKY_POWER",
    "DEFAULT_SMOOTHING",
    "Layout",
    "accuracy_from_confusion_matrix",
    "as_class_pair",
    "class_probabilities",
    "confusion_matrix",
    "cross_entropy_loss",
    "dice_coefficient",
    "dice_from_confusion_matrix",
    "dice_loss",
    "focal_loss",
    "focal_tversky_distance",
    "focal_tversky_loss",
    "generalized_dice_coefficient",
    "generalized_dice_loss",
    "iou_from_confusion_matrix",
    "jaccard_coefficient",
    "jaccard_loss",
    "micro_dice",
    "one_hot",
    "per_class_counts",
    "precision_recall_f1_from_confusion_matrix",
    "soft_dice_loss",
    "tversky_coefficient",
    "tversky_loss",
]

#: ``channels_first`` is the PyTorch convention; ``channels_last`` is what the ported
#: TensorFlow-era code kept assuming. Pass it explicitly -- it is never inferred, because for a
#: square image ``(B, C, H)`` and ``(B, H, C)`` are indistinguishable.
Layout = Literal["channels_first", "channels_last"]

#: Focal weighting factor, from Lin et al. 2017. Single source of truth: ``metrics.focal_loss``
#: used ``4.0`` and ``losses.FocalLoss`` used ``0.25`` and nothing linked the two.
DEFAULT_FOCAL_ALPHA = 0.25
#: Focal focusing parameter -- the one parameter both legacy modules agreed on.
DEFAULT_FOCAL_GAMMA = 2.0
#: Focal-Tversky exponent ``p`` (Abdulrahman et al. 2021); the legacy code used ``1.3`` on a
#: different quantity entirely.
DEFAULT_FOCAL_TVERSKY_POWER = 3.0

#: Additive term in the ratio-based coefficients, matching ``losses.DiceLoss(smooth=1.0)``.
DEFAULT_SMOOTHING = 1.0
#: Guard for a division that can legitimately be by zero (a class with no pixels at all).
DEFAULT_EPSILON = 1e-8

_FLOAT = torch.float32


# --------------------------------------------------------------------- shapes


def _validate_labels(labels: Tensor, n_classes: int) -> None:
    """Raise on any label outside ``[0, n_classes)`` instead of silently mislabelling it."""
    low, high = int(labels.min()), int(labels.max())
    if low < 0 or high >= n_classes:
        raise ValueError(
            f"label ids [{low}, {high}] are outside [0, {n_classes}) for {n_classes} classes"
        )


def _validate_label_shape(logits: Tensor, labels: Tensor) -> None:
    """Check integer ``labels`` agree with ``logits`` on batch size and spatial extent.

    ``F.cross_entropy`` answers a shape mismatch with an opaque ``RuntimeError``, and the other
    losses would broadcast without complaining at all, so every integer-label loss validates here.
    """
    if labels.dim() == 0:
        raise ValueError("expected integer labels of shape (B, ...), got a 0-d tensor")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"batch mismatch: logits have {logits.shape[0]} samples, labels have "
            f"{labels.shape[0]}"
        )
    if labels.dim() == logits.dim() and tuple(labels.shape) == tuple(logits.shape):
        raise ValueError(
            f"integer labels shaped {tuple(labels.shape)} match the logits exactly, which looks "
            "like a one-hot target; pass it as a float tensor instead"
        )
    if labels.dim() > 1 and tuple(labels.shape[1:]) != tuple(logits.shape[2:]):
        raise ValueError(
            f"spatial shape mismatch: logits cover {tuple(logits.shape[2:])} per sample, labels "
            f"cover {tuple(labels.shape[1:])}"
        )


def one_hot(labels: Tensor, n_classes: int, ignore_index: int | None = None) -> Tensor:
    """Integer class ids ``(B, ...)`` -> ``(B, n_classes, N)`` float one-hot.

    Args:
        labels: Integer class ids. ``(B, H, W)`` for segmentation, ``(B,)`` for classification;
            trailing dimensions are flattened into ``N``.
        n_classes: Number of classes. Required -- never guessed from the data.
        ignore_index: Label value whose pixels belong to no class (SBB uses this for a real
            class that should not be trained on). The matching rows are all zero, which is the
            single documented meaning of ``ignore_index`` in this package.

    Raises:
        TypeError: ``labels`` is a float tensor -- pass logits to :func:`class_probabilities`.
        ValueError: a label is outside ``[0, n_classes)`` and is not ``ignore_index``.
    """
    labels = torch.as_tensor(labels)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    if torch.is_floating_point(labels):
        raise TypeError(f"one_hot expects integer class ids, got dtype {labels.dtype}")
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")

    flat = labels.to(torch.long).reshape(labels.shape[0], -1)
    if ignore_index is None:
        _validate_labels(flat, n_classes)
        return F.one_hot(flat, num_classes=n_classes).to(_FLOAT).transpose(1, 2)

    if not 0 <= int(ignore_index) < n_classes:
        raise ValueError(f"ignore_index={ignore_index} is outside [0, {n_classes})")
    outside = flat[(flat < 0) | (flat >= n_classes)]
    if outside.numel():
        raise ValueError(
            f"label ids {sorted({int(v) for v in outside.tolist()})} are outside "
            f"[0, {n_classes}); only ignore_index={ignore_index} is exempt"
        )
    hot = F.one_hot(flat, num_classes=n_classes).to(_FLOAT)
    hot = hot * (flat != int(ignore_index)).unsqueeze(-1).to(_FLOAT)
    return hot.transpose(1, 2)  # (B, N, C) -> (B, C, N)


def class_probabilities(
    predictions: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
) -> Tensor:
    """Normalise any prediction tensor to ``(B, C, N)`` float probabilities.

    Dispatch is explicit, never inferred from an ambiguous shape:

    * **integer dtype** -- class ids; one-hot encoded (``n_classes`` required). This is the
      "hard labels" path, so a metric given argmaxed predictions and one given logits agree by
      construction.
    * **float, 2-D** ``(B, C)`` -- classification logits; ``softmax`` over ``C``, ``N = 1``.
    * **float, 4-D** ``(B, C, H, W)`` or ``(B, H, W, C)`` -- image logits; ``softmax`` over the
      channel axis named by ``layout``, then ``H, W`` flattened into ``N``.
    * **float, 3-D** ``(B, C, N)`` -- already-normalised probabilities in the canonical shape,
      i.e. the output of :func:`one_hot` or of a softmax over some collapsed axis; taken as given,
      with no second softmax. Requires ``n_classes`` to disambiguate it from ``(B, H, W)``.
    * **float, 1-D or 5-D+** -- rejected; a 5-D input has no documented layout and a 1-D input has
      no class axis.

    Raises:
        ValueError: ``n_classes`` missing where required, or the channel count disagrees.
    """
    predictions = torch.as_tensor(predictions)

    if not torch.is_floating_point(predictions):
        if n_classes is None:
            raise ValueError("n_classes is required to one-hot encode integer predictions")
        return one_hot(predictions, n_classes)

    if predictions.dim() == 2:  # (B, C) classification logits
        if n_classes is not None and predictions.shape[1] != n_classes:
            raise ValueError(
                f"predictions have {predictions.shape[1]} classes, n_classes={n_classes}"
            )
        return predictions.log_softmax(dim=1).exp().unsqueeze(-1).to(_FLOAT)  # (B, C, 1)

    if predictions.dim() == 3:  # (B, C, N) probabilities -- the canonical shape, taken as given
        if n_classes is None or predictions.shape[1] != n_classes:
            raise ValueError(
                f"cannot interpret float predictions of shape {tuple(predictions.shape)}; "
                "expected (B, C) logits, (B, C, H, W) / (B, H, W, C) logits, (B, C, N) "
                "probabilities or integer labels -- pass layout= explicitly, since a square "
                "(B, C, H) tensor is ambiguous"
            )
        return predictions.to(_FLOAT)

    if predictions.dim() == 4:
        logits = predictions if layout == "channels_first" else predictions.permute(0, 3, 1, 2)
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError(
                f"predictions have {logits.shape[1]} channels with layout={layout!r}, "
                f"n_classes={n_classes}"
            )
        probs = logits.log_softmax(dim=1).exp()
        return probs.reshape(probs.shape[0], probs.shape[1], -1).to(_FLOAT)

    raise ValueError(
        f"cannot interpret float predictions of shape {tuple(predictions.shape)}; expected "
        "(B, C) logits, (B, C, H, W) / (B, H, W, C) logits, (B, C, N) probabilities, or integer "
        "labels -- pass layout= explicitly, since a square (B, C, H) tensor is ambiguous"
    )


def as_class_pair(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(probs, one_hot_target)``, both ``(B, C, N)``, for the shared metric/loss API.

    This is the only place the two inputs are reconciled, so a metric and a loss fed the same
    tensors cannot disagree about what they mean.

    Raises:
        ValueError: the two tensors disagree on batch size or class count.
    """
    probs = class_probabilities(predictions, n_classes, layout)
    if torch.is_floating_point(targets):
        target = class_probabilities(targets, probs.shape[1], layout)
    else:
        target = one_hot(targets, probs.shape[1], ignore_index)
    if probs.shape[0] != target.shape[0]:
        raise ValueError(
            f"batch mismatch: predictions have {probs.shape[0]} samples, targets have "
            f"{target.shape[0]}"
        )
    return probs, target


def per_class_counts(probs: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(B, C)`` soft intersection, prediction mass and target mass.

    The single place this triplet is computed; ``metrics.py`` and ``losses.py`` between them
    recomputed it five times with five different smoothing constants.
    """
    return (probs * target).sum(dim=2), probs.sum(dim=2), target.sum(dim=2)


# ------------------------------------------------------------ confusion matrix


def confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
) -> Tensor:
    """Hard ``(C, C)`` confusion matrix: ``cm[i, j]`` pixels whose target is ``i``, prediction ``j``.

    One ``bincount`` over ``target * C + prediction`` instead of the python loop over classes
    that ``metrics.IoU.update`` and ``DiceScore.update`` each used (``n_classes`` full tensor
    comparisons per batch, for no benefit). Accumulating this matrix across batches is what
    :class:`sbb_pytorch.metrics.segmentation.ConfusionMatrixMetric` does, which is how every
    one-shot number below becomes an epoch-level metric.

    Pixels excluded by ``ignore_index`` (their one-hot row is all zero) are left out entirely.
    """
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    count = probs.shape[1]
    predicted = probs.argmax(dim=1).reshape(-1)
    gold = target.argmax(dim=1).reshape(-1)

    # A soft target sums to 1 per pixel; an ignored pixel sums to 0. That is the exclusion test.
    keep = target.sum(dim=1).reshape(-1) > 0
    predicted, gold = predicted[keep], gold[keep]
    if predicted.numel() == 0:
        return torch.zeros((count, count), dtype=torch.long)
    return torch.bincount(gold * count + predicted, minlength=count * count).reshape(count, count)


def _tp_fp_fn(cm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-class true positives, false positives and false negatives from a confusion matrix."""
    tp = torch.diagonal(cm).to(torch.float64)
    fp = cm.sum(dim=0).to(torch.float64) - tp  # predicted c, target was something else
    fn = cm.sum(dim=1).to(torch.float64) - tp  # target c, predicted something else
    return tp, fp, fn


def iou_from_confusion_matrix(cm: Tensor, epsilon: float = DEFAULT_EPSILON) -> Tensor:
    """Per-class IoU ``TP / (FP + FN + TP)`` from a confusion matrix.

    ``metrics.IoU`` computed the same ratio and then applied ``iou = iou[iou != ignore_index]``,
    comparing a *ratio* against a *class id* -- which silently deleted every class scored exactly
    ``0`` and inflated mIoU. Here a class with no pixels yields ``0`` through ``epsilon``, and
    :func:`mean_iou` decides whether to include it.
    """
    tp, fp, fn = _tp_fp_fn(cm)
    return tp / (fp + fn + tp + epsilon)


def dice_from_confusion_matrix(cm: Tensor, epsilon: float = DEFAULT_EPSILON) -> Tensor:
    """Per-class Dice ``2TP / (2TP + FP + FN)`` from a confusion matrix.

    Identical to the macro-F1 from :func:`precision_recall_f1_from_confusion_matrix`, which is
    why ``metrics.py`` imported ``sklearn.metrics.f1_score`` and never called it.
    """
    tp, fp, fn = _tp_fp_fn(cm)
    return 2 * tp / (2 * tp + fp + fn + epsilon)


def precision_recall_f1_from_confusion_matrix(
    cm: Tensor, epsilon: float = DEFAULT_EPSILON
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-class precision ``TP/(TP+FP)``, recall ``TP/(TP+FN)`` and their harmonic mean."""
    tp, fp, fn = _tp_fp_fn(cm)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, f1


def accuracy_from_confusion_matrix(cm: Tensor) -> float:
    """Global accuracy ``diag(cm).sum() / cm.sum()``.

    Replaces ``metrics.Accuracy``, which worked only for ``(B, C)`` classification logits and
    averaged per-batch means instead of pooling counts (so a batch of 1 counted as much as a
    batch of 64).
    """
    total = float(cm.sum())
    if total == 0:
        return float("nan")
    return float(torch.diagonal(cm).sum()) / total


# --------------------------------------------------------------- coefficients


def dice_coefficient(probs: Tensor, target: Tensor, smoothing: float = DEFAULT_SMOOTHING) -> Tensor:
    """Per-image-per-class Dice ``(2*sum(p*t) + s) / (sum(p) + sum(t) + s)``, shape ``(B, C)``.

    The formula ``losses.DiceLoss`` used; that class flattened every class and pixel together,
    i.e. it computed the *micro* Dice -- see :func:`micro_dice`.
    """
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    return (2 * intersection + smoothing) / (pred_sum + target_sum + smoothing)


def micro_dice(probs: Tensor, target: Tensor, smoothing: float = DEFAULT_SMOOTHING) -> Tensor:
    """One Dice over all classes and pixels together -- exactly what ``losses.DiceLoss`` returned."""
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    return (2 * intersection.sum() + smoothing) / (pred_sum.sum() + target_sum.sum() + smoothing)


def tversky_coefficient(
    probs: Tensor,
    target: Tensor,
    alpha: float = 0.2,
    beta: float = 0.8,
    smoothing: float = DEFAULT_SMOOTHING,
) -> Tensor:
    """Per-class Tversky ``(TP + s) / (TP + alpha*FP + beta*FN + s)``, shape ``(B, C)``.

    **Fixed:** ``metrics.class_tversky`` put ``alpha`` on the false negatives and ``beta`` on the
    false positives -- the reverse of the definition -- and hard-coded both, so the shipped
    ``alpha=0.2/beta=0.8`` pairing emphasised false positives when it names the
    false-negative-weighted setting. It also did ``pred.permute(3, 1, 2, 0)``, a channels-last
    reshape that folded the class axis into the spatial one on a PyTorch ``(B, C, H, W)`` tensor.
    """
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    true_pos = intersection
    false_pos = pred_sum - intersection
    false_neg = target_sum - intersection
    return (true_pos + smoothing) / (true_pos + alpha * false_pos + beta * false_neg + smoothing)


def focal_tversky_distance(
    probs: Tensor,
    target: Tensor,
    alpha: float = 0.2,
    beta: float = 0.8,
    epsilon: float = DEFAULT_EPSILON,
) -> Tensor:
    """The dice distance ``DT = (FP + TP) / (TP + alpha*FP + beta*FN)`` that focal-Tversky needs.

    **Fixed:** this quantity was never computed by the legacy code, which squared the wrong term
    -- see :func:`focal_tversky_loss`.
    """
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    true_pos = intersection
    false_pos = pred_sum - intersection
    false_neg = target_sum - intersection
    return (false_pos + true_pos) / (true_pos + alpha * false_pos + beta * false_neg + epsilon)


def generalized_dice_coefficient(probs: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Generalized Dice per image, shape ``(B,)`` (Sud et al. 2017).

    ``GDC = 2 * sum_c w_c * sum_pixels(p*t) / sum_c w_c * sum_pixels(p + t)``, ``w_c =
    1 / (sum_pixels(t)**2 + eps)``.

    **Fixed:** the legacy ``generalized_dice_coeff`` used ``axes = range(1, ndim - 1)``, which on
    a ``(B, C, H, W)`` tensor is ``(C, H)``: the weights became per-*image* sums instead of
    per-class, and every image was merged into one number instead of one coefficient per image.
    The ``1 / count**2`` weighting itself was correct and is unchanged.
    """
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    weights = 1 / (target_sum.square() + epsilon)
    numerator = (weights * intersection).sum(dim=1)
    denominator = (weights * (pred_sum + target_sum)).sum(dim=1)
    return 2 * numerator / (denominator + epsilon)


def jaccard_coefficient(
    probs: Tensor, target: Tensor, smoothing: float = DEFAULT_SMOOTHING
) -> Tensor:
    """Per-class Jaccard ``(sum(p*t) + s) / (sum(p) + sum(t) - sum(p*t) + s)``, shape ``(B, C)``.

    **Fixed:** ``jaccard_distance_loss`` returned ``(1 - jac) * smooth`` with ``smooth=100``, so
    the "loss" sat near 100 instead of in ``[0, 1]``. The distance is now ``1 - jac``.
    """
    intersection, pred_sum, target_sum = per_class_counts(probs, target)
    return (intersection + smoothing) / (pred_sum + target_sum - intersection + smoothing)


# -------------------------------------------------------------- loss functions


def _reduce(values: Tensor, reduction: str) -> Tensor:
    """``mean`` / ``sum`` / ``none`` on a tensor of per-class or per-image losses."""
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def dice_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    smoothing: float = DEFAULT_SMOOTHING,
    reduction: Literal["micro", "macro", "none"] = "micro",
) -> Tensor:
    """Dice loss. ``reduction`` picks which of the two legacy behaviours to keep.

    Args:
        reduction: ``"micro"`` (default -- the ``losses.DiceLoss`` behaviour, one Dice over all
            classes and pixels), ``"macro"`` (mean of per-class Dices, so a rare class counts as
            much as a common one) or ``"none"`` (per-class ``(B, C)``).

    The default preserves the shipped formula rather than the nicer one: switching to ``macro``
    would move every training curve, which the refactor's rules forbid without a sign-off.
    """
    if reduction not in ("micro", "macro", "none"):
        raise ValueError(f"reduction must be 'micro', 'macro' or 'none', got {reduction!r}")
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    if reduction == "micro":  # the losses.DiceLoss behaviour: one Dice over every class and pixel
        return 1 - micro_dice(probs, target, smoothing)
    per_class_loss = 1 - dice_coefficient(probs, target, smoothing)
    return per_class_loss if reduction == "none" else per_class_loss.mean()


def soft_dice_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Soft Dice loss ``1 - 2*sum(p*t) / (sum(p^2) + sum(t^2))``, per image then mean.

    A genuinely different formula from :func:`dice_loss` (squared terms; per *image* rather than
    per class) -- this is the ``is_loss_soft_dice`` branch the old ``training.py`` exposed, and it
    is kept separate rather than merged, because merging would change both. The legacy version
    summed over ``axes = range(1, ndim - 1)``, which on ``(B, C, H, W)`` covered the class axis
    and the height only.
    """
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    numerator = 2 * (probs * target).sum(dim=(1, 2))
    denominator = (probs.square() + target.square()).sum(dim=(1, 2))
    return _reduce(1 - numerator / (denominator + epsilon), reduction)


def tversky_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    smoothing: float = DEFAULT_SMOOTHING,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """``1 - tversky_coefficient``, per class then mean."""
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    return _reduce(1 - tversky_coefficient(probs, target, alpha, beta, smoothing), reduction)


def focal_tversky_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    power: float = DEFAULT_FOCAL_TVERSKY_POWER,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Focal Tversky loss ``mean_c (1 + DT_c) ** p``, ``p = 3`` (Abdulrahman et al. 2021).

    **Fixed:** the legacy ``metrics.focal_tversky_loss`` returned ``sum((1 - tversky) ** 1.3)``,
    which is neither the dice distance nor the published exponent. See
    :func:`focal_tversky_distance`.
    """
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    distance = focal_tversky_distance(probs, target, alpha, beta, epsilon)
    return _reduce((1 + distance) ** power, reduction)


def generalized_dice_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    epsilon: float = 1e-6,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """``1 - generalized_dice_coefficient``."""
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    return _reduce(1 - generalized_dice_coefficient(probs, target, epsilon), reduction)


def jaccard_loss(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    smoothing: float = DEFAULT_SMOOTHING,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """``1 - jaccard_coefficient``, without the legacy ``* smooth`` scaling."""
    probs, target = as_class_pair(predictions, targets, n_classes, layout, ignore_index)
    return _reduce(1 - jaccard_coefficient(probs, target, smoothing), reduction)


def _channels_first(logits: Tensor, layout: Layout) -> Tensor:
    """Return ``logits`` as ``(B, C, ...)``, permuting a channels-last ``(B, ..., C)`` tensor.

    ``F.cross_entropy`` and ``torch.gather`` both need the class axis first, so a channels-last
    tensor is moved rather than rejected: the legacy repo trained TF-style ``channels_last``
    configs and PyTorch can hand back a ``channels_last``-strided batch. Tensors with no spatial
    axes -- ``(B, C)`` classification logits, ``(B, C, N)`` probabilities -- have nothing to move
    and are returned untouched; :func:`class_probabilities` covers every other shape.
    """
    if layout == "channels_first" or logits.dim() < 4:
        return logits
    return logits.permute((0, *range(2, logits.dim()), 1))


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    weights: Tensor | None = None,
    ignore_index: int | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    layout: Layout = "channels_first",
) -> Tensor:
    """Weighted cross entropy: ``(B, C, ...)`` logits against ``(B, ...)`` integer labels.

    Replaces ``losses.WeightedCrossEntropyLoss`` (correct, but the only variant that worked) and
    ``metrics.weighted_categorical_crossentropy`` / ``metrics.image_categorical_cross_entropy``,
    which fed *probabilities* into ``binary_cross_entropy_with_logits`` -- a sigmoid on top of a
    softmax -- and built their weight mask with ``torch.maximum(..., ones)``, silently flooring
    every class weight at 1.0 so a weight below 1 could not be expressed. One implementation, on
    logits, as PyTorch expects.

    A float one-hot target is also accepted (the Keras-era shape) and folds over the class axis.
    ``layout`` says which axis of a 4-D ``logits`` holds the classes, as everywhere else here: the
    legacy class assumed channels-first without asking and never looked at the target shape.
    """
    if logits.dim() < 2:
        raise ValueError(f"expected logits of shape (B, C, ...), got {tuple(logits.shape)}")
    logits = _channels_first(logits, layout)
    count = n_classes or logits.shape[1]
    if weights is not None and torch.as_tensor(weights).numel() != count:
        raise ValueError(
            f"weights has {torch.as_tensor(weights).numel()} entries, n_classes={count}"
        )
    labels = torch.as_tensor(targets)

    if torch.is_floating_point(labels):  # soft/one-hot target (B, C, ...)
        # `logits` is already (B, C, ...) after the permute above, so the target is moved the same
        # way rather than read through `layout` a second time (which would permute it back).
        probs, target = as_class_pair(logits, _channels_first(labels, layout), count)
        per_pixel = -(probs.clamp_min(1e-12).log() * target).sum(dim=1)  # (B, N)
        if weights is None:
            return _reduce(per_pixel, reduction)
        # The same normalisation F.cross_entropy applies on the integer path -- divide by the
        # weights actually used -- so the loss does not change value with the target encoding.
        weight_t = (target * weights.to(per_pixel.dtype).view(count, 1)).sum(dim=1)  # (B, N)
        weighted = per_pixel * weight_t
        if reduction == "mean":
            return weighted.sum() / weight_t.sum().clamp_min(1e-12)
        return _reduce(weighted, reduction)

    labels = labels.to(torch.long)
    _validate_label_shape(logits, labels)
    if ignore_index is None:
        _validate_labels(labels.reshape(-1), count)
    return F.cross_entropy(
        logits,
        labels,
        weight=weights,
        ignore_index=-1 if ignore_index is None else int(ignore_index),
        reduction=reduction,
    )


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    n_classes: int | None = None,
    alpha: float | Tensor = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    ignore_index: int | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    layout: Layout = "channels_first",
) -> Tensor:
    """Multi-class focal loss ``-alpha_t * (1 - p_t) ** gamma * log(p_t)`` (Lin et al. 2017).

    ``alpha`` takes a scalar or a per-class weight vector; ``alpha_t`` is gathered at the true
    label, which is the definition -- ``metrics.focal_loss`` instead multiplied the per-class
    loss vector by the one-hot target and averaged over **all** ``C`` classes, so the ``C-1``
    zero entries diluted the result by a factor of ``C``. It also clamped *probabilities* where
    ``losses.FocalLoss`` applied ``binary_cross_entropy_with_logits`` to multiclass logits.
    :data:`DEFAULT_FOCAL_ALPHA` is now the only source of truth. ``p_t`` is the probability of the
    true class -- ``exp(log p_t)``, i.e. ``exp(-nll)``; the two legacy copies each took a different
    route to a wrong ``p_t`` (a sigmoid on a softmax, or the un-softmaxed logits clamped to
    ``(1e-9, 1 - 1e-9)``). ``layout`` names the class axis of a 4-D ``logits`` tensor, exactly as
    in :func:`cross_entropy_loss`.
    """
    if logits.dim() < 2:
        raise ValueError(f"expected logits of shape (B, C, ...), got {tuple(logits.shape)}")
    logits = _channels_first(logits, layout)  # the gather below needs the class axis first
    labels = torch.as_tensor(targets)
    count = n_classes or logits.shape[1]
    if logits.shape[1] != count:
        raise ValueError(f"logits have shape {tuple(logits.shape)}, n_classes={count}")
    if isinstance(alpha, Tensor) and alpha.numel() != count:
        raise ValueError(f"alpha has {alpha.numel()} entries, n_classes={count}")

    if torch.is_floating_point(labels):  # soft/one-hot target (B, C, ...)
        probs, target = as_class_pair(logits, _channels_first(labels, layout), count)
        p_t = (probs * target).sum(dim=1)
        if isinstance(alpha, Tensor):
            alpha_t = (target * alpha.to(p_t.dtype).view(count, 1)).sum(dim=1)
        else:
            alpha_t = torch.full_like(p_t, float(alpha))
        loss = -alpha_t * (1 - p_t).pow(gamma) * p_t.clamp_min(1e-12).log()
        return _reduce(loss, reduction)

    labels = labels.to(torch.long)
    _validate_label_shape(logits, labels)
    if ignore_index is None:
        _validate_labels(labels.reshape(-1), count)
    flat = labels.reshape(labels.shape[0], -1)  # (B, N)
    index = flat.unsqueeze(1)  # (B, 1, N) -- gather needs the same rank as the logits
    logits_flat = logits.reshape(logits.shape[0], count, -1)  # (B, C, N)
    log_p_t = logits_flat.log_softmax(dim=1).gather(1, index).squeeze(1)  # log-prob of true class
    nll = -log_p_t
    p_t = log_p_t.exp()  # p_t = exp(log p_t) = exp(-nll), not exp(nll)
    if isinstance(alpha, Tensor):
        alpha_t = alpha.to(nll.dtype)[index].squeeze(1)  # per-class weight at the true label
    else:
        alpha_t = torch.full_like(nll, float(alpha))
    loss = alpha_t * (1 - p_t).pow(gamma) * nll
    if ignore_index is None:
        return _reduce(loss, reduction)

    keep = (flat != int(ignore_index)).to(loss.dtype)
    if reduction == "mean":
        return loss.mul(keep).sum() / keep.sum().clamp(min=1)
    return _reduce(loss * keep, reduction)
