"""``nn.Module`` losses -- thin wrappers over :mod:`sbb_pytorch.metrics`.

Every formula lives in :mod:`sbb_pytorch.metrics.functional` (segmentation) and
:mod:`sbb_pytorch.metrics.enhance` (SSIM); this module only holds the ``nn.Module`` form that
:mod:`sbb_pytorch.training` instantiates, so a loss and the metric that reports it can never
disagree about the maths. The legacy repo kept a second copy of each formula here -- with its own
smoothing constants and, in two cases, its own bugs -- which is why ``losses.DiceLoss`` and
``metrics.DiceScore`` could report different numbers for the same batch.
:mod:`sbb_pytorch.metrics` lists the changes that produces; the ones that originate in this file
are:

* ``SSIMLoss`` raised ``NameError`` on construction (``np.exp`` with numpy unimported), which made
  ``get_loss_function("enhancement")`` -- the only enhancement loss -- unreachable.
* ``FocalLoss`` applied ``binary_cross_entropy_with_logits`` to multiclass logits (a sigmoid over
  a softmax) and averaged over every class; ``metrics.focal_loss`` used ``alpha=4.0`` where this
  file used ``0.25``. Both now come from ``metrics.DEFAULT_FOCAL_ALPHA``.
* ``DiceLoss`` only ever computed the *micro* Dice; ``reduction="macro"`` and ``"none"`` are now
  available and ``"micro"`` stays the default, so existing runs are unchanged.
* ``SoftDiceLoss`` did not exist here (it was the ``is_loss_soft_dice`` branch inside
  ``training.py``) and ``TverskyLoss``/``FocalTverskyLoss``/``GeneralizedDiceLoss``/``JaccardLoss``
  existed only in ``metrics.py`` as plain functions. They are all constructible now.
* ``get_loss_function`` accepted three tasks and raised on everything else; it now covers the
  whole ``TaskType`` enum and takes ``loss=`` so a config can name the loss instead of getting
  whatever the task default was.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from sbb_pytorch.exceptions import UnsupportedTaskError
from sbb_pytorch.metrics import enhance, functional
from sbb_pytorch.metrics.functional import (
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_FOCAL_TVERSKY_POWER,
    DEFAULT_SMOOTHING,
    Layout,
)

_R = TypeVar("_R")

# Typed as the exact literal tuples so a validated `reduction` keeps its Literal type when stored
# on a module and passed back to the functions in metrics.functional.
_MEAN_SUM_NONE: tuple[Literal["mean", "sum", "none"], ...] = ("mean", "sum", "none")
_MEAN_NONE: tuple[Literal["mean", "none"], ...] = ("mean", "none")
_MICRO_MACRO_NONE: tuple[Literal["micro", "macro", "none"], ...] = ("micro", "macro", "none")


def _check_reduction(value: _R, allowed: tuple[_R, ...]) -> _R:
    """Validate a ``reduction`` argument at construction, not on the first forward pass."""
    if value not in allowed:
        raise ValueError(f"reduction must be one of {allowed}, got {value!r}")
    return value


__all__ = [
    "CE_ALIASES",
    "LOSS_REGISTRY",
    "CombinedLoss",
    "CrossEntropy",
    "DiceLoss",
    "FocalLoss",
    "FocalTverskyLoss",
    "GeneralizedDiceLoss",
    "JaccardLoss",
    "SoftDiceLoss",
    "SSIMLoss",
    "TverskyLoss",
    "WeightedCrossEntropyLoss",
    "available_losses",
    "get_loss_function",
]


class CrossEntropy(nn.Module):
    """Weighted cross entropy on ``(B, C, ...)`` logits against ``(B, ...)`` integer labels.

    Args:
        weights: Optional per-class ``weight`` tensor, registered as a buffer so ``.to(device)``
            moves it with the model (the legacy class stored it as a plain attribute, so a
            CUDA-moved model kept its weights on the CPU and raised).
        ignore_index: Label whose pixels are excluded. ``None`` keeps PyTorch's ``-1`` behaviour.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        n_classes: Checked against the logits width instead of being trusted.
        layout: Which axis of a 4-D logits tensor holds the classes; ``"channels_last"`` logits
            are permuted rather than mis-read (``F.cross_entropy`` only accepts channels-first).

    A float one-hot target is accepted too (the Keras-era shape used by
    ``metrics.image_categorical_cross_entropy``); see
    :func:`sbb_pytorch.metrics.functional.cross_entropy_loss` for why that path changed.
    """

    def __init__(
        self,
        weights: Tensor | None = None,
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        n_classes: int | None = None,
        layout: Layout = "channels_first",
    ) -> None:
        super().__init__()
        if weights is not None:
            weights = torch.as_tensor(weights)
            if weights.dim() != 1:
                raise ValueError(
                    f"weights must be a 1-D per-class vector, got {tuple(weights.shape)}"
                )
        self.register_buffer("weights", weights)
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)
        self.n_classes = n_classes
        self.layout: Layout = layout

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Weighted cross entropy; see :func:`~sbb_pytorch.metrics.functional.cross_entropy_loss`."""
        return functional.cross_entropy_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            weights=self.weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            layout=self.layout,
        )


#: Legacy spelling kept as an alias -- ``losses.WeightedCrossEntropyLoss`` was the only weighted CE
#: in the old code, and configs/tests may still import it by that name.
WeightedCrossEntropyLoss = CrossEntropy


class DiceLoss(nn.Module):
    """Dice loss. ``reduction="micro"`` reproduces the legacy value exactly.

    Args:
        smooth: Smoothing term; ``1.0`` matches ``losses.DiceLoss``.
        reduction: ``"micro"`` (the shipped behaviour: one Dice over all classes and pixels),
            ``"macro"`` (mean over classes) or ``"none"`` (per-class ``(B, C)``).
        n_classes: Required whenever ``targets`` are integer labels, and checked against the
            logits -- the legacy class never looked at either shape.
    """

    def __init__(
        self,
        smooth: float = DEFAULT_SMOOTHING,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["micro", "macro", "none"] = "micro",
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MICRO_MACRO_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Dice loss; see :func:`sbb_pytorch.metrics.functional.dice_loss`."""
        return functional.dice_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            smoothing=self.smooth,
            reduction=self.reduction,
        )


class SoftDiceLoss(nn.Module):
    """Soft Dice loss (the old ``is_loss_soft_dice`` branch of ``training.py``, as a module)."""

    def __init__(
        self,
        epsilon: float = functional.DEFAULT_EPSILON,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Soft Dice loss; see :func:`sbb_pytorch.metrics.functional.soft_dice_loss`."""
        return functional.soft_dice_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """Multi-class focal loss on logits.

    ``alpha`` may be a scalar or a per-class vector; either way it defaults to
    :data:`~sbb_pytorch.metrics.functional.DEFAULT_FOCAL_ALPHA`, ending the split between the
    ``0.25`` used here and the ``4.0`` ``metrics.focal_loss`` used.

    ``alpha`` is validated at construction and ``reduction`` too, so a bad config fails when the
    model is built rather than on the first training step. ``layout`` is appended last (after
    ``reduction``) to keep every existing positional call working.
    """

    def __init__(
        self,
        alpha: float | Tensor = DEFAULT_FOCAL_ALPHA,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        n_classes: int | None = None,
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        layout: Layout = "channels_first",
    ) -> None:
        super().__init__()
        if isinstance(alpha, Tensor) and alpha.dim() != 1:
            raise ValueError(
                f"alpha must be a scalar or a 1-D per-class vector, got {tuple(alpha.shape)}"
            )
        self.alpha: float | Tensor = float(alpha) if not isinstance(alpha, Tensor) else alpha
        self.gamma = float(gamma)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Focal loss; see :func:`sbb_pytorch.metrics.functional.focal_loss`."""
        return functional.focal_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            layout=self.layout,
        )


class TverskyLoss(nn.Module):
    """``1 - tversky`` with ``alpha`` on false positives and ``beta`` on false negatives.

    The legacy ``metrics.class_tversky`` had the two swapped; ``alpha=0.2, beta=0.8`` therefore
    penalised false positives lightly -- the opposite of what that setting names. See
    :func:`sbb_pytorch.metrics.functional.tversky_coefficient`.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.8,
        smoothing: float = DEFAULT_SMOOTHING,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smoothing = float(smoothing)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Tversky loss; see :func:`sbb_pytorch.metrics.functional.tversky_loss`."""
        return functional.tversky_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            beta=self.beta,
            smoothing=self.smoothing,
            reduction=self.reduction,
        )


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss ``mean_c (1 + DT_c) ** power`` (the published form, not the old one)."""

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.8,
        power: float = DEFAULT_FOCAL_TVERSKY_POWER,
        epsilon: float = functional.DEFAULT_EPSILON,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.power = float(power)
        self.epsilon = float(epsilon)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Focal Tversky loss; see :func:`sbb_pytorch.metrics.functional.focal_tversky_loss`."""
        return functional.focal_tversky_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            beta=self.beta,
            power=self.power,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice loss, computed per image and then reduced.

    The legacy ``metrics.generalized_dice_coeff`` summarised over the wrong axes and merged the
    whole batch into one value; see
    :func:`sbb_pytorch.metrics.functional.generalized_dice_coefficient`.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Generalized Dice loss; see :func:`...functional.generalized_dice_loss`."""
        return functional.generalized_dice_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )


class JaccardLoss(nn.Module):
    """``1 - jaccard``, in ``[0, 1]`` -- without the legacy ``* smooth`` (``smooth=100``)."""

    def __init__(
        self,
        smoothing: float = DEFAULT_SMOOTHING,
        n_classes: int | None = None,
        layout: Layout = "channels_first",
        ignore_index: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.n_classes = n_classes
        self.layout: Layout = layout
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction, _MEAN_SUM_NONE)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Jaccard (Dice-equivalent) loss; see :func:`...functional.jaccard_loss`."""
        return functional.jaccard_loss(
            logits,
            targets,
            n_classes=self.n_classes,
            layout=self.layout,
            ignore_index=self.ignore_index,
            smoothing=self.smoothing,
            reduction=self.reduction,
        )


class SSIMLoss(nn.Module):
    """``1 - SSIM`` for enhancement. Constructible again -- it used to raise ``NameError``.

    Args:
        window_size: Odd size of the Gaussian window.
        data_range: Dynamic range the images live in (``1.0`` for normalised floats, ``255.0``
            for uint8-derived ones). The legacy class hard-coded the constants that depend on it.
        reduction: ``"mean"`` over images (the legacy ``size_average=True``) or ``"none"``, which
            returns one value per image (the legacy ``size_average=False``).
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        sigma: float = 1.5,
        reduction: Literal["mean", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.data_range = float(data_range)
        self.sigma = float(sigma)
        self.reduction = _check_reduction(reduction, _MEAN_NONE)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """``1 - ssim``; see :func:`sbb_pytorch.metrics.enhance.ssim`."""
        values = enhance.ssim(predictions, targets, self.data_range, self.window_size, self.sigma)
        per_image_loss = 1 - values
        return per_image_loss if self.reduction == "none" else per_image_loss.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of two losses -- the generalisation of the legacy Dice + CrossEntropy pair.

    ``losses.CombinedLoss(dice_weight=0.5, ce_weight=0.5)`` is reproduced exactly by
    ``CombinedLoss("dice", "cross_entropy", 0.5, 0.5)``; the two names in ``CE_ALIASES`` and the
    default arguments keep every existing call site working.
    """

    def __init__(
        self,
        first: str | nn.Module = "dice",
        second: str | nn.Module = "cross_entropy",
        first_weight: float = 0.5,
        second_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if first_weight < 0 or second_weight < 0:
            raise ValueError(
                f"loss weights must be non-negative, got {first_weight} and {second_weight}"
            )
        self.first = _build(first, **kwargs)
        self.second = _build(second, **kwargs)
        self.first_weight = float(first_weight)
        self.second_weight = float(second_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """``first_weight * first(logits, targets) + second_weight * second(logits, targets)``."""
        return self.first_weight * self.first(logits, targets) + self.second_weight * self.second(
            logits, targets
        )


#: Every loss name ``get_loss_function`` / ``CombinedLoss`` accept, mapped to its class. SSIM is
#: absent because it is an image-pair loss, not a logits loss -- ``_build`` handles it separately.
LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "cross_entropy": CrossEntropy,
    "dice": DiceLoss,
    "soft_dice": SoftDiceLoss,
    "focal": FocalLoss,
    "tversky": TverskyLoss,
    "focal_tversky": FocalTverskyLoss,
    "generalized_dice": GeneralizedDiceLoss,
    "jaccard": JaccardLoss,
}

#: Names the config files used for the same class, so old configs keep validating.
CE_ALIASES = {
    "weighted_cross_entropy": "cross_entropy",
    "weighted_categorical_crossentropy": "cross_entropy",
    "ce": "cross_entropy",
}


def _build(name_or_module: str | nn.Module, **kwargs: Any) -> nn.Module:
    """Instantiate one loss by name, dropping kwargs the chosen class does not take."""
    if isinstance(name_or_module, nn.Module):
        return name_or_module
    key = CE_ALIASES.get(str(name_or_module), str(name_or_module))
    if key == "ssim":
        accepted = {"window_size", "data_range", "sigma", "reduction"}
        return SSIMLoss(**{k: v for k, v in kwargs.items() if k in accepted})
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"unknown loss {name_or_module!r}; available: {', '.join(available_losses())}, ssim"
        )
    cls = LOSS_REGISTRY[key]
    accepted = set(cls.__init__.__code__.co_varnames[1 : cls.__init__.__code__.co_argcount])
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def available_losses() -> tuple[str, ...]:
    """Names accepted by :func:`get_loss_function` (``combined`` and ``ssim`` included)."""
    return (*tuple(LOSS_REGISTRY), "combined", "ssim")


def get_loss_function(
    task: str,
    weights: Tensor | None = None,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    *,
    loss: str | nn.Module | None = None,
    n_classes: int | None = None,
    layout: Layout = "channels_first",
    ignore_index: int | None = None,
    data_range: float = 1.0,
) -> nn.Module:
    """Build the loss for a task -- the replacement for ``losses.get_loss_function``.

    The legacy positional signature ``(task, weights, dice_weight, ce_weight, alpha, gamma)`` is
    unchanged, so ``get_loss_function("segmentation", w, 0.7, 0.3)`` still means the same thing.

    Args:
        task: ``segmentation`` | ``binarization`` | ``classification`` | ``enhancement`` |
            ``reading_order``. The old function raised ``ValueError`` on the last two.
        loss: Override the task default with any name from :func:`available_losses`, or pass a
            ready-made ``nn.Module``. Defaults: ``combined`` for segmentation/binarization,
            ``focal`` for classification, ``ssim`` for enhancement.
        n_classes: Forwarded to every loss that needs it; required for the logits-based ones
            whenever the targets are integer labels.

    Raises:
        ValueError: an unknown task or loss name.
        UnsupportedTaskError: ``reading_order``, which has no trainable loss in this port -- the
            task is inference-only (machine reading order comes from the GT pipeline). It is also
            a ``NotImplementedError``, and being an ``SBBError`` is what lets ``sbb train`` answer
            in one line instead of a traceback.
    """
    # str() of an Enum member is "TaskType.SEGMENTATION"; the value is what we mean.
    name = str(getattr(task, "value", task)).lower()
    if name == "reading_order":
        raise UnsupportedTaskError(
            f"task={task!r} has no trainable loss in this port; it is served by inference "
            "(see sbb_pytorch.gt for the ground-truth ordering)"
        )

    # Every branch goes through _build, which drops the kwargs the chosen class does not take --
    # calling FocalLoss(**shared) directly raised TypeError over `layout`, and the SSIM branch
    # cannot take n_classes/layout/ignore_index either.
    shared = {"n_classes": n_classes, "layout": layout, "ignore_index": ignore_index}
    if loss is None:
        if name == "classification":
            return _build("focal", weights=weights, alpha=alpha, gamma=gamma, **shared)
        if name == "enhancement":
            return _build("ssim", data_range=data_range, **shared)
        if name in ("segmentation", "binarization"):
            return CombinedLoss(
                first="dice",
                second="cross_entropy",
                first_weight=dice_weight,
                second_weight=ce_weight,
                weights=weights,
                **shared,
            )
        raise ValueError(
            f"unknown task {task!r}; expected segmentation, binarization, classification, "
            "enhancement or reading_order"
        )

    if isinstance(loss, nn.Module):
        return loss
    key = CE_ALIASES.get(str(loss), str(loss))
    if key == "combined":
        return CombinedLoss(
            "dice",
            "cross_entropy",
            dice_weight,
            ce_weight,
            weights=weights,
            alpha=alpha,
            gamma=gamma,
            **shared,
        )
    if name == "enhancement" or key == "ssim":
        if key != "ssim":
            raise ValueError(
                f"task={task!r} needs an image-pair loss; got loss={loss!r} (only 'ssim' applies)"
            )
        return _build("ssim", data_range=data_range, **shared)
    return _build(key, weights=weights, alpha=alpha, gamma=gamma, **shared)
