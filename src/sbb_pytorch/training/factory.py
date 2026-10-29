"""One builder per training concern, all of them reading ``SBBConfig``.

What this module replaces, and why each replacement was necessary:

``train.py``'s module-level ``get_optimizer`` **and** its ``SegmentationModel.configure_optimizers``
**and** ``initialization.py`` each built an optimiser -- three implementations of the same
if/elif over ``adam``/``sgd``, agreeing only on being incomplete. ``initialization.py``'s own
docstring says it plainly: while the config carried an ``"optimizer": "adam"`` string, the code
hard-coded ``torch.optim.Adam``. :func:`build_optimizer` is now the only one, and it covers every
member of :class:`~sbb_pytorch.config.models.OptimizerType`.

``from torch.optim.lr_scheduler import ReduceLROnPlateau`` at the top of ``train.py`` was
**imported and never constructed** -- no scheduler ever ran, whatever the config said.
:func:`build_scheduler` builds one; :func:`scheduler_from_config` sizes it from ``n_epochs``.

``train.py`` called ``create_model(config)`` -- which downloads ImageNet weights whenever
``pretraining: true`` -- and then ``initialize_weights(model)``, overwriting every one of them
with a fresh kaiming draw. :func:`init_weights` still offers all three schemes, but it is only
called when no pretrained weights were loaded (see ``docs/migration.md``).

``initialization.py``'s ``WeightRegularizer.l2_regularization`` returned ``weight_decay * ‖p‖₂``
-- a **norm**, not the usual ``0.5 * sum(p²)`` -- and ``elastic_net_regularization`` applied
``weight_decay`` to both of its terms. Both formulas are reproduced exactly so old runs match;
:func:`regularization_loss` documents what they actually compute rather than what the name
suggests. ``model_utils.save_model``/``load_model``/``save_checkpoint`` are replaced by
:class:`sbb_pytorch.storage.checkpoint.Checkpoint`, and loss selection lives in
:mod:`sbb_pytorch.losses`, not here.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sbb_pytorch.config.models import OptimizerType, SBBConfig
from sbb_pytorch.exceptions import ImageReadError, TrainingError
from sbb_pytorch.logging import get_logger
from sbb_pytorch.storage.images import list_images, read_mask

__all__ = [
    "InitMethod",
    "RegularizationKind",
    "build_optimizer",
    "build_scheduler",
    "class_weights",
    "class_weights_from_directory",
    "clip_gradients",
    "init_weights",
    "kaiming_init",
    "orthogonal_init",
    "regularization_loss",
    "scheduler_from_config",
    "xavier_init",
]

#: Every normalisation layer the models use -- initialising one is always ``weight=1, bias=0``.
_NORM_TYPES: tuple[type[nn.Module], ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.GroupNorm,
    nn.InstanceNorm2d,
    nn.LayerNorm,
)


class InitMethod(str, Enum):
    """Weight-initialisation schemes ported from ``initialization.py``."""

    KAIMING = "kaiming"
    XAVIER = "xavier"
    ORTHOGONAL = "orthogonal"


class RegularizationKind(str, Enum):
    """Penalty shapes ported from ``initialization.py``'s ``WeightRegularizer``."""

    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"


# --------------------------------------------------------------------- weight init


def _norm_init(module: nn.Module) -> None:
    """``weight=1, bias=0`` -- identical for every scheme, written once instead of three times."""
    if getattr(module, "weight", None) is not None:
        nn.init.ones_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _param_init(module: nn.Module, draw: callable, bias: float) -> bool:
    """Apply ``draw`` to ``module`` if it holds trainable weights; return whether it did."""
    if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.Linear):
        draw(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
        return True
    if isinstance(module, _NORM_TYPES):
        _norm_init(module)
        return True
    return False


def kaiming_init(
    module: nn.Module, mode: str = "fan_out", nonlinearity: str = "relu", bias: float = 0.0
) -> None:
    """Kaiming (He) initialisation -- the default the legacy code applied to every model."""
    _param_init(
        module, lambda w: nn.init.kaiming_normal_(w, mode=mode, nonlinearity=nonlinearity), bias
    )


def xavier_init(module: nn.Module, gain: float = 1.0, bias: float = 0.0) -> None:
    """Glorot/Xavier normal initialisation."""
    _param_init(module, lambda w: nn.init.xavier_normal_(w, gain=gain), bias)


def orthogonal_init(module: nn.Module, gain: float = 1.0, bias: float = 0.0) -> None:
    """Orthogonal initialisation."""
    _param_init(module, lambda w: nn.init.orthogonal_(w, gain=gain), bias)


_INIT_DISPATCH: dict[InitMethod, callable] = {
    InitMethod.KAIMING: kaiming_init,
    InitMethod.XAVIER: xavier_init,
    InitMethod.ORTHOGONAL: orthogonal_init,
}


def init_weights(
    model: nn.Module, method: InitMethod | str = InitMethod.KAIMING, **kwargs: object
) -> nn.Module:
    """Apply ``method`` to every conv/linear/norm module in ``model``, in place.

    Args:
        model: Model to initialise.
        method: An :class:`InitMethod` member or its string value.
        **kwargs: Passed to the scheme (``mode``/``nonlinearity`` for kaiming, ``gain`` for the
            other two) -- the legacy ``initialize_weights(model, init_fn=...)`` accepted them too.

    Returns:
        The same ``model``, so callers can chain.

    Raises:
        ValueError: unknown ``method``, or ``kwargs`` the chosen scheme does not take (the legacy
            version raised a bare ``TypeError`` from inside ``nn.init``).
    """
    kind = method if isinstance(method, InitMethod) else InitMethod(method)
    scheme = _INIT_DISPATCH.get(kind)
    if scheme is None:  # pragma: no cover - the Enum rejects unknown values first
        raise ValueError(f"unknown init method {method!r}")
    try:
        for module in model.modules():
            scheme(module, **kwargs)
    except TypeError as exc:
        raise ValueError(
            f"{kind.value} initialisation does not accept {sorted(kwargs)}: {exc}"
        ) from exc
    return model


# ---------------------------------------------------------------------- optimizers


def _optimizer_kind(config: SBBConfig) -> OptimizerType:
    return (
        config.optimizer
        if isinstance(config.optimizer, OptimizerType)
        else OptimizerType(config.optimizer)
    )


def build_optimizer(model: nn.Module, config: SBBConfig) -> torch.optim.Optimizer:
    """The one optimiser builder in the project, covering every :class:`OptimizerType` member.

    ``momentum`` applies to SGD only: it is the argument the duplicated if/elif chains passed to
    optimizers that do not take it, and the ``TypeError`` they raised. Every member below does take
    ``weight_decay``, so the config's value is passed to all four (Adadelta defaults it to ``0.0``,
    which is why the legacy hard-coded ``Adam`` never looked wrong).

    Raises:
        TrainingError: the config names an optimizer with no builder.
    """
    parameters = model.parameters()
    lr = float(config.learning_rate)
    decay = float(config.weight_decay)
    kind = _optimizer_kind(config)

    if kind is OptimizerType.ADAM:
        return torch.optim.Adam(parameters, lr=lr, weight_decay=decay)
    if kind is OptimizerType.ADAMW:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=decay)
    if kind is OptimizerType.SGD:
        return torch.optim.SGD(
            parameters, lr=lr, momentum=float(config.momentum), weight_decay=decay
        )
    if kind is OptimizerType.ADADELTA:
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=decay)
    raise TrainingError(f"no optimizer builder for {kind.value!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 10,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """``ReduceLROnPlateau`` -- the scheduler ``train.py`` imported and never constructed.

    Raises:
        ValueError: ``factor`` outside ``(0, 1)`` (a factor of 1 never decays, 0 wipes the rate).
    """
    if not 0.0 < float(factor) < 1.0:
        raise ValueError(f"scheduler factor must be in (0, 1), got {factor}")
    if mode not in ("min", "max"):
        raise ValueError(f"scheduler mode must be 'min' or 'max', got {mode!r}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=max(1, int(patience))
    )


def scheduler_from_config(
    config: SBBConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Plateau scheduler sized from the config: wait a tenth of the run, then halve the rate."""
    return build_scheduler(
        optimizer, mode="min", factor=0.5, patience=max(1, math.ceil(config.n_epochs / 10))
    )


# ------------------------------------------------------------------- regularization


def _l1_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.stack([p.abs().sum() for p in parameters]).sum()


def _l2_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    """``‖p‖₂`` over all parameters -- the formula ``initialization.py`` used, not ``sum(p²)``."""
    return torch.stack([p.pow(2).sum() for p in parameters]).sum().sqrt()


def regularization_loss(
    model: nn.Module,
    kind: RegularizationKind | str = RegularizationKind.L2,
    *,
    weight_decay: float = 1e-4,
    l1_ratio: float = 0.5,
) -> torch.Tensor:
    """Penalty over every parameter, ported from ``initialization.WeightRegularizer``.

    ``L1`` is ``weight_decay * Σ|p|``, ``L2`` is ``weight_decay * ‖p‖₂`` (a norm -- so it grows as
    the square root of the usual weight-decay term), and ``ELASTIC_NET`` applies ``weight_decay``
    to **both** of its terms. All three match the legacy code exactly.

    Raises:
        ValueError: unknown ``kind``, or ``l1_ratio`` outside ``[0, 1]``.
    """
    resolved = kind if isinstance(kind, RegularizationKind) else RegularizationKind(kind)
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return torch.zeros((), dtype=torch.float32)
    decay = float(weight_decay)

    if resolved is RegularizationKind.L1:
        return decay * _l1_norm(parameters)
    if resolved is RegularizationKind.L2:
        return decay * _l2_norm(parameters)
    return l1_ratio * (decay * _l1_norm(parameters)) + (1.0 - l1_ratio) * (
        decay * _l2_norm(parameters)
    )


def clip_gradients(
    model: nn.Module, *, max_norm: float | None = None, clip_value: float | None = None
) -> float | None:
    """Clip gradients by norm **or** by value -- never both, never neither.

    Returns the total gradient norm for norm clipping (``clip_grad_norm_`` reports it,
    ``clip_grad_value_`` does not, and the legacy ``GradientClipper`` returned a float from one and
    ``None`` from the other without saying so).

    Neither argument has a default on purpose: the legacy class defaulted ``max_norm=1.0``, so
    asking for value clipping meant writing ``max_norm=None`` by hand, and forgetting it clipped
    by norm while reporting the method that was asked for.

    Raises:
        ValueError: both or neither argument given.
    """
    if (max_norm is None) == (clip_value is None):
        raise ValueError("give exactly one of max_norm or clip_value")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return 0.0 if max_norm is not None else None
    if max_norm is not None:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(max_norm)))
    torch.nn.utils.clip_grad_value_(parameters, clip_value=float(clip_value or 0.0))
    return None


# --------------------------------------------------------------------- class weights


def class_weights(masks: np.ndarray | list[np.ndarray], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights -- the formula ``train.py`` used, unchanged.

    ``w = 1/(count + 1e-6)``, normalised, divided by its own minimum, normalised again. The middle
    step puts the most frequent class at ``1.0`` and the last one undoes that; redundant, but it is
    what produced the published numbers, so it is kept verbatim.

    Args:
        masks: ``(H, W)`` label arrays, a stacked ``(B, H, W)`` array, or a list of either.
        n_classes: Class count; values outside ``[0, n_classes)`` are ignored rather than
            crashing the run (the legacy code counted them and indexed past the end).

    Returns:
        ``(n_classes,)`` float32 summing to 1 -- ready for
        ``torch.nn.CrossEntropyLoss(weight=...)``, which is what ``config.weighted_loss`` wants.

    Raises:
        ValueError: ``n_classes`` is not positive, or every mask was empty.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    arrays = (
        [np.asarray(masks)]
        if isinstance(masks, np.ndarray) and masks.ndim == 2
        else [np.asarray(m) for m in masks]
    )
    if not arrays:
        raise ValueError("class_weights() got no masks")

    counts = np.zeros(int(n_classes), dtype=np.float64)
    for array in arrays:
        flat = array.ravel()
        if np.issubdtype(flat.dtype, np.floating):
            flat = np.rint(flat).astype(np.int64, copy=False)
        else:
            flat = flat.astype(np.int64, copy=False)
        valid = flat[(flat >= 0) & (flat < n_classes)]
        if valid.size:
            counts += np.bincount(valid, minlength=int(n_classes))[: int(n_classes)]

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    weights = weights / weights.min()
    weights = weights / weights.sum()
    return torch.from_numpy(weights.astype(np.float32))


def _label_directory(directory: str | Path) -> Path:
    """Resolve ``config.dir_train`` to the folder that actually holds the label images.

    The two packages the config feeds disagree about what ``dir_train`` is:
    :class:`~sbb_pytorch.data.dataset.SegmentationDataset` treats it as a split root (a folder with
    ``images/`` and ``labels/``) while :mod:`sbb_pytorch.data.patches` treats it as the images
    folder (``dir_img``). Both spellings resolve here, and a plain folder of masks is still itself.
    Reading the *labels* is the point: the legacy ``calculate_class_weights`` was handed a directory
    and globbed whatever images were in it, so pointing it at a folder of photos turned a page
    photograph into a class histogram.
    """
    root = Path(directory)
    if (root / "labels").is_dir():
        return root / "labels"
    if root.name == "images" and (root.parent / "labels").is_dir():
        return root.parent / "labels"
    return root


def class_weights_from_directory(directory: str | Path, n_classes: int) -> torch.Tensor:
    """Same weights, read from a label directory (``config.dir_train``) instead of an array.

    The legacy version opened masks with ``PIL.Image.open`` inside a bare ``except`` that printed
    and continued; here the failure reason is logged through the package logger, and a directory
    with no images at all raises instead of silently yielding uniform weights.

    Raises:
        FileNotFoundError: the resolved label folder has no readable images (``list_images`` raises).
    """
    logger = get_logger("sbb.training")
    found: list[np.ndarray] = []
    for path in list_images(_label_directory(directory)):
        try:
            found.append(read_mask(path))
        except ImageReadError as exc:
            logger.warning("skipping unreadable label %s: %s", path, exc)
    if not found:
        raise FileNotFoundError(f"no readable labels in {directory}")
    return class_weights(found, n_classes)
