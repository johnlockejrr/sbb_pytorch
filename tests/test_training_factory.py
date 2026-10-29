"""Tests for the training factory: one builder each for optimizer, scheduler, init, penalty.

Each test names the legacy behaviour it pins -- three duplicated ``get_optimizer`` bodies, an
imported-but-never-built scheduler, a weight init that ran *after* pretrained weights were
loaded, and a class-weight formula whose middle ``/ weights.min()`` step is redundant.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.exceptions import TrainingError
from sbb_pytorch.storage.images import imwrite
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


def make(**overrides: object) -> SBBConfig:
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": 32,
        "input_width": 32,
        **overrides,
    }
    return SBBConfig.model_validate(data)


@pytest.fixture
def tiny() -> nn.Module:
    """A conv + norm + linear stack: everything an init scheme has to handle."""
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4, 2))


# ---------------------------------------------------------------------- optimizers


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
        ("adadelta", torch.optim.Adadelta),
    ],
)
def test_every_configured_optimizer_has_a_builder(
    tiny: nn.Module, name: str, expected: type[torch.optim.Optimizer]
) -> None:
    """The three legacy ``get_optimizer`` bodies covered two of the four and typed the rest wrong."""
    optimizer = build_optimizer(tiny, make(optimizer=name, learning_rate=3e-4, weight_decay=1e-5))
    assert type(optimizer) is expected
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    if name == "sgd":
        # The only optimizer that takes momentum -- passing it to the others was the TypeError.
        assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)


def test_optimizer_accepts_the_enum_and_its_string(tiny: nn.Module) -> None:
    from sbb_pytorch.config.models import OptimizerType

    assert type(build_optimizer(tiny, make(optimizer=OptimizerType.ADAMW))) is torch.optim.AdamW


def test_unknown_optimizer_is_a_training_error_not_a_silent_default(tiny: nn.Module) -> None:
    with pytest.raises(ValueError, match="optimizer"):
        SBBConfig.model_validate(
            {
                "task": "segmentation",
                "n_classes": 3,
                "input_height": 32,
                "input_width": 32,
                "optimizer": "ranger",
            }
        )


# ---------------------------------------------------------------------- scheduler


def test_scheduler_is_actually_built() -> None:
    """``train.py`` imported ``ReduceLROnPlateau`` and never constructed one."""
    optimizer = build_optimizer(nn.Linear(2, 2), make())
    assert isinstance(build_scheduler(optimizer), torch.optim.lr_scheduler.ReduceLROnPlateau)


@pytest.mark.parametrize(("factor", "mode"), [(1.0, "min"), (0.0, "min"), (0.5, "median")])
def test_bad_scheduler_arguments_are_refused(factor: float, mode: str) -> None:
    optimizer = build_optimizer(nn.Linear(2, 2), make())
    with pytest.raises(ValueError, match="factor|mode"):
        build_scheduler(optimizer, factor=factor, mode=mode)


def test_scheduler_patience_scales_with_the_run() -> None:
    """A tenth of the run, and never zero -- ``patience=0`` stops on the first non-improvement."""
    for epochs in (1, 5, 50, 100):
        scheduler = scheduler_from_config(
            make(n_epochs=epochs), build_optimizer(nn.Linear(2, 2), make())
        )
        assert scheduler.patience == max(1, math.ceil(epochs / 10))


# ------------------------------------------------------------------------ init


@pytest.mark.parametrize("method", ["kaiming", "xavier", "orthogonal"])
def test_init_schemes_fill_weights_and_neutralise_norms(tiny: nn.Module, method: str) -> None:
    conv, norm, linear = tiny[0], tiny[1], tiny[3]
    before = conv.weight.detach().clone()
    norm.weight.data.zero_()
    norm.bias.data.fill_(7.0)

    init_weights(tiny, method)

    assert not torch.allclose(conv.weight, before), f"{method} left the conv weights untouched"
    assert torch.allclose(norm.weight, torch.ones_like(norm.weight))
    assert torch.allclose(norm.bias, torch.zeros_like(norm.bias))
    assert torch.isfinite(linear.weight).all()


def test_init_schemes_keep_their_own_arguments() -> None:
    """The legacy per-module helpers stay callable with their old signatures."""
    conv = nn.Conv2d(3, 4, 3)
    conv.bias.data.fill_(3.0)
    kaiming_init(conv, mode="fan_in", nonlinearity="sigmoid", bias=0.25)
    assert torch.allclose(conv.bias, torch.full_like(conv.bias, 0.25))
    xavier_init(conv, gain=2.0, bias=0.1)
    assert torch.allclose(conv.bias, torch.full_like(conv.bias, 0.1))
    orthogonal_init(conv, gain=1.5)


def test_init_accepts_the_enum_and_its_string(tiny: nn.Module) -> None:
    assert init_weights(tiny, InitMethod.XAVIER) is tiny
    assert init_weights(tiny, "xavier") is tiny


def test_init_rejects_arguments_the_scheme_does_not_take() -> None:
    """Used to be a ``TypeError`` from inside ``nn.init``; now it names the scheme and the kwargs."""
    with pytest.raises(ValueError, match="kaiming initialisation does not accept"):
        init_weights(nn.Linear(2, 2), "kaiming", gain=1.0)


# --------------------------------------------------------------- regularization


def test_regularization_matches_the_legacy_formulas(tiny: nn.Module) -> None:
    """``L2`` is a norm, not ``sum(p²)``; ``elastic_net`` decays *both* terms. Pinned verbatim."""
    expected_l1 = sum(p.abs().sum() for p in tiny.parameters()).item()
    expected_l2 = sum(p.pow(2).sum() for p in tiny.parameters()).sqrt().item()

    l1 = regularization_loss(tiny, "l1", weight_decay=0.1).item()
    l2 = regularization_loss(tiny, "l2", weight_decay=0.1).item()
    assert l1 == pytest.approx(0.1 * expected_l1)
    assert l2 == pytest.approx(0.1 * expected_l2)
    assert regularization_loss(tiny, "l2", weight_decay=0.0).item() == pytest.approx(0.0)
    mixed = regularization_loss(tiny, "elastic_net", weight_decay=0.1, l1_ratio=0.25)
    assert mixed.item() == pytest.approx(0.25 * 0.1 * expected_l1 + 0.75 * 0.1 * expected_l2)


def test_frozen_parameters_are_excluded_from_the_penalty(tiny: nn.Module) -> None:
    for p in tiny.parameters():
        p.requires_grad_(False)
    assert regularization_loss(tiny, "l2") == pytest.approx(0.0)


@pytest.mark.parametrize("kind", list(RegularizationKind))
def test_every_regularization_kind_accepts_the_enum_and_its_string(
    tiny: nn.Module, kind: RegularizationKind
) -> None:
    assert torch.isfinite(regularization_loss(tiny, kind))
    assert torch.isfinite(regularization_loss(tiny, kind.value))


def test_regularization_validates_its_arguments(tiny: nn.Module) -> None:
    with pytest.raises(ValueError, match="l1_ratio"):
        regularization_loss(tiny, "elastic_net", l1_ratio=1.5)
    with pytest.raises(ValueError):
        regularization_loss(tiny, "huber")


# ------------------------------------------------------------------ grad clipping


def test_clip_by_norm_reports_the_norm_and_clips(tiny: nn.Module) -> None:
    """One of the two legacy methods returned a norm and the other ``None``, silently."""
    for p in tiny.parameters():
        p.grad = torch.full_like(p, 10.0)
    norm = clip_gradients(tiny, max_norm=1.0)
    assert norm is not None and norm > 1.0  # the pre-clip total, as torch reports it
    assert max(p.grad.abs().max().item() for p in tiny.parameters()) <= norm

    for p in tiny.parameters():
        p.grad = torch.full_like(p, 10.0)
    assert clip_gradients(tiny, clip_value=2.0) is None
    assert max(p.grad.abs().max().item() for p in tiny.parameters()) == pytest.approx(2.0)


def test_clipping_requires_exactly_one_strategy(tiny: nn.Module) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        clip_gradients(tiny, max_norm=1.0, clip_value=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        clip_gradients(tiny, max_norm=None, clip_value=None)


# ------------------------------------------------------------------- class weights


def test_class_weights_reproduce_the_legacy_formula() -> None:
    """``1/(count+eps)``, normalise, ``/min``, normalise -- counts [2,1,1] -> [0.2, 0.4, 0.4]."""
    mask = np.array([[0, 0, 1, 2]], dtype=np.uint8)
    assert np.allclose(class_weights(mask, 3).numpy(), [0.2, 0.4, 0.4], atol=1e-6)


def test_class_weights_round_float_masks_and_ignore_out_of_range() -> None:
    """A float label image was indexed as an int array by the legacy code; now it is rounded."""
    mask = np.array([[0.0, 1.0], [7.0, 1.0]], dtype=np.float32)  # 7 is out of range for 2 classes
    weights = class_weights(mask, 2).numpy()
    assert np.allclose(weights, [2 / 3, 1 / 3], atol=1e-6)  # the rarer class 0 weighs more


def test_class_weights_reject_impossible_input() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        class_weights(np.zeros((2, 2), np.uint8), 0)
    with pytest.raises(ValueError, match="no masks"):
        class_weights([], 3)


def test_class_weights_from_directory_skips_unreadable_files(tmp_path: Path) -> None:
    """The legacy loop printed and continued on a bad file; the reason is logged now."""
    imwrite(tmp_path / "a.png", np.array([[0, 1]], np.uint8), channel_order="bgr")
    (tmp_path / "not-an-image.png").write_bytes(b"pretend")

    weights = class_weights_from_directory(tmp_path, 2)
    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()


def test_class_weights_from_directory_raises_on_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        class_weights_from_directory(tmp_path, 2)


def test_training_error_is_part_of_the_package_hierarchy() -> None:
    assert issubclass(TrainingError, Exception)
