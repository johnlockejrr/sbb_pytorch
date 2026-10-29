"""Tests for :mod:`sbb_pytorch.losses`: every wrapper must equal the function it delegates to.

Two things are checked for each loss, because each has historically been broken in a different way:

1. **Value parity with the metric side.** ``losses.py`` used to carry its own copies of the
   formulas, so ``losses.DiceLoss`` and ``metrics.DiceScore`` disagreed on the same batch. The
   wrappers are now thin, so a test that the module and the function return the same number is a
   test that the module contains no arithmetic of its own.
2. **Gradients actually flow.** ``metrics.class_tversky`` and ``generalized_dice_coeff`` permuted
   a channels-last tensor onto channels-first data, which produced a number that was wrong but
   differentiable -- and ``losses.SSIMLoss`` did not even construct (``NameError``). So
   ``.backward()`` is called on every loss and the parameter's gradient is asserted non-zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sbb_pytorch import losses
from sbb_pytorch.metrics import enhance, functional
from sbb_pytorch.metrics.functional import (
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_SMOOTHING,
)

LOGITS = torch.zeros(2, 3, 1, 2)
LABELS = torch.tensor([[[0, 1]], [[1, 2]]])
IMAGES = torch.rand(2, 3, 16, 16)

#: Each registry name -> the function it wraps, plus the reduction spelling that class accepts.
#: ``dice_loss`` speaks micro/macro/none while every other loss speaks mean/sum/none, so the
#: mapping is stated rather than probed.
_WRAPPED: dict[str, tuple[Callable[..., Tensor], dict[str, Any]]] = {
    "cross_entropy": (functional.cross_entropy_loss, {"reduction": "mean"}),
    "dice": (functional.dice_loss, {"reduction": "micro"}),
    "soft_dice": (functional.soft_dice_loss, {"reduction": "mean"}),
    "focal": (functional.focal_loss, {"reduction": "mean"}),
    "tversky": (functional.tversky_loss, {"reduction": "mean"}),
    "focal_tversky": (functional.focal_tversky_loss, {"reduction": "mean"}),
    "generalized_dice": (functional.generalized_dice_loss, {"reduction": "mean"}),
    "jaccard": (functional.jaccard_loss, {"reduction": "mean"}),
}


def _module(name: str) -> nn.Module:
    """Build a loss by name with the shapes these tests share."""
    if name == "ssim":
        return losses.SSIMLoss()
    if name == "combined":
        return losses.CombinedLoss(n_classes=3)
    return losses.LOSS_REGISTRY[name](n_classes=3)


# ----------------------------------------------------------- every loss, one test


class TestEveryLoss:
    @pytest.mark.parametrize("name", [*losses.LOSS_REGISTRY, "ssim", "combined"])
    def test_constructs_and_returns_finite_loss(self, name: str) -> None:
        module = _module(name)
        if name == "ssim":
            value = module(IMAGES, IMAGES)
        elif name == "combined":
            value = module(LOGITS, LABELS)
        else:
            value = module(LOGITS, LABELS)
        assert torch.isfinite(value).all(), f"{name} returned a non-finite loss"
        assert value.ndim == 0 or value.shape[0] == 2, f"{name} must reduce or return per-sample"

    @pytest.mark.parametrize("name", [*losses.LOSS_REGISTRY, "ssim", "combined"])
    def test_gradients_flow(self, name: str) -> None:
        # A silent autograd break is the failure mode these losses have historically had.
        if name == "ssim":
            parameter = nn.Parameter(IMAGES.clone())
            loss = losses.SSIMLoss()(parameter, IMAGES)
        else:
            parameter = nn.Parameter(torch.zeros(2, 3, 1, 2))
            loss = _module(name)(parameter, LABELS)
        loss = loss.sum() if loss.ndim else loss
        loss.backward()
        assert parameter.grad is not None, f"{name} produced no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} produced a non-finite gradient"
        assert float(parameter.grad.abs().sum()) > 0.0, f"{name} produced an all-zero gradient"

    @pytest.mark.parametrize("name", list(losses.LOSS_REGISTRY))
    def test_module_matches_the_function_it_wraps(self, name: str) -> None:
        # The wrapper must hold no arithmetic of its own: its number is the function's number.
        function, extra = _WRAPPED[name]
        expected = function(LOGITS, LABELS, n_classes=3, **extra).detach()
        assert torch.allclose(_module(name)(LOGITS, LABELS).detach(), expected, atol=1e-6)

    def test_registry_names_and_available_losses_agree(self) -> None:
        assert set(losses.LOSS_REGISTRY) == {
            "cross_entropy",
            "dice",
            "soft_dice",
            "focal",
            "tversky",
            "focal_tversky",
            "generalized_dice",
            "jaccard",
        }
        assert set(_WRAPPED) == set(losses.LOSS_REGISTRY), "_WRAPPED is out of date"
        assert set(losses.available_losses()) == set(losses.LOSS_REGISTRY) | {"combined", "ssim"}
        for name, cls in losses.LOSS_REGISTRY.items():
            assert issubclass(cls, nn.Module), f"{name} is not an nn.Module"
            assert cls.__name__ in losses.__all__, f"{cls.__name__} is missing from __all__"


# --------------------------------------------------- parity with the legacy losses


class TestLegacyParity:
    def test_dice_loss_micro_reproduces_legacy_formula(self) -> None:
        # The old losses.DiceLoss: softmax over C, flatten everything, smooth=1.0.
        logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        labels = torch.tensor([0, 1])
        probs = F.softmax(logits, dim=1)
        target = F.one_hot(labels, 2).float()
        intersection = (probs * target).sum()
        legacy = 1 - (2 * intersection + 1.0) / (probs.sum() + target.sum() + 1.0)
        assert float(losses.DiceLoss()(logits, labels)) == pytest.approx(float(legacy), rel=1e-6)
        assert float(losses.DiceLoss()(logits, labels)) == pytest.approx(
            float(functional.dice_loss(logits, labels, 2)), rel=1e-6
        )

    def test_dice_loss_reductions(self) -> None:
        module = losses.DiceLoss(smooth=DEFAULT_SMOOTHING, n_classes=3, reduction="macro")
        assert float(module(LOGITS, LABELS)) == pytest.approx(
            float(functional.dice_loss(LOGITS, LABELS, 3, reduction="macro")), rel=1e-6
        )
        per_class = losses.DiceLoss(n_classes=3, reduction="none")(LOGITS, LABELS)
        assert per_class.shape == (2, 3)

    def test_weighted_cross_entropy_alias_is_cross_entropy(self) -> None:
        # `losses.WeightedCrossEntropyLoss` was the old public name; it must still import.
        assert losses.WeightedCrossEntropyLoss is losses.CrossEntropy
        logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        labels = torch.tensor([0, 1])
        assert float(losses.CrossEntropy()(logits, labels)) == pytest.approx(
            float(F.cross_entropy(logits, labels)), rel=1e-6
        )

    def test_cross_entropy_weights_are_a_buffer_so_they_move_with_the_model(self) -> None:
        # The legacy class stored `weights` as a plain attribute, so a .cuda() model kept its
        # weights on the CPU and raised.
        module = losses.CrossEntropy(weights=torch.tensor([1.0, 2.0, 3.0]), n_classes=3)
        assert "weights" in module.state_dict()
        assert [b.shape for b in module.buffers()] == [torch.Size([3])]
        assert float(module(LOGITS, LABELS)) == pytest.approx(
            float(F.cross_entropy(LOGITS, LABELS, weight=torch.tensor([1.0, 2.0, 3.0]))), rel=1e-6
        )

    def test_cross_entropy_rejects_a_2d_weight_vector(self) -> None:
        with pytest.raises(ValueError, match="weights must be a 1-D per-class vector"):
            losses.CrossEntropy(weights=torch.zeros(2, 2))

    def test_focal_loss_defaults_match_the_single_alpha_constant(self) -> None:
        module = losses.FocalLoss()
        assert module.alpha == DEFAULT_FOCAL_ALPHA  # metrics.py used 4.0 for the same loss
        assert module.gamma == DEFAULT_FOCAL_GAMMA
        logits = torch.tensor([[0.0, 1.0]])
        assert float(module(logits, torch.tensor([0]))) == pytest.approx(
            float(functional.focal_loss(logits, torch.tensor([0]), 2)), rel=1e-6
        )

    def test_focal_loss_rejects_a_misshapen_alpha_vector(self) -> None:
        with pytest.raises(ValueError, match="alpha must be a scalar or a 1-D per-class vector"):
            losses.FocalLoss(alpha=torch.zeros(2, 2))

    def test_ssim_loss_is_one_minus_ssim(self) -> None:
        # The legacy SSIMLoss raised NameError on construction; a second bug returned the score
        # itself instead of 1 - score.
        module = losses.SSIMLoss()
        assert float(module(IMAGES, IMAGES)) == pytest.approx(
            float(1 - enhance.ssim(IMAGES, IMAGES).mean()), abs=1e-6
        )
        assert float(module(IMAGES, IMAGES)) == pytest.approx(0.0, abs=1e-6)
        assert module(IMAGES, IMAGES).ndim == 0

    def test_ssim_loss_reduction_none_is_per_image(self) -> None:
        per_image = losses.SSIMLoss(reduction="none")(IMAGES, IMAGES)
        assert per_image.shape == (2,)
        assert torch.allclose(per_image, 1 - enhance.ssim(IMAGES, IMAGES), atol=1e-6)

    def test_ssim_loss_honours_data_range(self) -> None:
        # C1/C2 used to be hard-coded for [0, 1]; a 255-range pair must score like the same
        # images normalised to [0, 1].
        wide = losses.SSIMLoss(data_range=2.0)(IMAGES * 2, IMAGES * 2)
        narrow = losses.SSIMLoss(data_range=1.0)(IMAGES, IMAGES)
        assert float(wide) == pytest.approx(float(narrow), rel=1e-5)

    def test_combined_loss_weighted_sum(self) -> None:
        module = losses.CombinedLoss("dice", "cross_entropy", 0.7, 0.3, n_classes=3)
        expected = 0.7 * losses.DiceLoss(n_classes=3)(LOGITS, LABELS) + 0.3 * losses.CrossEntropy(
            n_classes=3
        )(LOGITS, LABELS)
        assert float(module(LOGITS, LABELS)) == pytest.approx(float(expected), rel=1e-6)
        assert (module.first_weight, module.second_weight) == (0.7, 0.3)

    def test_combined_loss_defaults_reproduce_the_legacy_class(self) -> None:
        module = losses.CombinedLoss(n_classes=3)
        assert isinstance(module.first, losses.DiceLoss)
        assert isinstance(module.second, losses.CrossEntropy)
        assert (module.first_weight, module.second_weight) == (0.5, 0.5)

    def test_combined_loss_accepts_prebuilt_modules(self) -> None:
        dice = losses.DiceLoss(n_classes=3)
        module = losses.CombinedLoss(dice, "cross_entropy", 1.0, 0.0, n_classes=3)
        assert module.first is dice
        assert float(module(LOGITS, LABELS)) == pytest.approx(float(dice(LOGITS, LABELS)), rel=1e-6)

    def test_combined_loss_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="loss weights must be non-negative"):
            losses.CombinedLoss("dice", "ce", -1.0, 0.5, n_classes=3)


# ------------------------------------------------------------- get_loss_function


class TestGetLossFunction:
    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("segmentation", losses.CombinedLoss),
            ("binarization", losses.CombinedLoss),
            ("classification", losses.FocalLoss),
            ("enhancement", losses.SSIMLoss),
        ],
    )
    def test_task_defaults(self, task: str, expected: type[nn.Module]) -> None:
        assert type(losses.get_loss_function(task, n_classes=2)) is expected

    def test_reading_order_raises_not_implemented(self) -> None:
        # The legacy function said "Unknown task type"; this port says why there is no loss.
        with pytest.raises(NotImplementedError, match="reading_order.* has no trainable loss"):
            losses.get_loss_function("reading_order")

    def test_unknown_task(self) -> None:
        with pytest.raises(ValueError, match="unknown task 'nope'"):
            losses.get_loss_function("nope")

    def test_legacy_positional_signature_still_works(self) -> None:
        # get_loss_function("segmentation", weights, dice_weight, ce_weight) is the old call.
        weights = torch.tensor([1.0, 2.0, 3.0])
        module = losses.get_loss_function("segmentation", weights, 0.7, 0.3, n_classes=3)
        assert isinstance(module, losses.CombinedLoss)
        assert (module.first_weight, module.second_weight) == (0.7, 0.3)
        # The weights belong to the cross-entropy half; Dice has none (the old call passed them
        # positionally, so the wiring -- not the argument order -- is what has to stay put).
        assert isinstance(module.second, losses.CrossEntropy)
        assert module.second.weights is not None and torch.equal(module.second.weights, weights)
        assert not hasattr(module.first, "weights")

    def test_task_is_case_insensitive(self) -> None:
        assert isinstance(
            losses.get_loss_function("Segmentation", n_classes=3), losses.CombinedLoss
        )

    @pytest.mark.parametrize("name", list(losses.LOSS_REGISTRY))
    def test_loss_override_by_name(self, name: str) -> None:
        module = losses.get_loss_function("segmentation", loss=name, n_classes=3)
        assert type(module) is losses.LOSS_REGISTRY[name]

    def test_ssim_override_for_enhancement(self) -> None:
        assert isinstance(losses.get_loss_function("enhancement", loss="ssim"), losses.SSIMLoss)

    def test_enhancement_rejects_a_logits_loss(self) -> None:
        with pytest.raises(ValueError, match="task='enhancement' needs an image-pair loss"):
            losses.get_loss_function("enhancement", loss="dice")

    @pytest.mark.parametrize(
        "alias", ["ce", "weighted_cross_entropy", "weighted_categorical_crossentropy"]
    )
    def test_legacy_ce_aliases(self, alias: str) -> None:
        assert (
            type(losses.get_loss_function("segmentation", loss=alias, n_classes=3))
            is losses.CrossEntropy
        )

    def test_prebuilt_module_passes_through(self) -> None:
        module = losses.JaccardLoss(n_classes=3)
        assert losses.get_loss_function("segmentation", loss=module, n_classes=3) is module

    def test_unknown_loss_name_lists_the_options(self) -> None:
        with pytest.raises(ValueError, match=r"unknown loss 'nope'; available: .*ssim"):
            losses.get_loss_function("segmentation", loss="nope", n_classes=3)

    def test_alpha_gamma_forward_to_focal(self) -> None:
        module = losses.get_loss_function("classification", alpha=0.4, gamma=1.5, n_classes=2)
        assert (module.alpha, module.gamma) == (0.4, 1.5)

    def test_ignore_index_and_layout_forward(self) -> None:
        module = losses.get_loss_function(
            "segmentation", loss="dice", n_classes=3, layout="channels_last", ignore_index=0
        )
        assert module.layout == "channels_last" and module.ignore_index == 0

    def test_combined_via_loss_argument(self) -> None:
        module = losses.get_loss_function("binarization", loss="combined", n_classes=2)
        assert isinstance(module, losses.CombinedLoss)

    def test_module_is_stateable(self) -> None:
        # A Lightning module owns its loss, so the loss must survive state_dict round-trips.
        module = losses.CrossEntropy(weights=torch.tensor([1.0, 2.0]), n_classes=2)
        clone = losses.CrossEntropy(weights=torch.zeros(2), n_classes=2)
        clone.load_state_dict(module.state_dict())
        assert torch.equal(clone.weights, module.weights)
        # An unweighted loss has no `weights` entry at all (PyTorch drops None buffers), so its
        # state_dict is empty -- and it only accepts another unweighted one, strict as usual.
        assert losses.CrossEntropy(n_classes=2).state_dict() == {}
        losses.CrossEntropy(n_classes=2).load_state_dict({})


# ------------------------------------------------------- constructor plumbing


class TestModulePlumbing:
    @pytest.mark.parametrize(
        ("name", "extra"),
        [
            ("cross_entropy", {"reduction": "sum"}),
            ("dice", {"smooth": 0.5, "reduction": "macro"}),
            ("soft_dice", {"epsilon": 1e-3}),
            ("focal", {"alpha": 0.4, "gamma": 1.0}),
            ("tversky", {"alpha": 0.3, "beta": 0.7}),
            ("focal_tversky", {"alpha": 0.3, "beta": 0.7, "power": 2.0}),
            ("generalized_dice", {"epsilon": 1e-5}),
            ("jaccard", {"smoothing": 0.5}),
        ],
    )
    def test_constructor_arguments_are_stored(self, name: str, extra: dict[str, object]) -> None:
        module = losses.LOSS_REGISTRY[name](n_classes=3, **extra)  # type: ignore[call-arg]
        for attribute, value in extra.items():
            assert getattr(module, attribute) == value, f"{name}.{attribute} was not stored"
        assert module.n_classes == 3

    @pytest.mark.parametrize("name", list(losses.LOSS_REGISTRY))
    def test_layout_and_ignore_index_are_stored(self, name: str) -> None:
        module = losses.LOSS_REGISTRY[name](n_classes=3, layout="channels_last", ignore_index=2)
        assert module.layout == "channels_last"
        assert module.ignore_index == 2

    def test_loss_functions_module_exports_nothing_undefined(self) -> None:
        assert len(losses.__all__) == len(set(losses.__all__))
        for name in losses.__all__:
            assert hasattr(losses, name), f"{name} is exported but missing"

    def test_reductions_are_validated_at_construction(self) -> None:
        with pytest.raises(ValueError, match="reduction must be"):
            losses.CrossEntropy(n_classes=3, reduction="median")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reduction must be"):
            losses.DiceLoss(n_classes=3, reduction="median")  # type: ignore[arg-type]

    def test_loss_value_matches_between_module_call_and_forward(self) -> None:
        module = losses.DiceLoss(n_classes=3)
        assert torch.equal(module(LOGITS, LABELS), module.forward(LOGITS, LABELS))

    def test_float32_and_float64_inputs_both_work(self) -> None:
        # metrics.functional casts to float32 internally; a float64 batch must not raise.
        value = losses.DiceLoss(n_classes=3)(LOGITS.double(), LABELS)
        assert math.isfinite(float(value))
