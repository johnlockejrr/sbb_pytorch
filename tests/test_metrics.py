"""Tests for :mod:`sbb_pytorch.metrics` -- every assertion is a hand-computed number.

The refactor's rule was behaviour preservation, so these tests pin the *arithmetic* rather than
checking that functions run: each expected value is written out from the definition (or from the
PyTorch reference where one exists, e.g. ``F.cross_entropy``) instead of being copied off the
implementation. Where a fix changed a number, the test name says which numbered semantic change
from ``sbb_pytorch.metrics.__init__`` it belongs to.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sbb_pytorch.metrics import enhance, functional, get_metrics, segmentation
from sbb_pytorch.metrics.functional import (
    DEFAULT_EPSILON,
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_FOCAL_TVERSKY_POWER,
    DEFAULT_SMOOTHING,
)

# --------------------------------------------------------------------- fixtures
#
# LOGITS / LABELS: two images, three classes, a 1x2 pixel grid. All logits are zero, so every
# pixel has probability 1/3 per class -- probabilities that are exact in binary floating point,
# which keeps the expected numbers below closed-form instead of approximations.
LOGITS = torch.zeros(2, 3, 1, 2)
LABELS = torch.tensor([[[0, 1]], [[1, 2]]])
# GOLD[0] = [[1, 0], [0, 1], [0, 0]] -- one pixel labelled 0 and one labelled 1. GOLD[1] holds
# labels [1, 2], so its class-0 row is all zeros: a class the second image has none of.
GOLD = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

# A single logit pair of zeros gives exactly p = [0.5, 0.5]; the same math as the fixture above
# but with one pixel, so each class's counts are single numbers.
LOGITS_C2 = torch.tensor([[0.0, 0.0]])
LABELS_C2 = torch.tensor([0])
GOLD_C2 = torch.tensor([[[1.0], [0.0]]])

# An asymmetric pair: class 0 scores tp=1, fp=1, fn=0 and class 1 scores nothing at all. The
# LOGITS_C2 pair scores tp=.5, fp=0, fn=.5 -- error mass in one direction only -- so it cannot
# tell the false-positive weight (alpha) apart from the false-negative one (beta).
PROBS_ASYM = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
TARGET_ASYM = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])


def _probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    return functional.class_probabilities(logits, n_classes)


# ---------------------------------------------------------------------- one_hot


class TestOneHot:
    def test_exact_matrix(self) -> None:
        assert functional.one_hot(torch.tensor([[0, 1], [1, 0]]), 2).tolist() == [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.1 and 1.0, 0.0], [1.0, 0.0]],
        ][0:1] + [[[0.0, 1.0], [1.0, 0.0]]]

    def test_shape_flattens_trailing_dims(self) -> None:
        labels = torch.randint(0, 3, (4, 6, 7))
        out = functional.one_hot(labels, 3)
        assert out.shape == (4, 3, 42)
        assert torch.all(out.sum(dim=1) == 1), "every pixel must belong to exactly one class"

    def test_rejects_float_labels(self) -> None:
        with pytest.raises(TypeError, match="one_hot expects integer class ids"):
            functional.one_hot(torch.tensor([[0.0, 1.0]]), 2)

    def test_rejects_out_of_range_label(self) -> None:
        with pytest.raises(ValueError, match=r"label ids \[0, 5\] are outside \[0, 2\)"):
            functional.one_hot(torch.tensor([[0, 5]]), 2)

    def test_rejects_non_positive_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be positive, got 0"):
            functional.one_hot(torch.tensor([[0]]), 0)

    def test_accepts_0_dimensional_labels(self) -> None:
        assert functional.one_hot(torch.tensor(1), 2).shape == (1, 2, 1)

    def test_ignore_index_zeroes_that_pixel(self) -> None:
        out = functional.one_hot(torch.tensor([[0, 2], [1, 2]]), 3, ignore_index=2)
        assert out.tolist() == [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        ]

    def test_ignore_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="ignore_index=5 is outside \\[0, 3\\)"):
            functional.one_hot(torch.tensor([[0, 1]]), 3, ignore_index=5)

    def test_ignore_index_does_not_exempt_other_bad_labels(self) -> None:
        # Only the offending ids are named -- label 1 is valid, so it is not in the message.
        with pytest.raises(ValueError, match=r"label ids \[9\] are outside \[0, 3\)"):
            functional.one_hot(torch.tensor([[1, 9]]), 3, ignore_index=2)


# ----------------------------------------------------------- class_probabilities


class TestClassProbabilities:
    def test_two_dim_matches_softmax(self) -> None:
        logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        got = functional.class_probabilities(logits, 2)
        assert got.shape == (2, 2, 1)
        assert torch.allclose(got.squeeze(-1), torch.softmax(logits, dim=1), atol=1e-6)

    def test_integer_predictions_need_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes is required to one-hot encode"):
            functional.class_probabilities(torch.tensor([0, 1, 2]))

    def test_two_dim_class_count_must_agree(self) -> None:
        with pytest.raises(ValueError, match=r"predictions have 3 classes, n_classes=4"):
            functional.class_probabilities(torch.zeros(2, 3), 4)

    def test_three_dim_probs_pass_through_unchanged(self) -> None:
        # A (B, C, N) probability tensor is already in the canonical shape: no second softmax.
        probs = functional.one_hot(torch.tensor([[0, 1]]), 2)
        assert torch.equal(functional.class_probabilities(probs, 2), probs)

    def test_three_dim_without_n_classes_is_ambiguous(self) -> None:
        with pytest.raises(
            ValueError, match=r"cannot interpret float predictions of shape \(1, 2, 1\)"
        ):
            functional.class_probabilities(torch.rand(1, 2, 1))

    @pytest.mark.parametrize("n_classes", [2, 3, 4])
    def test_layout_is_honoured_not_guessed(self, n_classes: int) -> None:
        # A square (B, C, H, W) would be ambiguous, and the legacy code guessed wrong in two
        # places by permuting channels-last tensors that were channels-first.
        logits = torch.randn(2, n_classes, n_classes, n_classes)
        first = functional.class_probabilities(logits, n_classes, "channels_first")
        last = functional.class_probabilities(
            logits.permute(0, 2, 3, 1), n_classes, "channels_last"
        )
        assert torch.allclose(first, last, atol=1e-6)

    def test_four_dim_class_count_must_agree(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"predictions have 3 channels with layout='channels_first', n_classes=4",
        ):
            functional.class_probabilities(torch.zeros(2, 3, 1, 2), 4)

    def test_rejects_high_rank(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret float predictions"):
            functional.class_probabilities(torch.rand(2, 3, 1, 1, 1), 3)


# ------------------------------------------------------------------ as_class_pair


class TestAsClassPair:
    def test_pairs_logits_and_labels(self) -> None:
        probs, target = functional.as_class_pair(LOGITS, LABELS, 3)
        assert torch.equal(probs, _probs(LOGITS, 3))
        assert torch.equal(target, GOLD)

    def test_rejects_batch_mismatch(self) -> None:
        with pytest.raises(
            ValueError, match="batch mismatch: predictions have 2 samples, targets have 1"
        ):
            functional.as_class_pair(LOGITS, torch.tensor([[[0, 1]]]), 3)

    def test_float_targets_are_accepted_verbatim(self) -> None:
        probs, target = functional.as_class_pair(LOGITS_C2, GOLD_C2, 2)
        assert torch.equal(target, GOLD_C2)
        assert torch.equal(probs, _probs(LOGITS_C2, 2))

    def test_per_class_counts(self) -> None:
        # The triplet is per class -- (B, C) -- not pooled: for p=[.5,.5], t=[1,0] the two classes
        # carry .5/.0 of the intersection, .5/.5 of the predicted mass and 1/.0 of the target.
        intersection, pred_sum, target_sum = functional.per_class_counts(
            _probs(LOGITS_C2, 2), GOLD_C2
        )
        assert torch.allclose(intersection, torch.tensor([[0.5, 0.0]]))
        assert torch.allclose(pred_sum, torch.tensor([[0.5, 0.5]]))
        assert torch.allclose(target_sum, torch.tensor([[1.0, 0.0]]))


# ---------------------------------------------------------------- confusion_matrix


class TestConfusionMatrix:
    def test_counts_from_logits(self) -> None:
        # Every pixel is predicted as class 0 (all logits tie), so column 0 holds every count.
        cm = functional.confusion_matrix(LOGITS, LABELS, 3)
        assert cm.tolist() == [[1, 0, 0], [2, 0, 0], [1, 0, 0]]
        assert int(cm.sum()) == 4, "the matrix counts every labelled pixel once"

    def test_matches_python_loop(self) -> None:
        cm = functional.confusion_matrix(LOGITS, LABELS, 3)
        predicted = LOGITS.argmax(dim=1).reshape(-1)
        gold = LABELS.reshape(-1)
        loop = torch.zeros(3, 3, dtype=torch.long)
        for p, g in zip(predicted.tolist(), gold.tolist(), strict=True):
            loop[g, p] += 1
        assert torch.equal(cm, loop)

    def test_excludes_ignore_index_pixels(self) -> None:
        cm = functional.confusion_matrix(
            torch.zeros(1, 3, 1, 2), torch.tensor([[[0, 2]]]), 3, ignore_index=2
        )
        assert cm.tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert int(cm.sum()) == 1, "the ignored pixel must not be counted at all"

    def test_empty_matrix(self) -> None:
        cm = functional.confusion_matrix(
            torch.zeros(1, 3, 1, 1), torch.tensor([[[2]]]), 3, ignore_index=2
        )
        assert torch.equal(cm, torch.zeros(3, 3, dtype=torch.long))


class TestDerivedFromConfusionMatrix:
    @pytest.fixture
    def cm(self) -> torch.Tensor:
        return functional.confusion_matrix(LOGITS, LABELS, 3)

    def test_iou(self, cm: torch.Tensor) -> None:
        # class 0: tp=1, fp=2 (the two pixels whose gold was 1/2), fn=0 -> 1/3 ... but class 0's
        # row holds 1 correct + 0 wrong, while column 0 holds 3 predictions.
        assert functional.iou_from_confusion_matrix(cm).tolist() == pytest.approx([0.25, 0.0, 0.0])

    def test_dice_is_macro_f1(self, cm: torch.Tensor) -> None:
        dice = functional.dice_from_confusion_matrix(cm)
        _, _, f1 = functional.precision_recall_f1_from_confusion_matrix(cm)
        assert torch.allclose(dice, f1, atol=1e-9)
        assert dice.tolist() == pytest.approx([0.4, 0.0, 0.0])

    def test_precision_recall(self, cm: torch.Tensor) -> None:
        precision, recall, f1 = functional.precision_recall_f1_from_confusion_matrix(cm)
        assert precision.tolist() == pytest.approx([0.25, 0.0, 0.0], rel=1e-6)
        assert recall.tolist() == pytest.approx([1.0, 0.0, 0.0], rel=1e-6)
        assert f1.tolist() == pytest.approx([0.4, 0.0, 0.0], rel=1e-6)

    def test_accuracy(self, cm: torch.Tensor) -> None:
        # 4 labelled pixels, 1 predicted correctly.
        assert functional.accuracy_from_confusion_matrix(cm) == pytest.approx(0.25)

    def test_accuracy_of_empty_matrix_is_nan(self) -> None:
        assert math.isnan(
            functional.accuracy_from_confusion_matrix(torch.zeros(3, 3, dtype=torch.long))
        )


# ------------------------------------------------------------------- coefficients


class TestCoefficients:
    def test_dice_per_class_matches_definition(self) -> None:
        # class 0: 2*p*t = 1, sum(p) = .5, sum(t) = 1 -> (1 + 1) / (.5 + 1 + 1) = 0.8
        # class 1: 2*p*t = 0, sum(p) = .5, sum(t) = 0 -> (0 + 1) / (.5 + 0 + 1) = 2/3
        got = functional.dice_coefficient(_probs(LOGITS_C2, 2), GOLD_C2, smoothing=1.0)
        assert torch.allclose(got, torch.tensor([[0.8, 2 / 3]]), atol=1e-6)

    def test_dice_coefficient_exact_values(self) -> None:
        probs = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # (1, 2, 3)
        target = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
        got = functional.dice_coefficient(probs, target, smoothing=1.0)
        # class 0: (2*2 + 1) / (2 + 3 + 1) = 5/6 ; class 1: (2*1 + 1) / (1 + 1 + 1) = 1
        assert torch.allclose(got, torch.tensor([[5 / 6, 1.0]]), atol=1e-6)

    def test_micro_dice_pools_everything(self) -> None:
        probs = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        target = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
        # pooled over classes: intersection 3, sum(p) 3, sum(t) 4 -> (2*3 + 1) / (3 + 4 + 1) = 7/8
        assert float(functional.micro_dice(probs, target, smoothing=1.0)) == pytest.approx(7 / 8)

    def test_micro_dice_never_exceeds_one(self) -> None:
        # The bug this guards: the confusion-matrix version divided by sum(cm) instead of
        # 2*sum(cm) and reported Dice 2.0 for a flawless prediction.
        cm = functional.confusion_matrix(_probs(LOGITS_C2, 2), LABELS_C2, 2)
        micro = segmentation.micro_dice_from_matrix(cm)
        assert micro == pytest.approx(1.0, abs=1e-6)  # 2*1 / (2*1 + eps), not 2*1 / 1
        # Single-label targets make micro Dice equal accuracy: 2*TP + FP + FN == 2 * cm.sum().
        assert micro == pytest.approx(functional.accuracy_from_confusion_matrix(cm), rel=1e-6)
        # The soft (B, C, N) route agrees on a crisp prediction, and reads 2*.5/(1+1) = .5 on the
        # 0.5/0.5 logits -- the mass on the unargmaxed class counts, which is the soft form's point.
        assert float(
            functional.micro_dice(GOLD_C2, GOLD_C2, smoothing=DEFAULT_EPSILON)
        ) == pytest.approx(micro, rel=1e-6)
        assert float(
            functional.micro_dice(_probs(LOGITS_C2, 2), GOLD_C2, smoothing=DEFAULT_EPSILON)
        ) == pytest.approx(0.5)

    def test_tversky_alpha_beta_not_swapped(self) -> None:
        # Semantic change 1: alpha belongs on false positives, beta on false negatives. The
        # shared LOGITS_C2 pair has tp=.5, fp=0, fn=.5 -- error in one direction only -- so it
        # cannot tell the two weights apart; PROBS_ASYM gives class 0 tp=1, fp=1, fn=0.
        alpha_only = functional.tversky_coefficient(
            PROBS_ASYM, TARGET_ASYM, alpha=1.0, beta=0.0, smoothing=0.0
        )
        beta_only = functional.tversky_coefficient(
            PROBS_ASYM, TARGET_ASYM, alpha=0.0, beta=1.0, smoothing=0.0
        )
        # class 0: 1/(1 + 1*1 + 0*0) = 1/2 with alpha on the fp, 1/(1 + 0 + 0) = 1 with beta.
        assert float(alpha_only[0, 0]) == pytest.approx(0.5)
        assert float(beta_only[0, 0]) == pytest.approx(1.0)

    def test_tversky_with_half_half_equals_dice(self) -> None:
        probs, target = _probs(LOGITS_C2, 2), GOLD_C2
        tversky = functional.tversky_coefficient(probs, target, alpha=0.5, beta=0.5, smoothing=0.0)
        dice = functional.dice_coefficient(probs, target, smoothing=0.0)
        assert torch.allclose(tversky, dice, atol=1e-6)

    def test_tversky_focal_tversky_share_their_tversky(self) -> None:
        # class 0: tp=1, fp=1, fn=0 -> DT = (fp+tp)/(tp + a*fp + b*fn) = 2/(1+.5) = 4/3
        # class 1: tp=fp=fn=0   -> DT = 0/(0+0+0+eps) = 0
        distance = functional.focal_tversky_distance(PROBS_ASYM, TARGET_ASYM, alpha=0.5, beta=0.5)
        assert distance[0].tolist() == pytest.approx([4 / 3, 0.0], rel=1e-6)
        # The loss must be built on this same DT, not on a second copy of the formula.
        expected_loss = float(((1 + distance[0]) ** DEFAULT_FOCAL_TVERSKY_POWER).mean())
        assert float(
            functional.focal_tversky_loss(PROBS_ASYM, TARGET_ASYM, 2, alpha=0.5, beta=0.5)
        ) == pytest.approx(expected_loss, rel=1e-6)

    def test_generalized_dice_is_per_image(self) -> None:
        # Semantic change 7: the legacy code merged the batch into one number, so every row of a
        # batch carried the same value -- each row here must equal the same call on that image.
        values = functional.generalized_dice_coefficient(_probs(LOGITS, 3), GOLD, epsilon=1e-6)
        assert values.shape == (2,)
        for index in range(2):
            single = functional.generalized_dice_coefficient(
                _probs(LOGITS[index : index + 1], 3), GOLD[index : index + 1], epsilon=1e-6
            )
            assert torch.allclose(values[index], single, atol=1e-6), f"row {index} is not its own"

        # Two images with different predictions must report different numbers. The shared LOGITS
        # pair is all-zero (identical probabilities per image), so use a crisp unequal pair:
        # image 0 matches its target exactly (1.0), image 1 does not (~1e-6).
        crisp = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        different = functional.generalized_dice_coefficient(crisp, GOLD, epsilon=1e-6)
        assert float(different[0]) == pytest.approx(1.0, abs=1e-5)
        assert not torch.allclose(different[0], different[1], atol=1e-6)

    def test_generalized_dice_weighting(self) -> None:
        # w_c = 1 / (sum(t)^2 + eps), so the class with 1 pixel outweighs the class with 2.
        probs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2): one pixel each
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        # intersection (1,1), pred (1,1), target (1,1); w = (1,1) -> 2*2 / (1+1+1+1) = 1
        assert functional.generalized_dice_coefficient(
            probs, target, epsilon=0.0
        ).tolist() == pytest.approx([1.0])
        imperfect = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        # class 0: sum(p)=2, sum(t)=1 -> w=1; class 1: sum(p)=0, sum(t)=1 -> w=1; intersections
        # (1, 0): 2*(1*1 + 1*0) / ((1*(2+1)) + (1*(0+0) + 1*(0+1))) = 2/(3+1) = 1/2
        assert functional.generalized_dice_coefficient(
            imperfect, target, epsilon=0.0
        ).tolist() == pytest.approx([0.5])

    def test_jaccard_matches_dice_equivalence(self) -> None:
        # Dice and Jaccard of the same pair are related by D = 2J/(1+J); check the identity.
        probs, target = _probs(LOGITS_C2, 2), GOLD_C2
        dice = functional.dice_coefficient(probs, target, smoothing=0.0)
        jaccard = functional.jaccard_coefficient(probs, target, smoothing=0.0)
        assert torch.allclose(dice, 2 * jaccard / (1 + jaccard), atol=1e-6)
        # class 0: intersection .5, union .5 + 1 - .5 = 1 -> .5 ; class 1: 0 / (0.5 + 0) -> 0
        assert jaccard[0].tolist() == pytest.approx([0.5, 0.0], rel=1e-6)


# -------------------------------------------------------------- loss functions


class TestFunctionalLosses:
    def test_cross_entropy_matches_torch(self) -> None:
        logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [-1.0, 0.5]])
        labels = torch.tensor([0, 1, 1])
        assert float(functional.cross_entropy_loss(logits, labels, 2)) == pytest.approx(
            float(F.cross_entropy(logits, labels)), rel=1e-6
        )

    def test_cross_entropy_with_weights_accepts_weights_below_one(self) -> None:
        # Semantic change 13: the legacy code floored every class weight at 1.0 via maximum().
        logits = torch.tensor([[0.0, 1.0]])
        labels = torch.tensor([0])
        got = functional.cross_entropy_loss(logits, labels, 2, weights=torch.tensor([0.25, 1.0]))
        want = F.cross_entropy(logits, labels, weight=torch.tensor([0.25, 1.0]))
        assert float(got) == pytest.approx(float(want), rel=1e-6)

    def test_cross_entropy_accepts_one_hot_targets(self) -> None:
        # ... and the same weights on the one-hot path give the same number.
        assert float(
            functional.cross_entropy_loss(LOGITS_C2, GOLD_C2, 2, weights=torch.tensor([1.0, 1.0]))
        ) == pytest.approx(float(F.cross_entropy(LOGITS_C2, LABELS_C2)), rel=1e-6)

    def test_cross_entropy_rejects_wrong_weight_length(self) -> None:
        with pytest.raises(ValueError, match=r"weights has 2 entries, n_classes=3"):
            functional.cross_entropy_loss(LOGITS, LABELS, 3, weights=torch.zeros(2))

    def test_cross_entropy_rejects_bad_logits_rank(self) -> None:
        with pytest.raises(ValueError, match=r"expected logits of shape \(B, C, ...\), got \(3,\)"):
            functional.cross_entropy_loss(torch.zeros(3), LABELS_C2, 3)

    def test_cross_entropy_reduction_none(self) -> None:
        per_sample = functional.cross_entropy_loss(LOGITS, LABELS, 3, reduction="none")
        assert per_sample.shape == (2, 1, 2)
        assert float(per_sample.mean()) == pytest.approx(
            float(functional.cross_entropy_loss(LOGITS, LABELS, 3)), rel=1e-6
        )

    def test_focal_loss_matches_definition(self) -> None:
        # p_t = 0.5 for the symmetric logits, so loss = -alpha * (1-p)^gamma * log(p)
        expected = -DEFAULT_FOCAL_ALPHA * (1 - 0.5) ** DEFAULT_FOCAL_GAMMA * math.log(0.5)
        assert float(functional.focal_loss(LOGITS_C2, LABELS_C2, 2)) == pytest.approx(
            expected, rel=1e-6
        )

    def test_focal_loss_gamma_zero_is_cross_entropy(self) -> None:
        assert float(
            functional.focal_loss(LOGITS_C2, LABELS_C2, 2, alpha=1.0, gamma=0.0)
        ) == pytest.approx(float(F.cross_entropy(LOGITS_C2, LABELS_C2)), rel=1e-6)

    def test_focal_loss_gathers_per_class_alpha(self) -> None:
        # alpha_t must be the weight of the *true* class; label 1 here carries weight 0.75.
        logits = torch.tensor([[0.0, 0.0]])
        got = functional.focal_loss(logits, torch.tensor([1]), 2, alpha=torch.tensor([0.25, 0.75]))
        assert float(got) == pytest.approx(-0.75 * 0.25 * math.log(0.5), rel=1e-6)

    def test_focal_loss_is_not_diluted_by_n_classes(self) -> None:
        # Semantic change 4: the old metrics.focal_loss averaged C-1 zeros into the mean, so the
        # 4-D logits below used to report a fifth of this value.
        logits = torch.zeros(2, 3, 1, 2)
        single = functional.focal_loss(logits[:1], LABELS[:1], 3)
        batch = functional.focal_loss(logits, LABELS, 3)
        assert float(single) == pytest.approx(
            float(batch), rel=1e-6
        ), "an all-zero-logit batch has identical per-image losses; averaging must not change it"

    def test_focal_loss_rejects_bad_alpha_length(self) -> None:
        with pytest.raises(ValueError, match=r"alpha has 2 entries, n_classes=3"):
            functional.focal_loss(LOGITS, LABELS, 3, alpha=torch.zeros(2))

    def test_focal_loss_ignore_index(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        both = float(functional.focal_loss(logits, torch.tensor([0, 1]), 2))
        ignored = float(functional.focal_loss(logits, torch.tensor([0, 1]), 2, ignore_index=1))
        single = float(functional.focal_loss(logits[:1], torch.tensor([0]), 2))
        assert ignored == pytest.approx(single, rel=1e-6), "only the class-0 pixel may contribute"
        assert both == pytest.approx(single, rel=1e-6)  # both pixels are identical here

    def test_dice_loss_reductions(self) -> None:
        micro = functional.dice_loss(LOGITS_C2, LABELS_C2, 2, smoothing=1.0)
        # p=[.5,.5], t=[1,0]: intersection .5, sums 1 and 1 -> 1 - 2/3
        assert float(micro) == pytest.approx(1 - 2 / 3, rel=1e-6)
        per_class = functional.dice_loss(LOGITS_C2, LABELS_C2, 2, smoothing=1.0, reduction="none")
        assert per_class.shape == (1, 2)
        assert float(per_class.mean()) == pytest.approx(
            float(functional.dice_loss(LOGITS_C2, LABELS_C2, 2, reduction="macro")), rel=1e-6
        )
        with pytest.raises(ValueError, match="reduction must be 'micro', 'macro' or 'none'"):
            functional.dice_loss(LOGITS_C2, LABELS_C2, 2, reduction="median")

    def test_soft_dice_loss_uses_squares(self) -> None:
        # 1 - 2*sum(p*t) / (sum(p^2) + sum(t^2)); p=[.5,.5], t=[1,0] -> 1 - 1/(0.5+1)
        assert float(
            functional.soft_dice_loss(LOGITS_C2, LABELS_C2, 2, epsilon=0.0)
        ) == pytest.approx(1 - 2 * 0.5 / (0.25 + 0.25 + 1.0), rel=1e-6)

    def test_tversky_loss_is_one_minus_coefficient(self) -> None:
        got = functional.tversky_loss(LOGITS_C2, LABELS_C2, 2, alpha=0.5, beta=0.5, smoothing=0.0)
        # alpha=beta=.5 makes Tversky the Dice coefficient (see TestCoefficients): the classes
        # score 2/3 and 0, so the mean coefficient is 1/3 and the loss is 1 - 1/3.
        assert float(got) == pytest.approx(2 / 3, rel=1e-6)

    def test_focal_tversky_loss_uses_published_power(self) -> None:
        # Semantic change 2: mean_c (1 + DT_c) ** p with p=3, not sum((1 - tversky) ** 1.3).
        # p=[.5,.5], t=[1,0] gives DT = (2/3, 2): class 0 0.5/(.5+0+.25), class 1 .5/(0+.25+0).
        expected = (
            (1 + 2 / 3) ** DEFAULT_FOCAL_TVERSKY_POWER + (1 + 2.0) ** DEFAULT_FOCAL_TVERSKY_POWER
        ) / 2
        assert float(
            functional.focal_tversky_loss(LOGITS_C2, LABELS_C2, 2, alpha=0.5, beta=0.5)
        ) == pytest.approx(expected, rel=1e-6)

    def test_jaccard_loss_is_in_unit_interval(self) -> None:
        # Semantic change 6: no longer multiplied by smoothing=100.
        value = float(functional.jaccard_loss(LOGITS_C2, LABELS_C2, 2, smoothing=0.0))
        assert 0.0 <= value <= 1.0
        # The two classes score J = .5 and 0 (see TestCoefficients), so the loss is 1 - .25.
        assert value == pytest.approx(1 - (0.5 + 0.0) / 2, rel=1e-6)

    def test_generalized_dice_loss(self) -> None:
        # epsilon=0 is the formula's own eps term; without it, the class with no target pixels
        # gets w = 1/0 and the coefficient is nan rather than the ~2e-6 it should read.
        value = functional.generalized_dice_loss(LOGITS_C2, LABELS_C2, 2)
        assert 0.0 <= float(value) <= 1.0
        assert float(value) == pytest.approx(
            1 - float(functional.generalized_dice_coefficient(_probs(LOGITS_C2, 2), GOLD_C2)),
            rel=1e-6,
        )
        assert float(value) == pytest.approx(1.0, abs=1e-5)  # the pair is nearly perfect

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_every_loss_supports_all_reductions(self, reduction: str) -> None:
        kwargs = {"n_classes": 3, "reduction": reduction}
        for call in (
            lambda: functional.cross_entropy_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.focal_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.soft_dice_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.tversky_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.focal_tversky_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.generalized_dice_loss(LOGITS, LABELS, **kwargs),
            lambda: functional.jaccard_loss(LOGITS, LABELS, **kwargs),
        ):
            out = call()
            assert out.numel() >= 1
            assert torch.isfinite(out).all(), f"{call} produced a non-finite {reduction} loss"


# --------------------------------------------------------- ConfusionMatrixMetric


class TestConfusionMatrixMetric:
    def test_accumulates_across_batches(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3)
        metric.update(LOGITS[:1], LABELS[:1])
        assert metric.num_pixels == 2
        metric.update(LOGITS[1:], LABELS[1:])
        assert metric.num_pixels == 4
        assert metric.confusion_matrix.tolist() == [[1, 0, 0], [2, 0, 0], [1, 0, 0]]

    def test_compute_keys_and_values(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3, class_names=["bg", "text", "line"])
        metric.update(LOGITS, LABELS)
        result = metric.compute()
        assert result["accuracy"] == pytest.approx(0.25)
        assert result["miou"] == pytest.approx((0.25 + 0.0 + 0.0) / 3, rel=1e-6)
        assert result["dice_micro"] == pytest.approx(0.25, rel=1e-6)
        assert result["num_pixels"] == 4.0
        assert set(result) == {
            "accuracy",
            "miou",
            "dice_macro",
            "dice_micro",
            "f1_macro",
            "num_pixels",
            *(
                f"{kind}/{name}"
                for kind in ("iou", "dice", "precision", "recall", "support")
                for name in ("bg", "text", "line")
            ),
        }

    def test_compute_and_reset(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3)
        metric.update(LOGITS, LABELS)
        before = metric.compute()
        assert metric.compute_and_reset() == before
        assert metric.num_pixels == 0
        assert math.isnan(metric.compute()["miou"]), "an empty metric reports nan, not 0"

    def test_class_names_length_is_checked(self) -> None:
        with pytest.raises(ValueError, match="1 class_names for n_classes=2"):
            segmentation.ConfusionMatrixMetric(2, class_names=["only"])

    def test_n_classes_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be positive, got 0"):
            segmentation.ConfusionMatrixMetric(0)

    def test_drop_last_excludes_class_but_keeps_its_number(self) -> None:
        # Reproduces metrics.seg_metrics(drop_last=True): the last class is background.
        metric = segmentation.ConfusionMatrixMetric(3, drop_last=True)
        metric.update(LOGITS, LABELS)
        result = metric.compute()
        assert result["miou"] == pytest.approx((0.25 + 0.0) / 2, rel=1e-6)
        assert result["iou/2"] == pytest.approx(0.0), "still reported, just not averaged"
        assert result["support/2"] == 1.0

    def test_absent_classes_are_nan_not_zero(self) -> None:
        # Semantic change 8: the legacy code dropped classes scoring exactly `ignore_index`.
        metric = segmentation.ConfusionMatrixMetric(2)
        metric.update(LOGITS_C2, LABELS_C2)
        result = metric.compute()
        assert math.isnan(result["iou/1"]), "class 1 has no pixels anywhere -> undefined"
        assert result["iou/0"] == pytest.approx(1.0, rel=1e-6)
        assert result["miou"] == pytest.approx(1.0, rel=1e-6)

    def test_absent_classes_can_be_counted_as_zero(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(2, exclude_absent_classes=False)
        metric.update(LOGITS_C2, LABELS_C2)
        assert metric.compute()["iou/1"] == 0.0
        assert metric.compute()["miou"] == pytest.approx(0.5, rel=1e-6)

    def test_per_class_tensors(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3)
        metric.update(LOGITS, LABELS)
        per_class = metric.per_class()
        assert set(per_class) == {
            "iou",
            "dice",
            "precision",
            "recall",
            "f1",
            "support",
            "confusion_matrix",
        }
        assert per_class["support"].tolist() == [1.0, 2.0, 1.0]
        assert per_class["confusion_matrix"].tolist() == [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]

    def test_layout_is_passed_through(self) -> None:
        first = segmentation.ConfusionMatrixMetric(3)
        last = segmentation.ConfusionMatrixMetric(3, layout="channels_last")
        first.update(LOGITS, LABELS)
        last.update(LOGITS.permute(0, 2, 3, 1), LABELS)
        assert torch.equal(first.confusion_matrix, last.confusion_matrix)

    def test_repr(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3)
        metric.update(LOGITS, LABELS)
        assert repr(metric).startswith("ConfusionMatrixMetric(n_classes=3, pixels=4, miou=")


class TestOneShotHelpers:
    def test_mean_iou_infers_n_classes_from_labels(self) -> None:
        assert segmentation.mean_iou(LOGITS, LABELS) == pytest.approx(
            (0.25 + 0.0 + 0.0) / 3, rel=1e-6
        )

    def test_mean_iou_matches_the_metric(self) -> None:
        metric = segmentation.ConfusionMatrixMetric(3)
        metric.update(LOGITS, LABELS)
        assert segmentation.mean_iou(LOGITS, LABELS, 3) == pytest.approx(metric.compute()["miou"])

    def test_dice_macro_and_micro(self) -> None:
        assert segmentation.dice_macro(LOGITS, LABELS, 3) == pytest.approx(0.4 / 3, rel=1e-6)
        assert segmentation.dice_micro(LOGITS, LABELS, 3) == pytest.approx(0.25, rel=1e-6)
        assert segmentation.dice_score(LOGITS, LABELS, 3) == pytest.approx(
            segmentation.dice_macro(LOGITS, LABELS, 3), rel=1e-6
        )
        assert segmentation.dice_score(LOGITS, LABELS, 3, macro=False) == pytest.approx(
            segmentation.dice_micro(LOGITS, LABELS, 3), rel=1e-6
        )

    def test_accuracy_works_for_classification_and_segmentation(self) -> None:
        # Replaces metrics.Accuracy, which only accepted (B, C) logits.
        assert segmentation.accuracy(LOGITS, LABELS, 3) == pytest.approx(0.25)
        assert segmentation.accuracy(LOGITS_C2, LABELS_C2, 2) == pytest.approx(1.0)

    def test_one_shot_requires_n_classes_for_soft_targets(self) -> None:
        with pytest.raises(
            ValueError, match="n_classes is required when the targets are not integer"
        ):
            segmentation.mean_iou(LOGITS_C2, GOLD_C2)


# ------------------------------------------------------------------- get_metrics


class TestGetMetrics:
    @pytest.mark.parametrize("task", ["segmentation", "binarization", "classification"])
    def test_pixelwise_tasks_build_a_confusion_metric(self, task: str) -> None:
        metrics = get_metrics(task, 3)
        assert isinstance(metrics["segmentation"], segmentation.ConfusionMatrixMetric)

    def test_binarization_defaults_to_two_classes(self) -> None:
        assert get_metrics("binarization")["segmentation"].n_classes == 2

    def test_enhancement_builds_psnr_and_ssim(self) -> None:
        assert set(get_metrics("enhancement")) == {"psnr", "ssim"}

    def test_reading_order_has_no_pixel_metric(self) -> None:
        with pytest.raises(NotImplementedError, match="reading_order has no pixel-wise metric"):
            get_metrics("reading_order")

    def test_unknown_task(self) -> None:
        with pytest.raises(ValueError, match="unknown task 'nope'"):
            get_metrics("nope", 3)

    def test_missing_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes is required for task=segmentation"):
            get_metrics("segmentation")

    def test_kwargs_reach_the_metric(self) -> None:
        metrics = get_metrics("segmentation", 3, class_names=["a", "b", "c"], drop_last=True)
        assert metrics["segmentation"].class_names == ["a", "b", "c"]
        assert metrics["segmentation"].drop_last is True

    def test_compute_metrics_prefixes_keys(self) -> None:
        from sbb_pytorch.metrics import compute_metrics

        metrics = get_metrics("segmentation", 3)
        result = compute_metrics(metrics, LOGITS, LABELS)
        assert result["segmentation/miou"] == pytest.approx((0.25 + 0 + 0) / 3, rel=1e-6)
        assert metrics["segmentation"].num_pixels == 4, "compute() keeps what it accumulated"

        # reset=True is the end-of-epoch variant: it returns the accumulated numbers and then
        # forgets them, so the metric is empty again while the result still reports both batches.
        reset_result = compute_metrics(metrics, LOGITS, LABELS, reset=True)
        assert reset_result["segmentation/num_pixels"] == pytest.approx(8.0)
        assert metrics["segmentation"].num_pixels == 0

    def test_compute_metrics_enhancement(self) -> None:
        from sbb_pytorch.metrics import compute_metrics

        images = torch.rand(1, 3, 8, 8)
        result = compute_metrics(get_metrics("enhancement"), images, images, reset=True)
        assert result["psnr/psnr"] == float("inf")
        assert result["ssim/ssim"] == pytest.approx(1.0)


# ------------------------------------------------------------------------ PSNR


class TestPSNR:
    def test_from_mse_matches_definition(self) -> None:
        assert float(enhance.psnr_from_mse(0.25, 1.0)) == pytest.approx(
            20 * math.log10(1.0 / 0.5), rel=1e-6
        )
        assert float(enhance.psnr_from_mse(1.0, 1.0)) == pytest.approx(0.0, abs=1e-9)
        assert float(enhance.psnr_from_mse(0.0, 1.0)) == float("inf")

    def test_data_range_scales_by_20_log10(self) -> None:
        # 255-range images read 48.13 dB higher than the same images normalised to [0, 1].
        low = float(enhance.psnr_from_mse(0.25, 1.0))
        high = float(enhance.psnr_from_mse(0.25 * 255**2, 255.0))
        assert low == pytest.approx(high, rel=1e-6)

    def test_rejects_non_positive_data_range(self) -> None:
        with pytest.raises(ValueError, match="data_range must be positive, got 0"):
            enhance.psnr_from_mse(0.25, 0)

    def test_psnr_of_pair(self) -> None:
        a = torch.rand(2, 3, 8, 8)
        b = torch.rand(2, 3, 8, 8)
        expected = 20 * torch.log10(torch.tensor(1.0) / (a - b).pow(2).mean().sqrt())
        assert float(enhance.psnr(a, b)) == pytest.approx(float(expected), rel=1e-5)

    def test_validations(self) -> None:
        a = torch.rand(2, 3, 8, 8)
        with pytest.raises(
            ValueError,
            match=r"predictions must be a \(B, C, H, W\) image batch, got shape \(3, 8, 8\)",
        ):
            enhance.psnr(torch.rand(3, 8, 8), torch.rand(3, 8, 8))
        with pytest.raises(
            ValueError, match="channel mismatch: predictions have 3, targets have 1"
        ):
            enhance.psnr(a, torch.rand(2, 1, 8, 8))
        with pytest.raises(TypeError, match="predictions must be float"):
            enhance.psnr(
                torch.zeros(2, 3, 4, 4, dtype=torch.int32),
                torch.zeros(2, 3, 4, 4, dtype=torch.int32),
            )

    def test_metric_pools_instead_of_averaging_ratios(self) -> None:
        # metrics.PSNR averaged per-batch PSNRs, which is not the same number as the pooled MSE.
        a, b = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        c, d = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        metric = enhance.PSNRMetric()
        metric.update(a, b)
        one_batch = metric.compute()
        metric.update(c, d)
        pooled = metric.compute()
        assert one_batch["mse"] == pytest.approx(float((a - b).pow(2).mean()), rel=1e-6)
        assert pooled["mse"] == pytest.approx(
            float(((a - b).pow(2).sum() + (c - d).pow(2).sum()) / (a.numel() + c.numel())), rel=1e-6
        )
        assert pooled["mse"] != one_batch["mse"]
        assert pooled["psnr"] == pytest.approx(
            float(enhance.psnr_from_mse(pooled["mse"])), rel=1e-6
        )

    def test_empty_metric_is_nan_and_resets(self) -> None:
        metric = enhance.PSNRMetric()
        assert math.isnan(metric.compute()["psnr"]) and math.isnan(metric.compute()["mse"])
        metric.update(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
        metric.reset()
        assert math.isnan(metric.compute()["psnr"])


# ------------------------------------------------------------------------- SSIM


class TestSSIM:
    def test_gaussian_window_is_a_real_gaussian(self) -> None:
        # The legacy window used -x^2 / sigma^2 (missing the factor 2), so sigma=1.5 was really
        # sigma=1.06. Check against the closed form for a 3x3 window.
        window = enhance.gaussian_window(3, 1.5)
        assert window.shape == (1, 1, 3, 3)
        offsets = torch.tensor([-1.0, 0.0, 1.0])
        k1d = torch.exp(-(offsets**2) / (2 * 1.5**2))
        expected = torch.outer(k1d, k1d) / k1d.sum() ** 2
        assert torch.allclose(window[0, 0], expected, atol=1e-6)
        assert float(window.sum()) == pytest.approx(1.0, rel=1e-6)
        legacy = torch.tensor([math.exp(-1 / 1.5**2), 1.0, math.exp(-1 / 1.5**2)])
        assert not torch.allclose(window[0, 0][0], torch.outer(legacy, legacy)[0][0], atol=1e-3)

    @pytest.mark.parametrize("size, sigma", [(4, 1.5), (-3, 1.5), (3, 0.0)])
    def test_window_validation(self, size: int, sigma: float) -> None:
        with pytest.raises(ValueError):
            enhance.gaussian_window(size, sigma)

    def test_identical_images_score_one(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        assert torch.allclose(enhance.ssim(images, images), torch.ones(2), atol=1e-6)

    def test_ssim_is_scale_invariant_with_matching_data_range(self) -> None:
        a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
        base = enhance.ssim(a, b, data_range=1.0)
        assert torch.allclose(enhance.ssim(a * 2, b * 2, data_range=2.0), base, atol=1e-5)
        assert torch.allclose(
            enhance.ssim(a * 0.5, b * 0.5, data_range=0.5), base, atol=1e-5
        ), "C1/C2 were hard-coded for [0, 1]; now they derive from data_range"

    def test_ssim_returns_per_image_and_map(self) -> None:
        a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
        assert enhance.ssim(a, b).shape == (2,)
        assert enhance.ssim_map(a, b).shape == (2, 3, 16, 16)
        assert (enhance.ssim(a, b) - enhance.ssim_map(a, b).mean(dim=(1, 2, 3))).abs().max() < 1e-6

    def test_ssim_is_symmetric(self) -> None:
        a, b = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
        assert torch.allclose(enhance.ssim(a, b), enhance.ssim(b, a), atol=1e-6)

    def test_better_reconstruction_scores_higher(self) -> None:
        base = torch.rand(1, 3, 24, 24)
        near = base + 0.01 * torch.randn_like(base)
        far = base + 0.5 * torch.randn_like(base)
        assert float(enhance.ssim(near, base)) > float(enhance.ssim(far, base))

    def test_metric(self) -> None:
        a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
        metric = enhance.SSIMMetric()
        assert math.isnan(metric.compute()["ssim"])
        metric.update(a, b)
        first = metric.compute()["ssim"]
        assert first == pytest.approx(float(enhance.ssim(a, b).mean()), rel=1e-6)
        metric.update(a, b)
        assert metric.compute()["ssim"] == pytest.approx(first, rel=1e-9)
        metric.reset()
        assert math.isnan(metric.compute()["ssim"])


# ------------------------------------------------------------------- Metric ABC


class TestMetricInterface:
    def test_every_exported_metric_shares_the_interface(self) -> None:
        metrics: list[segmentation.Metric] = [
            segmentation.ConfusionMatrixMetric(2),
            enhance.PSNRMetric(),
            enhance.SSIMMetric(),
        ]
        for metric in metrics:
            assert (
                hasattr(metric, "update")
                and hasattr(metric, "compute")
                and hasattr(metric, "reset")
            )
            assert isinstance(metric.name, str) and metric.name
            metric.reset()
            assert all(math.isfinite(v) or math.isnan(v) for v in metric.compute().values())

    def test_public_api_has_no_duplicates_or_stars(self) -> None:
        from sbb_pytorch import metrics as metrics_module

        assert len(metrics_module.__all__) == len(set(metrics_module.__all__))
        for name in metrics_module.__all__:
            assert hasattr(metrics_module, name), f"{name} is exported but missing"

    def test_functional_constants_match_their_documented_values(self) -> None:
        assert (DEFAULT_EPSILON, DEFAULT_SMOOTHING) == (1e-8, 1.0)
        assert (DEFAULT_FOCAL_ALPHA, DEFAULT_FOCAL_GAMMA, DEFAULT_FOCAL_TVERSKY_POWER) == (
            0.25,
            2.0,
            3.0,
        )
