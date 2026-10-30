"""Whole-image heads: the classifier, the reading-order model, and the enhancement model.

All three take a ResNet, pool it to ``(B, feature_dim)`` and attach an MLP; none of them used
``input_height``/``input_width`` for anything (adaptive pooling makes any size work) but all
three accepted the arguments, and the legacy versions then baked an activation into the head --
``Softmax`` in the classifier, ``Sigmoid`` in the reading-order model -- which is the same
double-normalisation problem as the U-Nets: a ``CrossEntropyLoss`` applied on top of a softmaxed
head computes ``log(softmax(softmax(z)))``. These heads now emit logits and expose the
activation separately.

``EnhancementModel`` keeps its own sigmoid output, because its loss and its metrics (PSNR/SSIM)
are defined on the reconstructed image itself rather than on logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from sbb_pytorch.config.models import TaskType
from sbb_pytorch.exceptions import ModelBuildError
from sbb_pytorch.models.base import RESNET_LEVELS
from sbb_pytorch.models.unet import UNet

__all__ = ["EnhancementModel", "MachineBasedReadingOrderModel", "ResNet50Classifier"]


def resnet_features(backbone: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """``(feature_extractor, feature_dim)`` for a torchvision ResNet without its ``fc`` layer.

    Shared by the classifier and the reading-order model, which previously each rebuilt
    ``nn.Sequential(*list(resnet50.children())[:-1])`` themselves.
    """
    if backbone not in RESNET_LEVELS:
        raise ModelBuildError(
            f"unsupported backbone {backbone!r}; choose from {sorted(RESNET_LEVELS)}"
        )
    builder = getattr(tv_models, backbone)
    weights = None
    if pretrained:
        try:
            weights = getattr(tv_models, f"{backbone.upper()}_Weights").IMAGENET1K_V1
        except AttributeError:  # pragma: no cover
            weights = None
    network = builder(weights=weights)
    return nn.Sequential(*list(network.children())[:-1]), RESNET_LEVELS[backbone][-1]


class _MLPHead(nn.Module):
    """``Linear -> ReLU -> Dropout -> Linear``, the hidden layer both classification heads share."""

    def __init__(
        self, feature_dim: int, hidden: int, out_features: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResNet50Classifier(nn.Module):
    """Whole-image classification (``TaskType.CLASSIFICATION``).

    Args:
        n_classes: Number of classes.
        input_height, input_width: Recorded for reporting; adaptive pooling accepts any size.
        backbone: torchvision ResNet name.
        pretrained: Load ImageNet weights.
        hidden_units: Legacy head used a single 256-unit layer; a longer MLP is accepted too.
        dropout: Dropout rate between hidden layers (legacy used 0.2).
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_units: tuple[int, ...] = (256,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        self.n_classes = int(n_classes)
        self.input_height, self.input_width = int(input_height), int(input_width)
        self.features, feature_dim = resnet_features(backbone, pretrained)
        self.head = _MLPHead(feature_dim, hidden_units[0], self.n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, n_classes)`` logits."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) batch, got shape {tuple(x.shape)}")
        return self.head(self.features(x).flatten(1))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmaxed class probabilities (the legacy head returned these from ``forward``)."""
        return torch.softmax(self.forward(x), dim=1)


class MachineBasedReadingOrderModel(nn.Module):
    """Reading-order prediction over a page crop (``TaskType.READING_ORDER``).

    The legacy head ended in ``Sigmoid`` while being trained with a cross-entropy loss; like the
    classifier it now returns logits, and ``probs`` distinguishes single-label (softmax) from
    multi-label (sigmoid) use.
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_units: tuple[int, ...] = (256, 256),
        dropout: float = 0.2,
        multi_label: bool = True,
    ) -> None:
        super().__init__()
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        self.n_classes = int(n_classes)
        self.input_height, self.input_width = int(input_height), int(input_width)
        self.multi_label = multi_label
        self.features, feature_dim = resnet_features(backbone, pretrained)
        self.head = _MLPHead(feature_dim, hidden_units[0], self.n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, n_classes)`` logits."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) batch, got shape {tuple(x.shape)}")
        return self.head(self.features(x).flatten(1))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, dim=1)


class EnhancementModel(nn.Module):
    """RGB enhancement / binarisation reconstruction (``TaskType.ENHANCEMENT``).

    Composed from :class:`~sbb_pytorch.models.unet.UNet` with attention in the decoder -- the
    shape the legacy class hand-rolled (its own decoder copy had the same concatenate-width bug
    as the other broken variants, so it never ran). Its head is fixed at three output channels
    by the ``task=enhancement requires n_classes=3`` config rule.
    """

    def __init__(
        self,
        input_height: int = 224,
        input_width: int = 224,
        *,
        n_classes: int = 3,
        backbone: str = "resnet50",
        pretrained: bool = True,
        image_fusion: bool | None = None,
        fusion_channels: int = 32,
        logits_norm: bool | None = None,
    ) -> None:
        super().__init__()
        if n_classes != 3:
            raise ModelBuildError(
                f"enhancement reconstructs an RGB image, so n_classes must be 3, got {n_classes}"
            )
        self.input_height, self.input_width = int(input_height), int(input_width)
        self.n_classes = int(n_classes)
        self.unet = UNet(
            n_classes,
            input_height,
            input_width,
            backbone=backbone,
            pretrained=pretrained,
            attention=True,
            image_fusion=image_fusion,
            fusion_channels=fusion_channels,
            logits_norm=logits_norm,
            output_activation="sigmoid",
            task=TaskType.ENHANCEMENT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` in, ``(B, 3, H, W)`` enhanced image in ``[0, 1]`` out.

        ``UNet.forward`` already resizes its output to the input's spatial size, so nothing
        further is needed here -- the legacy class ended in ``Conv2d(64, 3, 1)`` + ``Sigmoid``
        and never checked that the result matched the page it started from.
        """
        return self.unet(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Every registry entry answers ``predict_proba``; here it is the sigmoided image itself.

        ``forward`` already applies the activation, so re-normalising would be the double-softmax
        mistake the other heads had -- these are per-channel intensities in ``[0, 1]``, not a
        class distribution.
        """
        return self.forward(x)
