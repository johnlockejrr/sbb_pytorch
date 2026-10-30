"""Shared blocks: one correct encoder, decoder and block set for every architecture.

``models_pytorch.py`` carried **five near-copies** of the same U-Net decoder -- one per model:
``ResNet50UNet``, ``ResNet50UNetLight``, ``ViTResNet50UNet``,
``ViTResNet50UNetTransformerBeforeCNN`` and ``EnhancementModel``. Four of the five could not
run a forward pass, because the copied ``forward`` did this::

    for i, (decoder_layer, upsample) in enumerate(zip(self.decoder, self.upsample)):
        x = upsample(x)
        if i < len(features) - 1:
            x = torch.cat([x, features[-(i + 2)]], dim=1)
        x = decoder_layer(x)

At ``i = 0`` that concatenates the upsampled 2048-channel ``f5`` with ``f4`` (1024 channels) and
feeds **3072** channels into a ``Conv2d(2048, 1024)`` -- ``RuntimeError`` on the first forward
pass. The one variant that ran (``ResNet50UNet``) worked only because it special-cased
``if i == 0``, and its pairing is still off by one: the decoder blocks consume
``[f5, f3, f2, f1, f1]``, so ``f4`` is never used as a skip and ``f1`` is used twice.

:class:`UNetDecoder` here is the single implementation the whole package uses. It pairs each
decoder block with the encoder feature map of the *same* level (the
`segmentation-models <https://github.com/qubvel-org/segmentation_models.pytorch>`_ convention),
sizes every fusion by ``F.interpolate`` to the skip's actual shape rather than assuming a clean
pyramid (levels 0 and 1 of a torchvision ResNet are both at ``H/4``, so
``Upsample(scale_factor=2)`` is wrong for one of them), and exposes ``pairing="legacy"`` so
checkpoints trained against ``ResNet50UNet``'s off-by-one decoder still load.

Three more cross-cutting fixes live here:

* ``weight_decay`` was an argument of every model constructor and was never used -- it is an
  optimiser setting and now lives in ``training/factory.py``.
* The head dropped the upstream ``Conv2D(32, 3x3)`` that fuses the input image at full
  resolution and only then predicts a class map, so the shipped models were an interpolation of
  an ``H/4`` prediction rather than a prediction. :class:`SegmentationHead` rebuilds that stage
  (plus the segmentation-only ``BatchNormalization`` on the logits) for ``pairing="aligned"``.
* The legacy models disagreed about activations: ``ResNet50UNet`` returned raw logits for
  segmentation and ``sigmoid`` otherwise, while ``ResNet50UNetLight`` and both ViT variants
  returned ``softmax``. A ``CrossEntropyLoss`` on softmaxed logits normalises twice, so those
  four models trained on a flattened loss surface. All models now return **raw logits**; use
  :func:`sbb_pytorch.models.base.probs` (or ``F.softmax``) where probabilities are wanted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from sbb_pytorch.config.models import TaskType

__all__ = [
    "RESNET_LEVELS",
    "RESNET_STRIDES",
    "AttentionBlock",
    "ConvBlock",
    "ResNetEncoder",
    "SegmentationHead",
    "UNetDecoder",
    "probs",
    "resize_to_input",
    "resolve_head",
]

Pairing = Literal["aligned", "legacy"]

#: ``{variant: (channels per level, shallowest -> deepest)}`` for the supported ResNets.
#: Level 0 is the stem (``conv1 -> bn -> relu -> maxpool``) and level 1 the first residual
#: stage; both are 4x smaller than the input because ``maxpool`` lives inside the stem and
#: ``layer1`` does not downsample, which is why the pyramid is not a clean halving per level.
RESNET_LEVELS: dict[str, tuple[int, ...]] = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "resnet101": (64, 256, 512, 1024, 2048),
}

#: Downsampling factor from the model input to each level's output, same order as the values in
#: :data:`RESNET_LEVELS`: the stem halves twice (``conv1`` stride 2 + ``maxpool`` stride 2) and
#: only ``layer2``-``layer4`` halve again, so levels 0 and 1 share ``H/4`` and the deepest is
#: ``H/32``. Every ResNet variant shares this pyramid.
#:
#: This constant exists because ``models/vit.py`` used to compute the deepest map as
#: ``height // 2**stage`` -- ``H/16``, a factor off for a *stride-2 stem*, so a model built for
#: a 448x448 page sized its position embeddings for a 28x28 grid while the encoder actually
#: produced 14x14. Any code that needs a feature-map size asks here (or the encoder's own
#: :attr:`ResNetEncoder.strides`) instead of re-deriving it.
RESNET_STRIDES: tuple[int, ...] = (4, 4, 8, 16, 32)


def probs(logits: torch.Tensor, task_binary: bool = False) -> torch.Tensor:
    """Turn model logits into probabilities.

    Args:
        logits: ``(B, C, H, W)`` raw scores.
        task_binary: Use ``sigmoid`` for the two-channel binary heads (binarization,
            enhancement), ``softmax`` otherwise.
    """
    return torch.sigmoid(logits) if task_binary else F.softmax(logits, dim=1)


def resize_to_input(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize ``(B, C, H, W)`` to the spatial ``size`` the model was given.

    Every segmentation head ends with this. The encoder downsamples by 32 and the decoder
    re-upsamples by interpolation, so without it the logits land on a rounded size and the
    predicted mask does not align with the label image -- the class of bug that showed up as
    "448 input, 224 grid" in the shipped configs.
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ConvBlock(nn.Sequential):
    """``Conv2d -> BatchNorm2d -> ReLU``, the single conv unit every decoder block uses.

    ``padding=1`` with ``kernel_size=3`` keeps spatial size, which is what makes the decoder's
    sizes predictable; the legacy code's ``Conv2d(1536, 512)``-style blocks relied on the same
    invariant without ever asserting it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.add_module(
            "conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.ReLU(inplace=True))


class AttentionBlock(nn.Module):
    """CBAM-style channel + spatial attention (ported from ``models_pytorch.AttentionBlock``).

    The maths was already correct; it is relocated here so the enhancement model can request it
    through :class:`UNetDecoder` instead of carrying its own decoder copy.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        spatial = self.spatial(
            torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        )
        return x * spatial


def resolve_head(
    image_fusion: bool | None = None,
    pairing: Pairing = "aligned",
    logits_norm: bool | None = None,
    task: TaskType | None = None,
) -> tuple[bool, bool]:
    """The two head rules -- ``(image_fusion, logits_norm)`` -- in the one place they are stated.

    An unset ``image_fusion`` follows the pairing: only the aligned head exists upstream, so a
    legacy head stays the bare ``Conv2d`` its checkpoint was written against. An unset
    ``logits_norm`` follows upstream's own rule, ``task == "segmentation"``, but only on a fused
    head -- and asking for a norm on an unfused one is refused rather than ignored, because it
    adds parameters the checkpoint that pairing exists to load cannot supply.

    Raises:
        ValueError: ``image_fusion`` on a pairing that has no fusion weights, or ``logits_norm``
            without ``image_fusion``.
    """
    fused = pairing == "aligned" if image_fusion is None else bool(image_fusion)
    if fused and pairing != "aligned":
        raise ValueError(
            f"image_fusion needs pairing='aligned', got {pairing!r}: that pairing reproduces the "
            "class the checkpoint came from, and its head has no fusion weights to fill"
        )
    normalise = (
        fused and task is TaskType.SEGMENTATION if logits_norm is None else bool(logits_norm)
    )
    if normalise and not fused:
        raise ValueError(
            "logits_norm needs image_fusion: the unfused head is the bare Conv2d its checkpoints "
            "were written against, and a BatchNorm2d has no weights there"
        )
    return fused, normalise


class SegmentationHead(nn.Module):
    """The output stage every segmentation model shares: optional image fusion, then the 1x1.

    ``image_fusion=True`` is the Keras tail -- ``unet_decoder`` in the upstream
    ``training/models.py:294-306``, the same six lines on every architecture they ship:
    upsample the decoder output to the image, ``concatenate([o, img])``,
    ``Conv2D(32, 3x3) -> BatchNormalization -> ReLU``, then ``Conv2D(n_classes, 1x1)``. The
    0.1.0 port dropped that stage and resized the ``H/4`` logits to the input instead, so the
    class conv never looked at a single input pixel -- a one-pixel textline became a blur, and
    the "prediction" at full resolution was an interpolation of a prediction at a quarter of it.

    ``logits_norm`` is the ``BatchNormalization`` upstream puts on the logits when
    ``task == "segmentation"`` and omits on the sigmoid tasks, which is why it is a flag rather
    than part of the convolution.

    Args:
        in_channels: Channels the decoder hands over.
        n_classes: Head width.
        image_fusion: Fuse the input image at full resolution before the class conv. Needs a
            3-channel input, and it is what a ``pairing="aligned"`` model must build to be the
            upstream network at all.
        fusion_channels: Width of the fusion conv; upstream uses ``32`` on every model.
        logits_norm: Apply ``BatchNorm2d(n_classes)`` to the logits.

    Raises:
        ValueError: if a channel count is not positive, or fusion was asked for on a model whose
            input is not 3 channels.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        *,
        image_fusion: bool = False,
        fusion_channels: int = 32,
        logits_norm: bool = False,
    ) -> None:
        super().__init__()
        if min(in_channels, n_classes, fusion_channels) < 1:
            raise ValueError(
                f"channel counts must be positive, got in={in_channels} classes={n_classes} "
                f"fusion={fusion_channels}"
            )
        self.n_classes = int(n_classes)
        self.image_fusion = image_fusion
        if image_fusion:
            self.fusion: ConvBlock | None = ConvBlock(in_channels + 3, int(fusion_channels))
            head_in = int(fusion_channels)
        else:
            self.fusion = None
            head_in = int(in_channels)
        self.classify = nn.Conv2d(head_in, self.n_classes, kernel_size=1)
        self.norm = nn.BatchNorm2d(self.n_classes) if logits_norm else None

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """``features`` is the decoder output, ``image`` the model input it must line up with."""
        size = (image.shape[-2], image.shape[-1])
        if self.fusion is None:
            logits = resize_to_input(self.classify(features), size)
        else:
            if image.shape[1] != 3:
                raise ValueError(
                    f"{type(self).__name__} fuses the input image into the head, so it needs 3 "
                    f"input channels, got {image.shape[1]}"
                )
            fused = torch.cat((resize_to_input(features, size), image), dim=1)
            logits = self.classify(self.fusion(fused))
        return logits if self.norm is None else self.norm(logits)


class ResNetEncoder(nn.Module):
    """A torchvision ResNet used as a 5-level feature extractor.

    The legacy code built this identically in five places and always from
    ``models.resnet50(weights=IMAGENET1K_V1)``, which forced a weight download even when a
    project config asked for ``"nontransformer"``. ``pretrained=False`` builds the same
    architecture with PyTorch's default initialisation and no network access.

    Args:
        variant: Any torchvision ResNet named in :data:`RESNET_LEVELS`.
        pretrained: Load ImageNet weights (``resnet50`` -> ``IMAGENET1K_V1``).
    """

    def __init__(self, variant: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        if variant not in RESNET_LEVELS:
            raise ValueError(
                f"unsupported backbone {variant!r}; choose from {sorted(RESNET_LEVELS)}"
            )
        self.variant = variant
        self.channels = RESNET_LEVELS[variant]
        self.strides = RESNET_STRIDES

        builder = getattr(tv_models, variant)
        weights = None
        if pretrained:
            try:
                weights = getattr(tv_models, f"{variant.upper()}_Weights").IMAGENET1K_V1
            except AttributeError:  # pragma: no cover - variant without a V1 weight enum
                weights = None
        # ``builder`` is the universal callable in torchvision >= 0.13; the legacy code used the
        # per-class constructors (``models.resnet50(...)``), which are deprecated aliases of it.
        resnet = builder(weights=weights)

        # Level 0 is the stem; levels 1-4 are the residual stages.
        stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.levels = nn.ModuleList(
            [stem, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return one feature map per level, shallowest first (``features[-1]`` is the deepest)."""
        features: list[torch.Tensor] = []
        for level in self.levels:
            x = level(x)
            features.append(x)
        return features


class UNetDecoder(nn.Module):
    """The decoder shared by every architecture in the package.

    Args:
        encoder_channels: Channels per encoder level, **shallowest first**, e.g.
            ``(64, 256, 512, 1024, 2048)`` for ResNet50.
        decoder_channels: Output channels per decoder block, **deepest first** -- the order the
            legacy lists were written in, e.g. ``(1024, 512, 256, 128, 64)`` for the full model
            and ``(512, 256, 128, 64, 32)`` for the light one. Length must match
            ``encoder_channels``.
        pairing: Which encoder feature each block fuses.

            * ``"aligned"`` (default) -- the block at level *j* fuses the deeper decoder output
              with ``encoder[j]``; this is the correct pairing.
            * ``"legacy"`` -- reproduces ``models_pytorch.ResNet50UNet``'s off-by-one
              (``[f5, f3, f2, f1, f1]``, so ``f4`` is never used and ``f1`` twice), which
              reproduces that file's exact ``Conv2d`` shapes
              (``2048->1024, 1536->512, 768->256, 320->128, 192->64``) so its weights load.
        attention: Add an :class:`AttentionBlock` to each block (the enhancement model did this).

    Raises:
        ValueError: if the two channel lists differ in length or a channel count is not positive.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 256, 512, 1024, 2048),
        decoder_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        pairing: Pairing = "aligned",
        attention: bool = False,
    ) -> None:
        super().__init__()
        enc = tuple(int(c) for c in encoder_channels)
        dec = tuple(int(c) for c in decoder_channels)
        if not enc or any(c <= 0 for c in enc):
            raise ValueError(f"encoder_channels must be positive integers, got {enc}")
        if len(enc) != len(dec):
            raise ValueError(
                f"decoder_channels {dec} must have one entry per encoder level ({len(enc)})"
            )
        if any(c <= 0 for c in dec):
            raise ValueError(f"decoder_channels must be positive integers, got {dec}")
        if pairing not in ("aligned", "legacy"):
            raise ValueError(f"pairing must be 'aligned' or 'legacy', got {pairing!r}")

        self.encoder_channels = enc
        self.decoder_channels = dec
        self.pairing: Pairing = pairing
        self.skip_indices = self._skip_indices(len(enc), pairing)

        # Channel count of every block input, and therefore the exact Conv2d it needs.
        # The deepest block is a 1x1 projection (a projection, not a convolution over space);
        # every other block is 3x3 -- the same pattern ``models_pytorch`` used, and the one that
        # keeps ``state_dict`` shapes compatible with its checkpoints.
        blocks = nn.ModuleList()
        previous = enc[-1]  # the deepest level feeds the deepest block
        for position, (skip_index, out_channels) in enumerate(
            zip(self.skip_indices, dec, strict=False)
        ):
            in_channels = enc[skip_index] if position == 0 else previous + enc[skip_index]
            block: list[nn.Module] = [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1 if position == 0 else 3,
                    # The legacy convs all carried a bias even in front of a BatchNorm (where the
                    # convention is none); keeping it for the legacy pairing is what lets a
                    # checkpoint trained on models_pytorch load with strict=True.
                    bias=pairing == "legacy",
                )
            ]
            if attention:
                block.append(AttentionBlock(out_channels))
            blocks.append(nn.Sequential(*block))
            previous = out_channels

        self.blocks = blocks

    @staticmethod
    def _skip_indices(levels: int, pairing: Pairing) -> list[int]:
        """Encoder level index each decoder block fuses, deepest block first."""
        if pairing == "aligned":
            return list(reversed(range(levels)))
        # models_pytorch.ResNet50UNet: f5 (as the input), then f3, f2, f1, then f1 again.
        if levels == 1:
            return [0]
        return [levels - 1, *reversed(range(levels - 2)), 0]

    @property
    def output_channels(self) -> int:
        """Channels produced by the final block -- what the head consumes.

        With ``decoder_channels`` ordered deepest-first, that is the last entry: ``64`` for the
        full model and ``32`` for the light one, matching the legacy
        ``final_conv = nn.Conv2d(64, n_classes, 1)`` / ``nn.Conv2d(32, n_classes, 1)``.
        """
        return self.decoder_channels[-1]

    def forward(self, features: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Fuse ``features`` (one per encoder level, shallowest first) into one dense map.

        A bare tensor is accepted for single-level encoders; it is treated as the deepest -- and
        only -- level.
        """
        if isinstance(features, torch.Tensor):
            features = [features]
        if len(features) != len(self.encoder_channels):
            raise ValueError(
                f"expected {len(self.encoder_channels)} encoder feature maps, got {len(features)}"
            )
        for index, (feature, expected) in enumerate(
            zip(features, self.encoder_channels, strict=False)
        ):
            if feature.shape[1] != expected:
                raise ValueError(
                    f"feature map {index} has {feature.shape[1]} channels, encoder declares "
                    f"{expected} -- the encoder and decoder were built from different configs"
                )

        x = features[-1]  # the deepest map enters the deepest block directly
        for block, skip_index in zip(self.blocks, self.skip_indices, strict=False):
            skip = features[skip_index]
            if skip is not x:
                # Sized to the skip rather than doubled: in a ResNet, f1 and f2 share a
                # resolution, so a fixed scale_factor=2 is wrong for one of them.
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
                x = torch.cat((x, skip), dim=1)
            x = block(x)
        return x
