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
`segmentation-models <https://github.com/qubvel-org/segmentation_models.pytorch>`_ convention,
and the pairing the upstream model uses), sized by ``F.interpolate`` to the skip's actual shape
instead of a fixed ``Upsample(scale_factor=2)`` -- exact for any input size, including the ones
that are not a clean power of two. The port's second shape, ``pairing="legacy"``, is gone: it
reproduced the off-by-one decoder only so its checkpoints could be loaded, and there are none.

Three more cross-cutting fixes live here:

* ``weight_decay`` was an argument of every model constructor and was never used -- it is an
  optimiser setting and now lives in ``training/factory.py``.
* The head dropped the upstream ``Conv2D(32, 3x3)`` that fuses the input image at full
  resolution and only then predicts a class map, so the shipped models were an interpolation of
  a downsampled prediction rather than a prediction. :class:`SegmentationHead` rebuilds that
  stage, plus the segmentation-only ``BatchNormalization`` on the logits, on every model.
* The legacy models disagreed about activations: ``ResNet50UNet`` returned raw logits for
  segmentation and ``sigmoid`` otherwise, while ``ResNet50UNetLight`` and both ViT variants
  returned ``softmax``. A ``CrossEntropyLoss`` on softmaxed logits normalises twice, so those
  four models trained on a flattened loss surface. All models now return **raw logits**; use
  :func:`sbb_pytorch.models.base.probs` (or ``F.softmax``) where probabilities are wanted.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

__all__ = [
    "RESNET_LEVELS",
    "RESNET_STRIDES",
    "AttentionBlock",
    "ConvBlock",
    "FUSION_CHANNELS",
    "ResNetEncoder",
    "SegmentationHead",
    "UNetDecoder",
    "probs",
    "resize_to_input",
]

#: The width of the head's fusion conv -- ``32`` on every upstream model, full and light alike.
FUSION_CHANNELS: int = 32

#: ``{variant: (channels per level, shallowest -> deepest)}`` for the supported ResNets.
#: Level 0 is ``conv1 -> bn -> relu``, level 1 ``maxpool -> layer1`` and levels 2-4 the remaining
#: residual stages -- the tap points the upstream model uses, so ``layer1``'s parameters live at
#: ``levels.1.1.*`` rather than the ``levels.1.*`` a plain torchvision stem would give them.
RESNET_LEVELS: dict[str, tuple[int, ...]] = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "resnet101": (64, 256, 512, 1024, 2048),
}

#: Downsampling factor from the model input to each level's output, same order as the values in
#: :data:`RESNET_LEVELS`: ``2, 4, 8, 16, 32``, one halving per level, because level 0 is
#: ``conv1`` alone (stride 2) and the ``maxpool`` (stride 2) that starts level 1 does the second.
#: Every ResNet variant shares this pyramid.
#:
#: This constant exists because ``models/vit.py`` used to compute the deepest map as
#: ``height // 2**stage`` -- ``H/16``, a factor off for a *stride-2 stem*, so a model built for
#: a 448x448 page sized its position embeddings for a 28x28 grid while the encoder actually
#: produced 14x14. Any code that needs a feature-map size asks here (or the encoder's own
#: :attr:`ResNetEncoder.strides`) instead of re-deriving it.
RESNET_STRIDES: tuple[int, ...] = (2, 4, 8, 16, 32)


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


class SegmentationHead(nn.Module):
    """The upstream output stage: fuse the image at full resolution, then predict with a 1x1.

    This is ``unet_decoder``'s tail in the upstream ``training/models.py:294-306``, the same six
    lines on every architecture they ship, ``resnet50_unet`` and ``resnet50_unet_light`` alike::

        o = UpSampling2D((2, 2), interpolation="bilinear")(o)   # to the image's own size
        o = concatenate([o, img])                                # the input pixels themselves
        o = Conv2D(32, (3, 3), padding="valid") -> BN -> ReLU    # FUSION_CHANNELS
        o = Conv2D(n_classes, (1, 1))                            # then BN if task == segmentation

    The port had only the ``Conv2D(n_classes, 1x1)``, run on the decoder's map and resized
    afterwards, so a "full resolution" class map was an interpolation of a prediction that had
    never looked at an input pixel -- a one-pixel textline became a blur.

    Args:
        in_channels: Channels the decoder hands over; the head consumes ``in_channels + 3`` once
            the image is concatenated.
        n_classes: Head width.
        logits_norm: ``BatchNorm2d(n_classes)`` on the logits -- upstream's
            ``if task == "segmentation"``, which the caller resolves from the task.

    Raises:
        ValueError: if ``in_channels`` or ``n_classes`` is not positive, or the input given to
            ``forward`` is not 3 channels.
    """

    def __init__(self, in_channels: int, n_classes: int, *, logits_norm: bool = False) -> None:
        super().__init__()
        if in_channels < 1 or n_classes < 1:
            raise ValueError(
                f"channel counts must be positive, got in={in_channels} classes={n_classes}"
            )
        self.n_classes = int(n_classes)
        self.fusion = ConvBlock(in_channels + 3, FUSION_CHANNELS)
        self.classify = nn.Conv2d(FUSION_CHANNELS, self.n_classes, kernel_size=1)
        self.norm = nn.BatchNorm2d(self.n_classes) if logits_norm else None

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """``features`` is the decoder output; ``image`` is the input whose size it matches.

        Raises:
            ValueError: the head concatenates the input onto the features, so it needs a
                3-channel batch.
        """
        if image.shape[1] != 3:
            raise ValueError(
                f"SegmentationHead fuses the input image into the head, so it needs 3 input "
                f"channels, got {image.shape[1]}"
            )
        upsampled = resize_to_input(features, (image.shape[-2], image.shape[-1]))
        logits = self.classify(self.fusion(torch.cat((upsampled, image), dim=1)))
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

        # The levels follow the upstream tap points rather than torchvision's grouping. Upstream
        # takes ``f1`` from ``conv1 -> bn -> relu`` at ``H/2``, *before* the pool -- the only way
        # the decoder's last block can sit at ``H/2`` and upsample exactly once into the
        # full-resolution head -- and ``f2`` from ``maxpool -> layer1`` at ``H/4``. torchvision
        # keeps the pool inside the stem, so this split moves layer1's parameters one level
        # deeper (``levels.1.*`` -> ``levels.1.1.*``).
        self.levels = nn.ModuleList(
            [
                nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
                nn.Sequential(resnet.maxpool, resnet.layer1),
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ]
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
            upstream lists are written in, ``(1024, 512, 256, 128, 64)`` for the full model and
            ``(512, 512, 256, 128, 64)`` for the light one. Length must match ``encoder_channels``.
        attention: Add an :class:`AttentionBlock` to each block (the enhancement model did this).

    Raises:
        ValueError: if the two channel lists differ in length or a channel count is not positive.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 256, 512, 1024, 2048),
        decoder_channels: Sequence[int] = (1024, 512, 256, 128, 64),
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

        self.encoder_channels = enc
        self.decoder_channels = dec
        # Block *i* of the deepest-first decoder fuses encoder level ``len(enc) - 1 - i`` -- the
        # pairing upstream uses, and the only one here now. The port's ``pairing="legacy"``
        # alternative (``[f5, f3, f2, f1, f1]``, ``f4`` never used, ``f1`` twice) existed to load
        # weights that were never trained, so the model built it as a second shape rather than
        # fixing the first.
        self.skip_indices = list(reversed(range(len(enc))))

        # Channel count of every block input, and therefore the exact Conv2d it needs.
        # The deepest block is a 1x1 projection (a projection, not a convolution over space);
        # every other block is 3x3, and every conv in front of a BatchNorm is unbiased.
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
                )
            ]
            if attention:
                block.append(AttentionBlock(out_channels))
            blocks.append(nn.Sequential(*block))
            previous = out_channels

        self.blocks = blocks

    @property
    def output_channels(self) -> int:
        """Channels produced by the final block -- what the head consumes.

        With ``decoder_channels`` ordered deepest-first, that is the last entry: ``64`` on both the
        full and the light model, which is what the upstream head is built to accept.
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
