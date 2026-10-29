"""Encoder-decoder-segmentation-head assembly, and the two ResNet50 U-Nets.

``models_pytorch.py`` declared ``ResNet50UNet`` and ``ResNet50UNetLight`` as two ~90-line
classes differing only in their decoder channel list; each rebuilt the encoder from scratch and
each carried its own (partly broken) decoder. Here both are three lines over a single
:class:`UNet`, which composes :class:`~sbb_pytorch.models.base.ResNetEncoder`,
:class:`~sbb_pytorch.models.base.UNetDecoder` and a 1x1 head.

Behaviour preserved: the ``n_classes`` / ``input_height`` / ``input_width`` constructor
signature, ImageNet-pretrained ResNet50 by default, and a ``(B, n_classes, H, W)`` output at the
input's own resolution.

Behaviour changed (see ``docs/migration.md``):

* ``weight_decay`` is gone from the signature -- it was accepted and ignored.
* ``ResNet50UNetLight`` used to return ``softmax`` and ``ResNet50UNet`` raw logits for the same
  task; both now return logits (``output_activation="none"``), because ``CrossEntropyLoss``
  applies ``log_softmax`` itself and feeding it softmaxed values normalises twice.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from sbb_pytorch.config.models import TaskType
from sbb_pytorch.models.base import ResNetEncoder, UNetDecoder, probs, resize_to_input

__all__ = ["ResNet50UNet", "ResNet50UNetLight", "UNet"]

#: ``(1024, 512, 256, 128, 64)`` -- the full model's decoder widths, deepest level first.
FULL_DECODER_CHANNELS: tuple[int, ...] = (1024, 512, 256, 128, 64)

#: ``(512, 256, 128, 64, 32)`` -- the light model's, as its ``# Reduced from ...`` comments said.
LIGHT_DECODER_CHANNELS: tuple[int, ...] = (512, 256, 128, 64, 32)

Activation = Literal["none", "softmax", "sigmoid"]


class UNet(nn.Module):
    """Segmentation model: ResNet encoder + shared decoder + 1x1 head.

    Args:
        n_classes: Number of output channels (classes).
        input_height: Height the model was configured for. Used for reporting; the output is
            resized to the *actual* input height so a differently-sized page still lines up.
        input_width: Width the model was configured for.
        backbone: Any ResNet in :data:`sbb_pytorch.models.base.RESNET_LEVELS` (``"resnet50"``
            covers both legacy spellings ``RESNET`` and ``NONTRANSFORMER``).
        pretrained: Load ImageNet weights; ``False`` builds the same graph with no network access.
        decoder_channels: Decoder widths, deepest first (see
            :class:`sbb_pytorch.models.base.UNetDecoder`).
        pairing: ``"aligned"`` for the correct skip pairing, ``"legacy"`` to reproduce
            ``models_pytorch.ResNet50UNet`` exactly so its checkpoints load.
        attention: Add CBAM attention to each decoder block (the enhancement model's shape).
        output_activation: ``"none"`` (default) returns logits; ``"softmax"`` and ``"sigmoid"``
            return probabilities for callers that were trained expecting them.
        task: Recorded so :meth:`predict_proba` knows whether to use softmax or sigmoid.
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        backbone: str = "resnet50",
        pretrained: bool = True,
        decoder_channels: tuple[int, ...] = FULL_DECODER_CHANNELS,
        pairing: Literal["aligned", "legacy"] = "aligned",
        attention: bool = False,
        output_activation: Activation = "none",
        task: TaskType | None = None,
    ) -> None:
        super().__init__()
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        if input_height < 1 or input_width < 1:
            raise ValueError(f"input size must be positive, got {input_height}x{input_width}")

        self.n_classes = int(n_classes)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.task = task
        self.output_activation: Activation = output_activation

        self.encoder = ResNetEncoder(backbone, pretrained=pretrained)
        self.decoder = UNetDecoder(
            self.encoder.channels, decoder_channels, pairing=pairing, attention=attention
        )
        self.head = nn.Conv2d(self.decoder.output_channels, self.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` in, ``(B, n_classes, H, W)`` out -- same H, W as the input."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) batch, got shape {tuple(x.shape)}")
        logits = self.head(self.decoder(self.encoder(x)))
        logits = resize_to_input(logits, (x.shape[-2], x.shape[-1]))
        if self.output_activation == "softmax":
            return probs(logits, task_binary=False)
        if self.output_activation == "sigmoid":
            return probs(logits, task_binary=True)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilities regardless of ``output_activation`` -- what inference and metrics use.

        A model that already activates in ``forward`` is not re-normalised: applying softmax to
        logits that were already softmaxed (the legacy light/ViT behaviour) is the bug this
        method exists to avoid.
        """
        output = self.forward(x)
        if self.output_activation != "none":
            return output
        return probs(output, task_binary=self.task in (TaskType.BINARIZATION, TaskType.ENHANCEMENT))


class ResNet50UNet(UNet):
    """The full ResNet50 U-Net (``BackboneType.RESNET`` / ``NONTRANSFORMER``, ``light_model=False``).

    The only legacy variant whose forward pass ran; its decoder pairs skips off-by-one
    (``[f5, f3, f2, f1, f1]``), so loading its weights needs ``pairing="legacy"``.
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        pretrained: bool = True,
        pairing: Literal["aligned", "legacy"] = "aligned",
        task: TaskType | None = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(
            n_classes,
            input_height,
            input_width,
            backbone="resnet50",
            pretrained=pretrained,
            decoder_channels=FULL_DECODER_CHANNELS,
            pairing=pairing,
            task=task,
        )


class ResNet50UNetLight(UNet):
    """The light ResNet50 U-Net (``light_model=True``): same encoder, halved decoder widths.

    Its own decoder could not run -- at block 0 it concatenated 2048 + 1024 channels and fed
    them to a ``Conv2d(512, ...)`` -- which :class:`UNetDecoder` resolves by computing every
    block's real input width (``2048->512, 1536->256, 768->128, 384->64, 128->32``).
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        pretrained: bool = True,
        pairing: Literal["aligned", "legacy"] = "aligned",
        task: TaskType | None = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(
            n_classes,
            input_height,
            input_width,
            backbone="resnet50",
            pretrained=pretrained,
            decoder_channels=LIGHT_DECODER_CHANNELS,
            pairing=pairing,
            task=task,
        )
