"""Encoder-decoder-segmentation-head assembly, and the two ResNet50 U-Nets.

``models_pytorch.py`` declared ``ResNet50UNet`` and ``ResNet50UNetLight`` as two ~90-line
classes differing only in their decoder channel list; each rebuilt the encoder from scratch and
each carried its own (partly broken) decoder. Here both are three lines over a single
:class:`UNet`, which composes :class:`~sbb_pytorch.models.base.ResNetEncoder`,
:class:`~sbb_pytorch.models.base.UNetDecoder` and a
:class:`~sbb_pytorch.models.base.SegmentationHead`.

Behaviour preserved: the ``n_classes`` / ``input_height`` / ``input_width`` constructor
signature, ImageNet-pretrained ResNet50 by default, and a ``(B, n_classes, H, W)`` output at the
input's own resolution.

Behaviour changed (see ``docs/migration.md``):

* ``weight_decay`` is gone from the signature -- it was accepted and ignored.
* ``ResNet50UNetLight`` used to return ``softmax`` and ``ResNet50UNet`` raw logits for the same
  task; both now return logits (``output_activation="none"``), because ``CrossEntropyLoss``
  applies ``log_softmax`` itself and feeding it softmaxed values normalises twice.
* The head is :class:`~sbb_pytorch.models.base.SegmentationHead`: it fuses the input image at
  full resolution through a 32-channel convolution before predicting -- the stage the port
  dropped -- and normalises the logits on segmentation tasks, which is what the upstream head
  does. The old ``head.weight`` is now ``head.classify.weight``.
* ``pairing="legacy"`` is gone. It existed to load ``models_pytorch.ResNet50UNet`` weights, and
  there are none, so the off-by-one decoder it built was a second shape standing where the
  correct one should have been.
* ``LIGHT_DECODER_CHANNELS`` is ``(512, 512, 256, 128, 64)``: the five widths the upstream light
  decoder actually has. The old ``(512, 256, 128, 64, 32)`` had merged the head's
  full-resolution fusion conv into the last decoder block and halved everything below the first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn as nn

from sbb_pytorch.config.models import TaskType
from sbb_pytorch.models.base import ResNetEncoder, SegmentationHead, UNetDecoder, probs

__all__ = ["ResNet50UNet", "ResNet50UNetLight", "UNet"]

#: ``(1024, 512, 256, 128, 64)`` -- the full model's decoder widths, deepest level first.
FULL_DECODER_CHANNELS: tuple[int, ...] = (1024, 512, 256, 128, 64)

#: ``(512, 512, 256, 128, 64)`` -- the light model's, deepest first: a 1x1 projection of f5 to
#: 512 and then the same halving as the full model. Its 32-channel full-resolution conv is the
#: head's, not a decoder block, so it is not in this list.
LIGHT_DECODER_CHANNELS: tuple[int, ...] = (512, 512, 256, 128, 64)

#: Upstream's light decoder projects one more skip before fusing it: ``f4`` goes through
#: ``Conv2D(512, (1,1)) + BN + ReLU`` (``training/models.py:256-260``), so the block above it sees
#: ``512 + 512`` channels rather than the ``512 + 1024`` an unprojected ``f4`` would hand over.
#: Without this the light model repeats, in miniature, the width error that stopped the legacy
#: ``ResNet50UNetLight`` from running at all.
LIGHT_SKIP_PROJECTIONS: Mapping[int, int] = {3: 512}

Activation = Literal["none", "softmax", "sigmoid"]


class UNet(nn.Module):
    """Segmentation model: ResNet encoder + shared decoder + segmentation head.

    Args:
        n_classes: Number of output channels (classes).
        input_height: Height the model was configured for. Used for reporting; the output is
            sized to the *actual* input height so a differently-sized page still lines up.
        input_width: Width the model was configured for.
        backbone: Any ResNet in :data:`sbb_pytorch.models.base.RESNET_LEVELS` (``"resnet50"``
            covers both legacy spellings ``RESNET`` and ``NONTRANSFORMER``).
        pretrained: Load ImageNet weights; ``False`` builds the same graph with no network access.
        decoder_channels: Decoder widths, deepest first (see
            :class:`sbb_pytorch.models.base.UNetDecoder`).
        attention: Add CBAM attention to each decoder block (the enhancement model's shape).
        skip_projections: Skips the decoder projects with a 1x1 before fusing them, passed
            straight to :class:`sbb_pytorch.models.base.UNetDecoder`; only the light model has
            one (``LIGHT_SKIP_PROJECTIONS``).
        output_activation: ``"none"`` (default) returns logits; ``"softmax"`` and ``"sigmoid"``
            return probabilities for callers that were trained expecting them.
        task: Decides whether the head normalises its logits (upstream: only for segmentation)
            and is what :meth:`predict_proba` uses to pick softmax or sigmoid.
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
        attention: bool = False,
        skip_projections: Mapping[int, int] | None = None,
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
            self.encoder.channels,
            decoder_channels,
            attention=attention,
            skip_projections=skip_projections,
        )
        # Upstream normalises the logits exactly when the task is segmentation; that rule is the
        # head's only per-task behaviour, so it is read from ``task`` rather than repeated at
        # every call site.
        self.head = SegmentationHead(
            self.decoder.output_channels,
            self.n_classes,
            logits_norm=task is TaskType.SEGMENTATION,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` in, ``(B, n_classes, H, W)`` out -- same H, W as the input."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) batch, got shape {tuple(x.shape)}")
        return self.activate(self.head(self.decoder(self.encoder(x)), x))

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply ``output_activation`` to raw logits.

        Split out of ``forward`` because the ViT variant that patchifies the input assembles the
        encoder, decoder and head by hand -- its head consumes a *different* image than its encoder
        does -- and still has to honour the same activation rule as every other model.
        """
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

    The only legacy variant whose forward pass ran -- its decoder paired skips off-by-one
    (``[f5, f3, f2, f1, f1]``), which :class:`UNetDecoder` now builds the upstream way.
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        pretrained: bool = True,
        task: TaskType | None = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(
            n_classes,
            input_height,
            input_width,
            backbone="resnet50",
            pretrained=pretrained,
            decoder_channels=FULL_DECODER_CHANNELS,
            task=task,
        )


class ResNet50UNetLight(UNet):
    """The light ResNet50 U-Net (``light_model=True``): same encoder, halved decoder widths.

    Its own decoder could not run -- at block 0 it concatenated 2048 + 1024 channels and fed
    them to a ``Conv2d(512, ...)`` -- which :class:`UNetDecoder` resolves by computing every
    block's real input width. The widths are now the upstream ones (``512, 512, 256, 128, 64``);
    the ``512, 256, 128, 64, 32`` this class declared had shifted the upstream head's
    32-channel conv into the last decoder block and halved everything below the first.

    Upstream dodges the very concatenation the legacy class attempted by projecting ``f4`` to 512
    first (``training/models.py:256-260``), so this class passes ``LIGHT_SKIP_PROJECTIONS`` -- the
    same width mistake, one level shallower, now fixed where upstream fixes it.
    """

    def __init__(
        self,
        n_classes: int,
        input_height: int = 224,
        input_width: int = 224,
        *,
        pretrained: bool = True,
        task: TaskType | None = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(
            n_classes,
            input_height,
            input_width,
            backbone="resnet50",
            pretrained=pretrained,
            decoder_channels=LIGHT_DECODER_CHANNELS,
            skip_projections=LIGHT_SKIP_PROJECTIONS,
            task=task,
        )
