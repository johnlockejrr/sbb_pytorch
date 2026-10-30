"""Patch embedding, transformer tower, and the two ViT-ResNet50 U-Net variants.

Both legacy ViT models raised ``RuntimeError`` on their first forward pass, and neither could
be fixed by editing the decoder alone:

``ViTResNet50UNet`` (``cnn_first=True``)
    ``Patches`` was applied to the ResNet's ``f5`` output (2048 channels at ``H/16``) and the
    result fed to ``PatchEncoder``, whose ``nn.Linear(projection_dim, projection_dim)`` accepted
    only ``projection_dim`` (=64) features while a patch holds ``C * ph * pw`` (=2048 * 4 * 4 =
    32768). The projection must map *from the patch dimension*, which is what it now does. It
    also assumed ``f5`` was evenly divisible by the patch size -- true only when the input is a
    multiple of 64, so a 448 or 224 page raised from ``unfold``.

``ViTResNet50UNetTransformerBeforeCNN`` (``cnn_first=False``)
    Patchified the *input* (3 channels), ran the tower, then reshaped the tokens with
    ``view(B, H//py, W//px, -1).permute(0, 3, 1, 2)`` into a ``(B, 3*ph*pw, H//ph, W//pw)``
    tensor and handed its output straight to ``ResNet50UNet`` -- whose encoder stage expects
    **3** channels and got 48. The reshape was also wrong in the general case: it treated the
    token axis as the row axis. The fix is a real inverse of patchify (:func:`unpatch`), which
    is tested to round-trip, so the CNN receives a genuine ``(B, 3, H, W)`` image.

Both variants also took ``num_patches`` from the config and built ``nn.Embedding(num_patches,
projection_dim)`` with it. ``config_params.json`` declares ``[56, 56]`` = 3136 patches for a
448x448 input with 4x4 patches, which actually yields ``[112, 112]`` = 12544 -- so the position
embedding size disagreed with the tokens and the add broadcast-raised. The number is now derived
from the geometry and a supplied value is *checked*, so a wrong config fails at build time with
both numbers in the message instead of at forward time with ``shape`` tuples.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sbb_pytorch.config.models import TaskType
from sbb_pytorch.exceptions import ModelBuildError
from sbb_pytorch.models.base import RESNET_STRIDES, probs
from sbb_pytorch.models.unet import UNet

__all__ = [
    "Patches",
    "PatchEncoder",
    "PatchDecoder",
    "TransformerBlock",
    "TransformerTower",
    "ViTResNet50UNet",
    "ViTResNet50UNetTransformerBeforeCNN",
    "patchify",
    "unpatch",
    "expected_patch_count",
]


def expected_patch_count(height: int, width: int, patch_height: int, patch_width: int) -> int:
    """Number of ``patch_height x patch_width`` patches in a ``height x width`` map.

    Raises:
        ModelBuildError: if either dimension is not evenly divisible -- the legacy code let
            ``Tensor.unfold`` raise an opaque ``RuntimeError`` instead of naming the numbers.
    """
    if patch_height < 1 or patch_width < 1:
        raise ModelBuildError(f"patch size must be positive, got {patch_height}x{patch_width}")
    if height % patch_height or width % patch_width:
        raise ModelBuildError(
            f"cannot lay {patch_height}x{patch_width} patches over a {height}x{width} map: "
            f"{height} % {patch_height} = {height % patch_height}, "
            f"{width} % {patch_width} = {width % patch_width}"
        )
    return (height // patch_height) * (width // patch_width)


def patchify(x: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    """``(B, C, H, W)`` -> ``(B, n_patches, C * patch_height * patch_width)``.

    The row-major ordering produced here is the exact inverse of :func:`unpatch`; the legacy
    ``Patches.forward`` produced the same layout but its consumers reassembled it with a plain
    ``view`` and got a transposed image.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
    _, channels, height, width = x.shape
    count = expected_patch_count(height, width, patch_height, patch_width)
    unfolded = x.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    return unfolded.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, count, channels * patch_height * patch_width
    )


def unpatch(
    tokens: torch.Tensor, height: int, width: int, patch_height: int, patch_width: int
) -> torch.Tensor:
    """``(B, n_patches, C * ph * pw)`` -> ``(B, C, height, width)`` -- the inverse of :func:`patchify`."""
    batch, count, features = tokens.shape
    expected = expected_patch_count(height, width, patch_height, patch_width)
    if count != expected:
        raise ValueError(
            f"{count} tokens cannot rebuild a {height}x{width} map ({expected} expected)"
        )
    channels = features // (patch_height * patch_width)
    if channels * patch_height * patch_width != features:
        raise ValueError(f"{features} features is not a multiple of {patch_height}x{patch_width}")
    rows, cols = height // patch_height, width // patch_width
    ungrouped = tokens.view(batch, rows, cols, channels, patch_height, patch_width)
    return ungrouped.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


class Patches(nn.Module):
    """Legacy name for :func:`patchify`, kept so existing call sites read the same.

    Unlike the legacy version it also validates the geometry and stores it, which is what makes
    :func:`unpatch` and the position-embedding size checkable.
    """

    def __init__(self, patch_size_x: int, patch_size_y: int) -> None:
        super().__init__()
        if patch_size_x < 1 or patch_size_y < 1:
            raise ModelBuildError(f"patch size must be positive, got {patch_size_x}x{patch_size_y}")
        self.patch_size_x = int(patch_size_x)
        self.patch_size_y = int(patch_size_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return patchify(x, self.patch_size_y, self.patch_size_x)


class PatchEncoder(nn.Module):
    """Project patches to ``projection_dim`` and add positional embeddings.

    Args:
        feature_dim: Per-patch feature count the input actually carries -- ``C * ph * pw``. The
            legacy version used ``projection_dim`` here, so any patch with more than
            ``projection_dim`` features (always) failed to multiply.
        projection_dim: Token width after the projection.
        num_patches: Token count the embeddings are built for; see
            :func:`expected_patch_count`.
    """

    def __init__(self, feature_dim: int, projection_dim: int, num_patches: int) -> None:
        super().__init__()
        if feature_dim < 1 or projection_dim < 1 or num_patches < 1:
            raise ModelBuildError(
                f"feature_dim/projection_dim/num_patches must be positive, got "
                f"{feature_dim}/{projection_dim}/{num_patches}"
            )
        self.feature_dim = int(feature_dim)
        self.projection_dim = int(projection_dim)
        self.num_patches = int(num_patches)
        self.projection = nn.Linear(self.feature_dim, self.projection_dim)
        self.position_embedding = nn.Embedding(self.num_patches, self.projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"patches carry {x.shape[-1]} features but this encoder projects "
                f"{self.feature_dim} -- the encoder was built for a different input geometry"
            )
        encoded = self.projection(x)
        return encoded + self._positions(x.shape[1], encoded.dtype)

    def _positions(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        """Position embeddings for ``count`` tokens, interpolating if the input size differs.

        A fixed ``nn.Embedding`` ties the model to one input resolution. Interpolating the
        embedding matrix (the ``timm`` approach) lets one trained model take any size while
        leaving the parameter -- and therefore the checkpoint -- unchanged.
        """
        table = self.position_embedding.weight.to(dtype)
        if count == self.num_patches:
            return table
        return F.interpolate(
            table.unsqueeze(0), size=count, mode="linear", align_corners=False
        ).squeeze(0)


class PatchDecoder(nn.Module):
    """Project tokens back to per-patch features (the inverse of :meth:`PatchEncoder.projection`)."""

    def __init__(self, projection_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(projection_dim, feature_dim)
        self.feature_dim = int(feature_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (identical maths to the legacy one; relocated for reuse).

    ``mlp_units`` is the block's own MLP -- one ``Linear -> GELU -> Dropout`` per entry, the
    legacy ``mlp(x, hidden_units, dropout_rate)`` loop (``training/models.py:109-113``) that
    ``transformer_block`` ran with the config's ``transformer_mlp_head_units``. The two ViT
    variants used to accept ``mlp_head_units``, store it on ``self`` and build something else.

    Its output is added back onto the attention output, so the chain always ends at ``dim``:
    upstream got that for free because its last unit *was* ``projection_dim``, which left any
    config that shrinks one without shrinking the other (every test config here shrinks
    ``projection_dim`` to 8 and keeps ``[128, 64]``) unbuildable in Keras and in this port
    alike. The declared widths are therefore the *hidden* layers and ``dim`` closes the chain --
    with the legacy ``[128, 64]`` at ``projection_dim=64`` that is exactly upstream's two
    ``Dense`` layers. ``mlp_ratio`` stays as the fallback for callers that pass no units.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mlp_units: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ModelBuildError(f"dim {dim} is not divisible by num_heads {num_heads}")
        declared = (
            (int(dim * mlp_ratio),) if mlp_units is None else tuple(int(unit) for unit in mlp_units)
        )
        if not declared or min(declared) < 1:
            raise ModelBuildError(f"mlp units {declared} must all be positive")
        units = declared if declared[-1] == dim else (*declared, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_units = units
        layers: list[nn.Module] = []
        in_features = dim
        for width in units:
            layers.extend([nn.Linear(in_features, width), nn.GELU(), nn.Dropout(dropout)])
            in_features = width
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TransformerTower(nn.Module):
    """``layers`` stacked :class:`TransformerBlock`s -- the piece both ViT variants duplicated."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        layers: int = 8,
        mlp_ratio: float = 4.0,
        mlp_units: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ModelBuildError(f"transformer layers must be >= 1, got {layers}")
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, mlp_ratio, mlp_units=mlp_units)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _ViTBackbone(nn.Module):
    """Shared body of the two ViT variants: patchify -> tower -> project back -> unpatch.

    Subclasses differ only in *what* they patchify (the ResNet's deepest feature map, or the
    input image), which is exactly the ``cnn_first`` switch the config exposes.
    """

    def __init__(
        self,
        feature_dim: int,
        patch_size_x: int,
        patch_size_y: int,
        num_patches: int,
        projection_dim: int,
        transformer_layers: int,
        num_heads: int,
        mlp_units: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.patch_encoder = PatchEncoder(feature_dim, projection_dim, num_patches)
        self.tower = TransformerTower(
            projection_dim, num_heads, transformer_layers, mlp_units=mlp_units
        )
        self.patch_decoder = PatchDecoder(projection_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_decoder(self.tower(self.patch_encoder(x)))


class ViTResNet50UNet(_ViTBackbone):
    """CNN first, transformer on the deepest feature map, then the shared U-Net decoder.

    The transformer refines ``f5`` in place: the patched-and-attended features are projected back
    to ``f5``'s own width and reassembled into ``(B, 2048, H/32, W/32)``, which is what the
    decoder expects as its deepest input. The legacy version produced a
    ``(B, C*ph*pw, ...)`` tensor and let ``UNetDecoder`` multiply it into a mismatched ``Conv2d``.
    """

    def __init__(
        self,
        n_classes: int,
        patch_size_x: int = 4,
        patch_size_y: int = 4,
        num_patches: int | None = None,
        mlp_head_units: tuple[int, ...] = (128, 64),
        transformer_layers: int = 8,
        num_heads: int = 4,
        projection_dim: int = 64,
        input_height: int = 224,
        input_width: int = 224,
        task: TaskType | None = TaskType.SEGMENTATION,
        *,
        pretrained: bool = True,
    ) -> None:
        from sbb_pytorch.models.base import (
            RESNET_LEVELS,
            ResNetEncoder,
            SegmentationHead,
            UNetDecoder,
        )

        encoder = ResNetEncoder("resnet50", pretrained=pretrained)
        deep_channels = RESNET_LEVELS["resnet50"][-1]
        deep_h, deep_w = _stage_spatial(input_height, input_width, stage=4)
        declared = _check_declared(num_patches, deep_h, deep_w, patch_size_y, patch_size_x)
        self.deep_shape = (deep_h, deep_w)

        _ViTBackbone.__init__(
            self,
            deep_channels * patch_size_y * patch_size_x,
            patch_size_x,
            patch_size_y,
            declared,
            projection_dim,
            transformer_layers,
            num_heads,
            mlp_units=tuple(mlp_head_units),
        )
        self.patch_size_x, self.patch_size_y = patch_size_x, patch_size_y
        self.mlp_head_units = tuple(mlp_head_units)
        self.encoder = encoder
        self.decoder = UNetDecoder(encoder.channels, (1024, 512, 256, 128, 64))
        self.head = SegmentationHead(
            self.decoder.output_channels,
            n_classes,
            logits_norm=task is TaskType.SEGMENTATION,
        )
        self.n_classes = n_classes
        self.input_height, self.input_width = input_height, input_width
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # ``PatchEncoder`` consumes *tokens*, so f5 is patchified first and the attended tokens
        # are reassembled into f5's own shape. Handing it the raw ``(B, 2048, h, w)`` map -- what
        # this line did before -- made ``nn.Linear`` see ``w`` features instead of ``C*ph*pw`` and
        # raise the same class of error the legacy model raised.
        tokens = patchify(features[-1], self.patch_size_y, self.patch_size_x)
        refined = unpatch(
            _ViTBackbone.forward(self, tokens),
            self.deep_shape[0],
            self.deep_shape[1],
            self.patch_size_y,
            self.patch_size_x,
        )
        features[-1] = refined  # the transformer output replaces f5 as the decoder's deepest input
        # The head concatenates ``x`` onto the decoder's map, so the class map is computed at the
        # input's own resolution from the input's own pixels.
        return self.head(self.decoder(features), x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilities for the metrics and inference, from this class's own ``forward``.

        Every registry entry answers ``predict_proba``; the two ViT classes assemble their heads by
        hand rather than through ``UNet``, so each spells the rule out here instead of inheriting it.
        """
        return probs(
            self.forward(x),
            task_binary=self.task in (TaskType.BINARIZATION, TaskType.ENHANCEMENT),
        )


class ViTResNet50UNetTransformerBeforeCNN(nn.Module):
    """Transformer on the input image first, then the ResNet U-Net (``cnn_first=False``)."""

    def __init__(
        self,
        n_classes: int,
        patch_size_x: int = 4,
        patch_size_y: int = 4,
        num_patches: int | None = None,
        mlp_head_units: tuple[int, ...] = (128, 64),
        transformer_layers: int = 8,
        num_heads: int = 4,
        projection_dim: int = 64,
        input_height: int = 224,
        input_width: int = 224,
        task: TaskType | None = TaskType.SEGMENTATION,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        declared = _check_declared(
            num_patches, input_height, input_width, patch_size_y, patch_size_x
        )
        # The legacy model rebuilt the image with 3*ph*pw channels and handed it to a
        # 3-channel conv1; unpatch() restores the true (B, 3, H, W) image instead.
        self.patch_size_x, self.patch_size_y = patch_size_x, patch_size_y
        self.input_height, self.input_width = input_height, input_width
        self.mlp_head_units = tuple(mlp_head_units)
        self.transformer = _ViTBackbone(
            3 * patch_size_y * patch_size_x,
            patch_size_x,
            patch_size_y,
            declared,
            projection_dim,
            transformer_layers,
            num_heads,
            mlp_units=tuple(mlp_head_units),
        )
        self.unet = UNet(
            n_classes,
            input_height,
            input_width,
            backbone="resnet50",
            pretrained=pretrained,
            task=task,
        )
        self.n_classes = n_classes
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = patchify(x, self.patch_size_y, self.patch_size_x)
        reconstructed = unpatch(
            self.transformer(tokens), x.shape[-2], x.shape[-1], self.patch_size_y, self.patch_size_x
        )
        # The encoder consumes the transformer's output, but the head fuses the *page*: upstream
        # hands ``inputs`` to ``unet_decoder`` (``models.py:427``) and only ``encoded_patches`` to
        # the ResNet (``:425``). ``self.unet(reconstructed)`` would show the head a reconstruction
        # of the page instead of the page, quietly changing the head's input.
        logits = self.unet.head(self.unet.decoder(self.unet.encoder(reconstructed)), x)
        return self.unet.activate(logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilities through this class's own ``forward``, not the wrapped U-Net's."""
        return probs(
            self.forward(x),
            task_binary=self.task in (TaskType.BINARIZATION, TaskType.ENHANCEMENT),
        )


def _stage_spatial(height: int, width: int, stage: int) -> tuple[int, int]:
    """Spatial size of ResNet ``f<stage>``, via :data:`sbb_pytorch.models.base.RESNET_STRIDES`.

    The inline formula this replaces (``divisor = 2 if stage == 0 else 2 ** stage``) counted one
    halving in the stem, but the stem is ``conv1`` *stride 2* **and** ``maxpool`` *stride 2*, so
    every level below it is a further power of two smaller: ``f4`` of a 448x448 page is 14x14,
    not 28x28. Deriving the patch count from the wrong size built position embeddings four times
    too large and the ``+`` in :class:`PatchEncoder` then failed on a shape tuple.
    """
    if not 0 <= stage < len(RESNET_STRIDES):
        raise ModelBuildError(
            f"ResNet stages are numbered 0-{len(RESNET_STRIDES) - 1}, got {stage}"
        )
    divisor = RESNET_STRIDES[stage]
    if height % divisor or width % divisor:
        raise ModelBuildError(
            f"{height}x{width} is not divisible by {divisor}, so stage {stage} is not integral"
        )
    return height // divisor, width // divisor


def _check_declared(
    declared: int | None, height: int, width: int, patch_height: int, patch_width: int
) -> int:
    """Return the patch count, verifying the config-supplied one against the geometry."""
    actual = expected_patch_count(height, width, patch_height, patch_width)
    if declared is None or declared == actual:
        return actual
    raise ModelBuildError(
        f"config declares {declared} patches but a {height}x{width} input with "
        f"{patch_height}x{patch_width} patches yields {actual} "
        f"({height // patch_height if height % patch_height == 0 else '?'} x "
        f"{width // patch_width if width % patch_width == 0 else '?'}); fix num_patches_xy "
        f"or the input size -- the model cannot be built from two different geometries"
    )
