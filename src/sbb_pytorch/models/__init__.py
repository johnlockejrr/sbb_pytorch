"""Every architecture the package can build, and the factory that picks one.

The legacy repository kept seven model classes with five copies of the same decoder in one
708-line ``models_pytorch.py``, four of which could not complete a forward pass. Those five
copies are now one :class:`~sbb_pytorch.models.base.UNetDecoder`, and the classes are thin
assemblies over it:

=========================== ================================================ ==============
Registry key                Class                                            Legacy class
=========================== ================================================ ==============
``resnet50_unet``           :class:`~sbb_pytorch.models.unet.ResNet50UNet`  ``ResNet50UNet``
``resnet50_unet_light``     :class:`~sbb_pytorch.models.unet.ResNet50UNetLight``ResNet50UNetLight``
``vit_resnet50_unet``       :class:`~sbb_pytorch.models.vit.ViTResNet50UNet``ViTResNet50UNet``
``vit_resnet50_unet_transformer_before_cnn``
                            :class:`~sbb_pytorch.models.vit.ViTResNet50UNetTransformerBeforeCNN``ViTResNet50UNetTransformerBeforeCNN``
``resnet50_classifier``     :class:`~sbb_pytorch.models.heads.ResNet50Classifier``create_model`` classification branch
``machine_based_reading_order``
                            :class:`~sbb_pytorch.models.heads.MachineBasedReadingOrderModel``MachineBasedReadingOrderModel``
``enhancement``             :class:`~sbb_pytorch.models.heads.EnhancementModel``EnhancementModel``
=========================== ================================================ ==============

Use :func:`~sbb_pytorch.models.registry.create_model` in preference to importing a class
directly: it reads the ``task`` / ``backbone_type`` / ``light_model`` / ``cnn_first`` fields the
same way the legacy factory did and reports an unusable combination as a ``ModelBuildError``
instead of a ``KeyError`` from a half-built module.
"""

from __future__ import annotations

from sbb_pytorch.models.base import (
    FUSION_CHANNELS,
    RESNET_LEVELS,
    RESNET_STRIDES,
    AttentionBlock,
    ConvBlock,
    ResNetEncoder,
    SegmentationHead,
    UNetDecoder,
    probs,
    resize_to_input,
)
from sbb_pytorch.models.heads import (
    EnhancementModel,
    MachineBasedReadingOrderModel,
    ResNet50Classifier,
    resnet_features,
)
from sbb_pytorch.models.registry import MODEL_REGISTRY, create_model, model_name
from sbb_pytorch.models.unet import (
    FULL_DECODER_CHANNELS,
    LIGHT_DECODER_CHANNELS,
    ResNet50UNet,
    ResNet50UNetLight,
    UNet,
)
from sbb_pytorch.models.vit import (
    PatchDecoder,
    PatchEncoder,
    Patches,
    TransformerBlock,
    TransformerTower,
    ViTResNet50UNet,
    ViTResNet50UNetTransformerBeforeCNN,
    expected_patch_count,
    patchify,
    unpatch,
)

__all__ = [
    "FULL_DECODER_CHANNELS",
    "FUSION_CHANNELS",
    "LIGHT_DECODER_CHANNELS",
    "MODEL_REGISTRY",
    "RESNET_LEVELS",
    "RESNET_STRIDES",
    "AttentionBlock",
    "ConvBlock",
    "EnhancementModel",
    "MachineBasedReadingOrderModel",
    "Patches",
    "PatchDecoder",
    "PatchEncoder",
    "ResNet50Classifier",
    "ResNet50UNet",
    "ResNet50UNetLight",
    "ResNetEncoder",
    "SegmentationHead",
    "TransformerBlock",
    "TransformerTower",
    "UNet",
    "UNetDecoder",
    "ViTResNet50UNet",
    "ViTResNet50UNetTransformerBeforeCNN",
    "create_model",
    "expected_patch_count",
    "model_name",
    "patchify",
    "probs",
    "resnet_features",
    "resize_to_input",
    "unpatch",
]
