"""Model registry and the ``create_model(config)`` factory.

The legacy ``create_model`` was an if/elif chain over ``config.task``, ``backbone_type``,
``light_model`` and ``transformer.cnn_first`` returning one of five classes. Two gaps followed
from writing the dispatch inline:

* ``TaskType.BINARIZATION`` had no branch at all -- worse, the enum had no such member, so a
  binarization config failed at *parse* time. It now routes to the plain U-Net, which is what
  a two-class binarization model is.
* There was no way to ask for a specific architecture without editing the config; ``model=``
  names one directly, and :data:`MODEL_REGISTRY` is the single table the CLI lists.
"""

from __future__ import annotations

import torch.nn as nn

from sbb_pytorch.config.models import BackboneType, SBBConfig, TaskType
from sbb_pytorch.exceptions import ModelBuildError
from sbb_pytorch.models.heads import (
    EnhancementModel,
    MachineBasedReadingOrderModel,
    ResNet50Classifier,
)
from sbb_pytorch.models.unet import ResNet50UNet, ResNet50UNetLight, UNet
from sbb_pytorch.models.vit import ViTResNet50UNet, ViTResNet50UNetTransformerBeforeCNN

__all__ = ["MODEL_REGISTRY", "create_model", "model_name"]

#: Every architecture the package can build, keyed by the name ``sbb --model`` / ``create_model`` take.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "resnet50_unet": ResNet50UNet,
    "resnet50_unet_light": ResNet50UNetLight,
    "vit_resnet50_unet": ViTResNet50UNet,
    "vit_resnet50_unet_transformer_before_cnn": ViTResNet50UNetTransformerBeforeCNN,
    "resnet50_classifier": ResNet50Classifier,
    "machine_based_reading_order": MachineBasedReadingOrderModel,
    "enhancement": EnhancementModel,
}

#: Configurations that mean "the plain CNN U-Net": segmentation and its two-class subsets.
_UNET_TASKS = (TaskType.SEGMENTATION, TaskType.BINARIZATION)


def model_name(config: SBBConfig) -> str:
    """The registry key :data:`MODEL_REGISTRY` should serve for ``config``.

    Raises:
        ModelBuildError: the task/backbone combination has no architecture. The message names
            the offending fields, where the legacy code raised ``ValueError: Unknown task type``.
    """
    if config.task is TaskType.ENHANCEMENT:
        return "enhancement"
    if config.task is TaskType.CLASSIFICATION:
        return "resnet50_classifier"
    if config.task is TaskType.READING_ORDER:
        return "machine_based_reading_order"
    if config.task in _UNET_TASKS:
        if config.backbone_type is BackboneType.TRANSFORMER:
            if config.transformer is None:
                raise ModelBuildError(
                    f"backbone_type={config.backbone_type.value} needs a transformer config"
                )
            return (
                "vit_resnet50_unet"
                if config.transformer.cnn_first
                else "vit_resnet50_unet_transformer_before_cnn"
            )
        return "resnet50_unet_light" if config.light_model else "resnet50_unet"
    raise ModelBuildError(
        f"no architecture for task={config.task.value} with backbone_type="
        f"{config.backbone_type.value}; known tasks are {[t.value for t in TaskType]}"
    )


def create_model(
    config: SBBConfig,
    *,
    model: str | None = None,
    pretrained: bool | None = None,
    pairing: str = "aligned",
) -> nn.Module:
    """Build the model ``config`` describes.

    Args:
        config: The resolved configuration.
        model: Registry key overriding the config-derived choice (``sbb train --model ...``).
        pretrained: Override ``config.pretraining`` -- ``False`` is what tests and offline
            machines use, since the legacy constructors always downloaded ImageNet weights.
        pairing: Decoder skip pairing for the U-Net family; ``"legacy"`` loads
            ``models_pytorch.ResNet50UNet`` checkpoints.

    Raises:
        ModelBuildError: unknown ``model`` name, or a config that names an architecture needing
            a transformer block it does not have.
    """
    name = model or model_name(config)
    try:
        target = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ModelBuildError(
            f"unknown model {name!r}; registry holds {', '.join(sorted(MODEL_REGISTRY))}"
        ) from exc

    load_pretrained = config.pretraining if pretrained is None else bool(pretrained)
    args: dict[str, object] = {
        "n_classes": config.n_classes,
        "input_height": config.input_height,
        "input_width": config.input_width,
        "pretrained": load_pretrained,
    }

    if issubclass(target, UNet):
        args["pairing"] = pairing
    if name in ("vit_resnet50_unet", "vit_resnet50_unet_transformer_before_cnn") and (
        config.transformer is None
    ):
        raise ModelBuildError(f"{name} needs a transformer config; none was given")

    if config.transformer is not None and name in (
        "vit_resnet50_unet",
        "vit_resnet50_unet_transformer_before_cnn",
    ):
        transformer = config.transformer
        # ``num_patches_xy`` describes a grid over the *input image*. That is the right number
        # for the model that patchifies the input (cnn_first=False); the cnn-first variant
        # patchifies f5, whose size follows from the encoder, so the config's grid cannot be its
        # token count and None lets the constructor derive it (and still verify it) instead.
        declared = None if transformer.cnn_first else (
            transformer.num_patches_xy[0] * transformer.num_patches_xy[1]
        )
        args.update(
            patch_size_x=transformer.patchsize_x,
            patch_size_y=transformer.patchsize_y,
            num_patches=declared,
            mlp_head_units=transformer.mlp_head_units,
            transformer_layers=transformer.layers,
            num_heads=transformer.num_heads,
            projection_dim=transformer.projection_dim,
        )

    try:
        built = target(**args)
    except TypeError as exc:
        # The usual cause is a kwargs set that does not fit the target's signature -- the class
        # of mistake the legacy if/elif chain made silently by passing the wrong subset.
        raise ModelBuildError(f"cannot build {name} from {sorted(args)}: {exc}") from exc

    built_config_name = getattr(built, "name", None)
    if built_config_name is None:
        built.name = name
    return built
