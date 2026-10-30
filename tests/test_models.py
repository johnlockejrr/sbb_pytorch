"""Every registry entry builds and completes a forward pass, and each documented fix stays fixed.

The four models that raised on their first forward pass are exercised here at ``(2, 3, 128,
128)`` with ``pretrained=False``. 128 is deliberate: it is the smallest size that keeps every
ResNet level integral *and* leaves the deepest map (``4x4``) patchifiable, so the ViT variants
get real tokens instead of the one-token case that hides shape errors. ``pretrained=False``
keeps the suite offline -- the legacy constructors always downloaded ImageNet weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sbb_pytorch.config.models import SBBConfig, TaskType
from sbb_pytorch.exceptions import ModelBuildError
from sbb_pytorch.models import (
    FULL_DECODER_CHANNELS,
    FUSION_CHANNELS,
    LIGHT_DECODER_CHANNELS,
    MODEL_REGISTRY,
    RESNET_STRIDES,
    UNetDecoder,
    create_model,
    expected_patch_count,
    model_name,
    patchify,
    probs,
    resize_to_input,
    unpatch,
)

IMAGE = torch.randn(2, 3, 128, 128)

#: ``{registry_key: (constructor kwargs, expected output shape)}`` -- the table both the
#: direct-construction and the ``create_model`` tests read, so a signature drift fails loudly.
CASES: dict[str, tuple[dict[str, object], tuple[int, ...]]] = {
    "resnet50_unet": (
        {"n_classes": 3, "input_height": 128, "input_width": 128, "pretrained": False},
        (2, 3, 128, 128),
    ),
    "resnet50_unet_light": (
        {"n_classes": 3, "input_height": 128, "input_width": 128, "pretrained": False},
        (2, 3, 128, 128),
    ),
    "vit_resnet50_unet": (
        {
            "n_classes": 3,
            "patch_size_x": 2,
            "patch_size_y": 2,
            "num_patches": 4,
            "transformer_layers": 1,
            "num_heads": 4,
            "projection_dim": 8,
            "input_height": 128,
            "input_width": 128,
            "pretrained": False,
        },
        (2, 3, 128, 128),
    ),
    "vit_resnet50_unet_transformer_before_cnn": (
        {
            "n_classes": 3,
            "patch_size_x": 8,
            "patch_size_y": 8,
            "num_patches": 256,
            "transformer_layers": 1,
            "num_heads": 4,
            "projection_dim": 8,
            "input_height": 128,
            "input_width": 128,
            "pretrained": False,
        },
        (2, 3, 128, 128),
    ),
    "resnet50_classifier": (
        {"n_classes": 3, "input_height": 128, "input_width": 128, "pretrained": False},
        (2, 3),
    ),
    "machine_based_reading_order": (
        {"n_classes": 3, "input_height": 128, "input_width": 128, "pretrained": False},
        (2, 3),
    ),
    "enhancement": (
        {"n_classes": 3, "input_height": 128, "input_width": 128, "pretrained": False},
        (2, 3, 128, 128),
    ),
}

_CACHE: dict[str, nn.Module] = {}


def build(name: str, **overrides: object) -> nn.Module:
    """Cached constructor: seven ResNet50s are built once for the whole file, per kwargs set."""
    settings: dict[str, object] = {**CASES[name][0], **overrides}
    key = f"{name}:{sorted(str(item) for item in settings.items())}"
    if key not in _CACHE:
        _CACHE[key] = MODEL_REGISTRY[name](**settings)  # type: ignore[arg-type]
    return _CACHE[key]


def config_for(name: str, **overrides: object) -> SBBConfig:
    """The config whose ``task``/``backbone_type``/``light_model``/``cnn_first`` route to ``name``."""
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": 128,
        "input_width": 128,
        "pretraining": False,
        **overrides,
    }
    if name == "resnet50_unet_light":
        data["light_model"] = True
    if name == "resnet50_classifier":
        data["task"] = "classification"
    if name == "machine_based_reading_order":
        data["task"] = "reading_order"
    if name == "enhancement":
        data["task"] = "enhancement"
    if name in ("vit_resnet50_unet", "vit_resnet50_unet_transformer_before_cnn"):
        cnn_first = name == "vit_resnet50_unet"
        data["backbone_type"] = "transformer"
        # The grid is the one the *model* patchifies: f5 (4x4) at 2x2 patches for the cnn-first
        # variant, the input (128x128) at 8x8 patches for the other. See registry.create_model.
        data.update(
            transformer_patchsize_x=2 if cnn_first else 8,
            transformer_patchsize_y=2 if cnn_first else 8,
            transformer_num_patches_xy=[2, 2] if cnn_first else [16, 16],
            transformer_projection_dim=8,
            transformer_mlp_head_units=[128, 64],
            transformer_layers=1,
            transformer_num_heads=4,
            transformer_cnn_first=cnn_first,
        )
    return SBBConfig.model_validate(data)


# ------------------------------------------------------------------ forward passes


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_registered_model_returns_the_expected_shape(name: str) -> None:
    """The regression this whole package exists for: four of the seven models could not run."""
    model = build(name).eval()
    with torch.no_grad():
        out = model(IMAGE)

    assert tuple(out.shape) == CASES[name][1]
    assert torch.isfinite(out).all(), f"{name} produced NaN/Inf logits from a finite input"


def test_registry_keys_and_classes_agree() -> None:
    """``MODEL_REGISTRY`` is the table ``sbb --model`` lists; a dead entry is a broken CLI flag."""
    assert set(MODEL_REGISTRY) == set(CASES)
    assert all(issubclass(cls, nn.Module) for cls in MODEL_REGISTRY.values())


# --------------------------------------------------------------------- the factory


@pytest.mark.parametrize("name", sorted(CASES))
def test_model_name_reproduces_the_legacy_dispatch(name: str) -> None:
    """The old if/elif chain over task + backbone_type + light_model + cnn_first, as a table."""
    assert model_name(config_for(name)) == name


@pytest.mark.parametrize("name", sorted(CASES))
def test_create_model_builds_the_class_the_name_names(name: str) -> None:
    built = create_model(config_for(name), model=name, pretrained=False)
    assert type(built) is MODEL_REGISTRY[name]
    assert built.name == name  # the label every log line and checkpoint carries


def test_create_model_always_passes_n_classes() -> None:
    """The legacy chain passed ``n_classes`` to some branches only; the ViT signatures need it too."""
    config = config_for("vit_resnet50_unet", n_classes=5)
    assert create_model(config, pretrained=False).n_classes == 5


def test_unknown_model_name_names_the_registry() -> None:
    with pytest.raises(ModelBuildError, match=r"unknown model 'nope'.*resnet50_unet"):
        create_model(config_for("resnet50_unet"), model="nope")


def test_vit_needs_a_transformer_config() -> None:
    """Asking for a ViT variant from a plain-CNN config used to be a ``KeyError``-style crash."""
    config = SBBConfig.model_validate(
        {
            "task": "segmentation",
            "backbone_type": "nontransformer",
            "n_classes": 3,
            "input_height": 128,
            "input_width": 128,
        }
    )
    with pytest.raises(ModelBuildError, match="needs a transformer config"):
        create_model(config, model="vit_resnet50_unet")


def test_binarization_task_routes_to_the_plain_unet() -> None:
    """``binarization`` was absent from the task enum *and* the dispatch chain; both now know it."""
    assert model_name(config_for("resnet50_unet", task="binarization")) == "resnet50_unet"


def test_declared_patch_grid_is_verified_against_the_shipped_config() -> None:
    """``config_params.json`` says 56x56 patches at 448x448; the input yields 112x112 = 12544."""
    config = SBBConfig.model_validate(
        {
            "task": "segmentation",
            "backbone_type": "transformer",
            "n_classes": 2,
            "input_height": 448,
            "input_width": 448,
            "pretraining": False,
            "transformer_num_patches_xy": [56, 56],
            "transformer_patchsize_x": 4,
            "transformer_patchsize_y": 4,
            "transformer_projection_dim": 64,
            "transformer_mlp_head_units": [128, 64],
            "transformer_layers": 1,
            "transformer_num_heads": 1,
            "transformer_cnn_first": False,
        }
    )
    with pytest.raises(ModelBuildError, match=r"declares 3136 patches.*12544"):
        create_model(config, pretrained=False)


# ------------------------------------------------------------------ decoder pairing


def test_the_decoder_pairs_every_encoder_level_once() -> None:
    """``ResNet50UNet`` consumed ``[f5, f3, f2, f1, f1]``: f4 unused, f1 twice."""
    assert build("resnet50_unet").decoder.skip_indices == [4, 3, 2, 1, 0]


def test_every_decoder_conv_is_sized_from_its_real_input() -> None:
    """The ``Conv2d`` each block needs once its input is computed, not guessed.

    Block 0 is a 1x1 projection of ``f5`` and every later block sees its predecessor's width
    concatenated with the skip of the same level: ``2048 -> 1024``, then ``1024+1024``,
    ``512+512``, ``256+256``, ``128+64``. The off-by-one decoder this replaced asked for
    ``1536``, ``768`` and ``320`` -- widths that came from pairing block ``i`` with level
    ``i-1`` -- which is why four of the five copied decoders raised on their first forward pass.
    """
    model = build("resnet50_unet")
    weights = {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if key.startswith("decoder.blocks.") and key.endswith(".conv.weight")
    }
    assert weights == {
        "decoder.blocks.0.0.conv.weight": (1024, 2048, 1, 1),
        "decoder.blocks.1.0.conv.weight": (512, 2048, 3, 3),
        "decoder.blocks.2.0.conv.weight": (256, 1024, 3, 3),
        "decoder.blocks.3.0.conv.weight": (128, 512, 3, 3),
        "decoder.blocks.4.0.conv.weight": (64, 192, 3, 3),
    }


def test_light_decoder_input_is_computed_instead_of_guessed() -> None:
    """The light model fed 2048+1024 channels into a ``Conv2d(512, ...)``; the size is now derived."""
    decoder = build("resnet50_unet_light").decoder
    assert tuple(decoder.blocks[0][0].conv.weight.shape) == (512, 2048, 1, 1)
    assert decoder.decoder_channels == LIGHT_DECODER_CHANNELS
    assert FULL_DECODER_CHANNELS == (1024, 512, 256, 128, 64)


def test_decoder_rejects_features_from_a_different_encoder() -> None:
    """A wrong-channel feature map used to surface as a broadcast error three layers deep."""
    decoder = UNetDecoder((64, 256), (32, 32))
    features = [torch.zeros(1, 64, 8, 8), torch.zeros(1, 64, 4, 4)]
    with pytest.raises(ValueError, match="encoder declares 256"):
        decoder(features)


def test_encoder_strides_match_the_feature_maps_it_produces() -> None:
    """The formula ``models/vit.py`` used to derive gave ``H/16`` for f5; the encoder says ``H/32``."""
    from sbb_pytorch.models import ResNetEncoder

    encoder = ResNetEncoder("resnet50", pretrained=False)
    with torch.no_grad():
        features = encoder(IMAGE)
    assert [IMAGE.shape[-1] // f.shape[-1] for f in features] == list(RESNET_STRIDES)
    assert RESNET_STRIDES == (2, 4, 8, 16, 32), "f1 is conv1's H/2 output, taken before the pool"
    assert [f.shape[1] for f in features] == [64, 256, 512, 1024, 2048]
    # The pool belongs to level 1, as it does upstream: conv1 keeps the level-0 keys and layer1
    # moves one container deeper, which is the rename section 3.2 records.
    keys = encoder.state_dict()
    assert "levels.0.0.weight" in keys and "levels.1.1.0.conv1.weight" in keys


# ----------------------------------------------------------------------- patch math


def test_patchify_and_unpatch_round_trip() -> None:
    """``ViTResNet50UNetTransformerBeforeCNN`` rebuilt the image with a ``view`` and got a transpose."""
    image = torch.randn(1, 3, 16, 24)
    tokens = patchify(image, 4, 4)
    assert tuple(tokens.shape) == (1, 4 * 6, 3 * 4 * 4)
    assert torch.equal(unpatch(tokens, 16, 24, 4, 4), image)


@pytest.mark.parametrize(
    ("height", "width", "patch_height", "patch_width", "expected"),
    [(448, 448, 4, 4, 12544), (128, 128, 4, 4, 1024), (672, 448, 4, 4, 18816)],
)
def test_expected_patch_count(
    height: int, width: int, patch_height: int, patch_width: int, expected: int
) -> None:
    assert expected_patch_count(height, width, patch_height, patch_width) == expected


def test_indivisible_patch_geometry_is_refused_with_the_remainders() -> None:
    """``Tensor.unfold`` used to raise a bare ``RuntimeError`` naming neither number."""
    with pytest.raises(ModelBuildError, match=r"14 % 4 = 2"):
        expected_patch_count(14, 16, 4, 4)


# -------------------------------------------------------------------- output heads


def test_segmentation_head_returns_logits_and_exposes_probabilities() -> None:
    """Light and the ViTs returned softmax into a ``CrossEntropyLoss`` -- softmax of a softmax."""
    model = build("resnet50_unet").eval()
    with torch.no_grad():
        logits, probabilities = model(IMAGE), model.predict_proba(IMAGE)

    assert not torch.allclose(logits, probabilities)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, 128, 128), atol=1e-5)
    with torch.no_grad():
        assert torch.equal(build("enhancement").predict_proba(IMAGE), build("enhancement")(IMAGE))


def test_the_fused_head_is_the_stage_the_port_dropped() -> None:
    """The Keras head upsamples, concatenates the image, convolves 64+3 -> 32, normalises,
    then predicts with a 1x1 -- and the port had only the 1x1, run at ``H/4`` and resized."""
    head = build("resnet50_unet").head  # decoder ends at 64 channels, the image adds 3
    assert tuple(head.fusion.conv.weight.shape) == (
        FUSION_CHANNELS,
        FULL_DECODER_CHANNELS[-1] + 3,
        3,
        3,
    )
    assert tuple(head.classify.weight.shape) == (3, FUSION_CHANNELS, 1, 1)
    assert isinstance(head.norm, nn.BatchNorm2d), "upstream normalises the logits it softmaxes"


def test_both_resnet_unets_end_at_the_same_head_widths() -> None:
    """``64`` then ``32`` on both: the light model's old ``(..., 64, 32)`` had merged the head's
    fusion conv into the last decoder block and halved every width below the first."""
    assert LIGHT_DECODER_CHANNELS == (512, 512, 256, 128, 64)
    for name in ("resnet50_unet", "resnet50_unet_light"):
        assert tuple(build(name).head.fusion.conv.weight.shape) == (FUSION_CHANNELS, 64 + 3, 3, 3)


def test_only_a_segmentation_task_gets_normalised_logits() -> None:
    """``if task == "segmentation": BatchNormalization`` -- the sigmoid tasks skip it, so a
    binarization model built on the same weights must not grow one."""
    assert build("resnet50_unet", task=TaskType.BINARIZATION).head.norm is None


def test_every_segmentation_model_shares_one_head_type() -> None:
    """Five architectures, one head class -- the ``nn.Conv2d`` each of them used to carry."""
    heads = [
        build("resnet50_unet").head,
        build("resnet50_unet_light").head,
        build("vit_resnet50_unet").head,
        build("vit_resnet50_unet_transformer_before_cnn").unet.head,
        build("enhancement").unet.head,
    ]
    assert {type(head) for head in heads} == {type(build("resnet50_unet").head)}


def test_probs_follows_the_task_not_the_caller_memory() -> None:
    logits = torch.zeros(1, 2, 2, 2)
    assert torch.allclose(probs(logits, task_binary=True), torch.full((1, 2, 2, 2), 0.5))
    assert torch.allclose(probs(logits), torch.full((1, 2, 2, 2), 0.5))


def test_resize_to_input_is_a_noop_when_the_size_already_matches() -> None:
    """The interpolation branch must not run at the configured size: resampling its own output
    is how the legacy models turned a 672x448 label into a blurred 671x447 prediction."""
    tensor = torch.zeros(1, 3, 5, 5)
    assert resize_to_input(tensor, (5, 5)) is tensor
    assert tuple(resize_to_input(tensor, (9, 9)).shape) == (1, 3, 9, 9)


@pytest.mark.parametrize("name", ["resnet50_unet", "resnet50_classifier"])
def test_wrong_input_rank_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        build(name)(torch.randn(3, 128, 128))
