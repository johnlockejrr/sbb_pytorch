"""The predictor and the renderings, checked against the paths ``inference_pytorch.py`` had.

The architectures are covered by ``tests/test_models.py``; here a stub model stands in for one so
these tests stay about the surrounding logic the legacy script got wrong -- channel order, resize
order, the size each output is written at, and what ``--save`` / ``--save-layout`` mean per task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.gt.render import DEFAULT_LABEL_PALETTE
from sbb_pytorch.inference import (
    Prediction,
    Predictor,
    binarization_image,
    colorize,
    overlay,
    render,
    save_images,
)
from sbb_pytorch.inference.visualize import enhancement_image
from sbb_pytorch.models import ResNet50UNet
from sbb_pytorch.storage.checkpoint import Checkpoint
from sbb_pytorch.storage.images import imwrite, read_image


class StubModel(nn.Module):
    """Emits one fixed logits tensor, so a test never pays for a ResNet50 it isn't testing."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.logits


def make(**overrides: object) -> SBBConfig:
    """A textline-shaped config at a size the stub can stand in for."""
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": 8,
        "input_width": 8,
        "pretraining": False,
        **overrides,
    }
    return SBBConfig.model_validate(data)


@pytest.fixture
def page(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 16x16 RGB page on disk -- the size a prediction has to come back at."""
    array = np.zeros((16, 16, 3), np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = 10, 20, 30
    path = tmp_path / "page.png"
    imwrite(path, array, channel_order="bgr")
    return path, array


# ------------------------------------------------------------------- single-page predict


def test_predict_returns_a_mask_at_the_page_size(tmp_path: Path, page) -> None:
    """The legacy path resized with bilinear on labels; the mask must keep the page's own size."""
    path, array = page
    model = StubModel(torch.tensor([[[[2.0, 1.0], [1.0, 1.0]]]]).expand(1, 3, 2, 2))
    prediction = Predictor(model, make()).predict(path)

    assert isinstance(prediction, Prediction)
    assert prediction.path == path
    assert prediction.mask is not None and prediction.mask.shape == array.shape[:2]
    assert set(np.unique(prediction.mask)).issubset({0, 1, 2})  # nothing interpolated in


def test_predict_accepts_an_array_and_a_path_equally(tmp_path: Path, page) -> None:
    """``SBBPredict`` only worked through its constructor fields; both forms now return the same."""
    path, array = page
    model = StubModel(torch.zeros(1, 3, 8, 8))
    from_path, from_array = Predictor(model, make()).predict(path), Predictor(model, make()).predict(array)
    assert np.array_equal(from_path.mask, from_array.mask)
    assert from_array.path is None


def test_save_writes_the_overlay_at_full_page_resolution(tmp_path: Path, page) -> None:
    """The overlay used to shrink the page to the model's output size and save a thumbnail."""
    path, array = page
    prediction = Predictor(StubModel(torch.zeros(1, 3, 8, 8)), make()).predict(path)
    written = prediction.save(save=str(tmp_path / "out.png"), save_layout=str(tmp_path / "l.png"))

    assert sorted(written) == ["layout", "overlay"]
    assert read_image(tmp_path / "out.png").shape == array.shape
    assert read_image(tmp_path / "l.png").shape == array.shape


def test_save_reports_a_name_the_task_cannot_produce(tmp_path: Path, page) -> None:
    """``--save-layout`` on a classification run used to silently write nothing at all."""
    path, _ = page
    prediction = Predictor(StubModel(torch.zeros(1, 3)), make(task="classification")).predict(path)
    with pytest.raises(ValueError, match="produced nothing to render"):
        prediction.save(save=str(tmp_path / "out.png"))


# --------------------------------------------------------------------------- per task


def test_binarization_writes_black_ink_on_white() -> None:
    """``prediction * -1 + 1``: class 1 is the ink layer, so it renders black on white paper."""
    image = binarization_image(np.array([[0, 1], [1, 0]], np.uint8))
    assert image.shape == (2, 2, 3)
    assert np.array_equal(image[0, 0], [255, 255, 255])
    assert np.array_equal(image[0, 1], [0, 0, 0])


def test_enhancement_output_becomes_a_uint8_image() -> None:
    """The legacy code multiplied by 255 with ``astype``; float input now rounds, never wraps."""
    image = enhancement_image(np.full((3, 4, 4), 0.5, np.float32))
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    assert set(np.unique(image)) <= {127, 128}


def test_classification_ranks_classes_by_probability() -> None:
    """The logits -> label mapping the old code did with ``argmax`` and a dict lookup."""
    logits = torch.tensor([[0.1, 2.0, 0.4]])
    prediction = Predictor(StubModel(logits), make(task="classification")).predict(
        np.zeros((8, 8, 3), np.uint8)
    )
    assert prediction.label == "class_1"
    assert [name for name, _ in prediction.probabilities] == ["class_1", "class_2", "class_0"]
    assert prediction.probabilities[0][1] > prediction.probabilities[1][1]


# -------------------------------------------------------------------------- tiling


def test_predict_grid_averages_overlapping_tiles() -> None:
    """``--patches`` was a ``pass``: patch inference returned the single downscaled prediction."""
    page = np.zeros((16, 16, 3), np.uint8)
    model = StubModel(torch.full((1, 3, 8, 8), 0.25))
    logits = Predictor(model, make()).predict_grid(page)

    assert tuple(logits.shape) == (3, 16, 16)
    assert model.calls == 9  # a 4x4-step grid over 16x16: three rows, three columns
    assert torch.allclose(logits, torch.full((3, 16, 16), 0.25))  # averaged, not summed


def test_predict_grid_rejects_a_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Predictor(StubModel(torch.zeros(1, 3, 8, 8)), make()).predict_grid(
            np.zeros((16, 16, 3), np.uint8), overlap=1.0
        )


def test_from_checkpoint_rebuilds_the_model_from_the_saved_config(tmp_path: Path) -> None:
    """``SBBPredict`` hardcoded ``model.pt`` and understood two checkpoint layouts; this one
    finds the newest checkpoint in a run directory and rebuilds from the config stored in it."""
    config = make()
    built = ResNet50UNet(3, 8, 8, pretrained=False)
    Checkpoint(model_state=built.state_dict(), config=config.model_dump()).save(
        tmp_path / "model.pt"
    )

    predictor = Predictor.from_checkpoint(tmp_path)
    assert predictor.config.n_classes == 3
    assert isinstance(predictor.model, ResNet50UNet)


# ------------------------------------------------------------------------ rendering


def test_colorize_uses_the_shared_palette_and_never_invents_colours() -> None:
    """One palette for the whole project: the ``gt`` table, not inference's own three colours."""
    mask = np.array([[0, 1], [2, 0]], np.uint8)
    image = colorize(mask)
    assert np.array_equal(image[0, 0], DEFAULT_LABEL_PALETTE[0])
    assert np.array_equal(image[0, 1], DEFAULT_LABEL_PALETTE[1])
    assert np.array_equal(image[1, 0], DEFAULT_LABEL_PALETTE[2])


def test_colorize_upscales_the_mask_rather_than_the_colours() -> None:
    """Nearest upscaling of the *labels* keeps class boundaries crisp; colours would smear."""
    image = colorize(np.array([[0, 1]], np.uint8), size=(4, 4))
    assert image.shape == (4, 4, 3)
    assert np.array_equal(image[0, 0], DEFAULT_LABEL_PALETTE[0])
    assert np.array_equal(image[0, 3], DEFAULT_LABEL_PALETTE[1])


def test_overlay_blends_at_the_page_size_not_the_output_size(tmp_path: Path, page) -> None:
    path, array = page
    blended = overlay(array, np.array([[1, 1]], np.uint8), top_is_mask=True)
    assert blended.shape == array.shape


def test_render_requires_the_input_its_task_needs() -> None:
    with pytest.raises(ValueError, match="needs a mask"):
        render("segmentation")
    with pytest.raises(ValueError, match="needs the model's image"):
        render("enhancement")


def test_save_images_skips_unnamed_and_reports_written(tmp_path: Path) -> None:
    images = render("binarization", mask=np.array([[0, 1]], np.uint8))
    # ``save_images`` is keyed by image name, not by CLI flag -- ``Predictor.save`` does that
    # mapping; here the names come straight from ``render``.
    assert save_images(images, prediction=str(tmp_path / "p.png"), layout=None) == ["prediction"]
    assert (tmp_path / "p.png").exists()
    with pytest.raises(ValueError, match="no 'nope' image"):
        save_images(images, nope=str(tmp_path / "x.png"))
