"""Tests for the image layer; each test names the legacy bug it exists to prevent."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sbb_pytorch.exceptions import ImageReadError
from sbb_pytorch.storage import images as im


@pytest.fixture
def photo() -> np.ndarray:
    """A 12x20 (H, W, 3) uint8 image whose pixels encode their own coordinates."""
    rows, cols = np.mgrid[0:12, 0:20]
    return np.stack(
        [rows.astype(np.uint8), cols.astype(np.uint8), np.full((12, 20), 128, np.uint8)], axis=-1
    )


# --------------------------------------------------------------------------- I/O


def test_imread_rgb_actually_flips_bgr(tmp_path: Path, photo: np.ndarray) -> None:
    """OpenCV reads BGR; the models were trained on RGB. Silent channel swap, now explicit."""
    path = tmp_path / "page.png"
    assert im.imwrite(path, photo, channel_order="bgr") == path  # already OpenCV order

    assert np.array_equal(im.imread_bgr(path), photo)
    rgb = im.imread_rgb(path)
    assert np.array_equal(rgb[..., 0], photo[..., 2])  # R <- stored B
    assert np.array_equal(rgb[..., 2], photo[..., 0])  # B <- stored R
    assert np.array_equal(im.read_image(path), rgb)


def test_imwrite_converts_the_canonical_rgb_order_back_to_bgr(tmp_path: Path) -> None:
    """Saving an RGB array verbatim stored it red-blue swapped: the write-side twin of the
    ``cv2.imread``-returns-BGR bug ``imread_rgb`` fixed on the read side."""
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    path = tmp_path / "page.png"

    im.imwrite(path, rgb)
    assert np.array_equal(im.imread_bgr(path), rgb[..., ::-1])
    assert np.array_equal(im.imread_rgb(path), rgb)

    with pytest.raises(ValueError, match="channel_order"):
        im.imwrite(path, rgb, channel_order="gray")


def test_imread_missing_file_raises_instead_of_returning_none(tmp_path: Path) -> None:
    """``cv2.imread`` returns ``None``; that ``None`` used to reach ``cv2.resize`` 200 lines later."""
    with pytest.raises(ImageReadError, match="could not be decoded"):
        im.imread_bgr(tmp_path / "nope.png")
    with pytest.raises(ImageReadError, match="could not be decoded"):
        im.read_mask(tmp_path / "nope.png")


def test_imwrite_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ImageReadError, match="could not be written"):
        im.imwrite(tmp_path / "x.xyz", np.zeros((2, 2, 3), np.uint8))


def test_read_mask_takes_channel_zero_without_a_colour_flip(tmp_path: Path) -> None:
    """SBB label PNGs keep the class id in channel 0 and a colour map in channels 1-2."""
    label = np.zeros((4, 6, 3), np.uint8)
    label[..., 0] = 2  # class id
    label[..., 1] = 99  # colour mapping, must be ignored
    path = tmp_path / "label.png"
    cv2.imwrite(str(path), label)
    assert np.array_equal(im.read_mask(path), np.full((4, 6), 2, np.uint8))


def test_list_images_is_case_insensitive_and_dots_safe(tmp_path: Path) -> None:
    """Legacy ``name.split('.')[0]`` read the extension of ``page.001.png`` as ``001``."""
    for name in ["b.JPG", "a.png", "page.001.png", "notes.txt", "archive.tar.gz"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "subdir").mkdir()

    assert [p.name for p in im.list_images(tmp_path)] == ["a.png", "b.JPG", "page.001.png"]
    with pytest.raises(FileNotFoundError, match="not a directory"):
        im.list_images(tmp_path / "subdir" / "missing")


# ----------------------------------------------------------------------- dtype


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ([0.0, 0.5, 1.0], [0, 128, 255]),  # [0,1] floats scale by 255, not truncate to 0/1
        ([-5.0, 300.0], [0, 255]),  # negatives clamp instead of wrapping to 255
        ([0.999999, 1.0], [255, 255]),
    ],
)
def test_to_uint8_scales_and_clips_instead_of_wrapping(
    src: list[float], expected: list[int]
) -> None:
    """``astype(np.uint8)`` truncated 0.5 to 0 and wrapped -5 to 251 (``do_degrading``)."""
    assert im.to_uint8(np.array(src, dtype=np.float32)).tolist() == expected


def test_to_uint8_passes_uint8_through_unchanged(photo: np.ndarray) -> None:
    assert im.to_uint8(photo) is photo or np.array_equal(im.to_uint8(photo), photo)


# --------------------------------------------------------------------- geometry


def test_resize_uses_height_width_not_the_cv2_swapped_order(photo: np.ndarray) -> None:
    """``cv2.resize`` wants ``(width, height)``; the old ``resize_image`` got it wrong repeatedly."""
    assert im.resize(photo, 5, 8).shape[:2] == (5, 8)
    assert im.resize(photo, 8, 5).shape[:2] == (8, 5)


def test_resize_mask_uses_nearest_and_invents_no_classes(tmp_path: Path) -> None:
    """Bilinear on a class-id mask manufactures labels like 171 that name no class."""
    mask = np.tile(np.arange(4, dtype=np.uint8), (4, 1))  # rows of 0..3
    resized = im.resize_mask(mask, 12, 12)
    assert resized.shape == (12, 12)
    assert set(np.unique(resized)) <= set(np.unique(mask))


def test_rotate90_rotates_instead_of_reflecting() -> None:
    """Legacy ``rotation_90`` transposed each channel -- a mirror image, not a rotation."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    assert np.array_equal(im.rotate90(arr), np.array([[3, 6], [2, 5], [1, 4]]))
    assert not np.array_equal(im.rotate90(arr), arr.T)  # the old behaviour
    assert np.array_equal(im.rotate90(arr, times=2), np.array([[6, 5, 4], [3, 2, 1]]))
    assert np.array_equal(im.rotate90(arr, times=4), arr)


def test_rotate_not_90_keeps_mask_values_legal(photo: np.ndarray) -> None:
    """``imutils.rotate`` warped masks bilinearly, producing fractional class ids."""
    mask = np.zeros(photo.shape[:2], np.uint8)
    mask[2:10, 4:16] = 1
    rotated_img, rotated_mask = im.rotate_not_90(photo, mask, angle_deg=7.5)
    assert rotated_img.ndim == 3 and rotated_img.shape[2] == 3
    assert set(np.unique(rotated_mask)) <= {0, 1}

    image_only, none_mask = im.rotate_not_90(photo, None, angle_deg=7.5)
    assert none_mask is None and image_only.ndim == 3


def test_rotated_rect_max_area_never_exceeds_the_source(photo: np.ndarray) -> None:
    wr, hr = im.rotated_rect_max_area(20, 12, 10.0)
    assert 0 < wr <= 20 and 0 < hr <= 12
    full_w, full_h = im.rotated_rect_max_area(20, 12, 0.0)
    assert (full_w, full_h) == pytest.approx((20.0, 12.0), abs=1.0)


# ------------------------------------------------------------------- transforms


@pytest.mark.parametrize("kind", ["gauss", "median", "blur"])
def test_blur_accepts_every_declared_kind(photo: np.ndarray, kind: str) -> None:
    assert im.blur(photo, kind, kernel=3).shape == photo.shape


def test_blur_rejects_the_legacy_misspelling_and_even_kernels(photo: np.ndarray) -> None:
    """A shipped config spelled it ``"guass"``; that produced an unbound-local, not an error."""
    with pytest.raises(ValueError, match="not a valid BlurKind"):
        im.blur(photo, "guass")
    with pytest.raises(ValueError, match="must be odd"):
        im.blur(photo, im.BlurKind.GAUSS, kernel=4)


def test_add_noise_with_zero_sigma_is_a_noop(photo: np.ndarray) -> None:
    """The legacy version cast the normal samples with ``astype``, speckling white dots."""
    np.random.seed(0)
    assert np.array_equal(im.add_noise(photo, 0.0), im.to_uint8(photo))
    np.random.seed(0)
    noisy = im.add_noise(photo, 10.0)
    assert noisy.dtype == np.uint8
    assert noisy.shape == photo.shape


@pytest.mark.parametrize("scale", [0.25, 1.0, 2.0])
def test_downscale_upscale_survives_scale_greater_than_one(photo: np.ndarray, scale: float) -> None:
    """``scale_image_for_no_patch`` raised ``ValueError: could not broadcast`` for ``scale > 1``."""
    assert im.downscale_upscale(photo, scale).shape == photo.shape
    with pytest.raises(ValueError, match="scale must be positive"):
        im.downscale_upscale(photo, 0)


def test_brighten_operates_on_the_array_not_a_file_path(photo: np.ndarray) -> None:
    """Legacy ``bright_image`` took a path and round-tripped through PIL + ImageEnhance."""
    darker = im.brighten(photo, 0.5)
    assert darker.dtype == np.uint8
    assert float(darker[..., 2].mean()) < float(photo[..., 2].mean())
    assert np.array_equal(im.brighten(photo, 1.0), photo)


def test_shuffle_channels_swaps_red_and_green(photo: np.ndarray) -> None:
    swapped = im.shuffle_channels(photo, (1, 0, 2))
    assert np.array_equal(swapped[..., 0], photo[..., 1])
    assert np.array_equal(swapped[..., 1], photo[..., 0])
    with pytest.raises(ValueError, match="permutation"):
        im.shuffle_channels(photo, (0, 0, 1))


def test_otsu_binarize_thresholds_once_and_returns_two_values(photo: np.ndarray) -> None:
    """Legacy ``otsu_copy`` computed three thresholds and used only the blue one."""
    binary = im.otsu_binarize(photo)
    assert binary.shape == photo.shape
    assert set(np.unique(binary)) <= {0, 255}


def test_normalize_matches_the_imagenet_formula(photo: np.ndarray) -> None:
    """Shared so training and inference cannot diverge again."""
    result = im.normalize(photo)
    expected = (photo.astype(np.float32) / 255.0 - im.IMAGENET_MEAN) / im.IMAGENET_STD
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6)

    # An already-normalised float array must not be divided by 255 a second time.
    np.testing.assert_allclose(im.normalize(expected), expected, rtol=1e-6)


# ---------------------------------------------------------------- label tensors


def test_remap_mask_validates_every_stored_value() -> None:
    """A 0/255 label image read as class ids: 255 is not class 1 unless the config says so."""
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert np.array_equal(
        im.remap_mask(mask, {0: 0, 255: 1}, n_classes=2), np.array([[0, 1], [1, 0]])
    )

    with pytest.raises(ValueError, match="no class id configured for label values"):
        im.remap_mask(mask, {0: 0}, n_classes=2)
    with pytest.raises(ValueError, match=r"class id 5 for stored value 255 is outside \[0, 2\)"):
        im.remap_mask(mask, {0: 0, 255: 5}, n_classes=2)


def test_one_hot_mask_is_a_single_gather() -> None:
    mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    one_hot = im.one_hot_mask(mask, 3)
    assert one_hot.shape == (2, 2, 3)
    np.testing.assert_array_equal(one_hot.sum(-1), np.ones((2, 2), np.uint8))
    assert one_hot[0, 0].tolist() == [1, 0, 0]
    assert one_hot[1, 0].tolist() == [0, 0, 1]

    with pytest.raises(ValueError, match="out of range for n_classes=2"):
        im.one_hot_mask(mask, 2)
    assert im.one_hot_mask(np.zeros((0, 0), np.uint8), 3).shape == (0, 0, 3)
