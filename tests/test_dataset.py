"""Dataset pairing, loading and patch generation -- the ``dataset.py`` + ``patches.py`` behaviour.

Most of the bugs these tests pin down were silent: ``im.split('.')[0]`` produced the wrong label
path, ``Image.convert('L')`` invented class ids, and a missing label only surfaced as a
``FileNotFoundError`` halfway through an epoch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sbb_pytorch.config.models import Page2Label, SBBConfig
from sbb_pytorch.data import (
    PatchConfig,
    SegmentationDataset,
    generate,
    label_path_for,
    list_image_pairs,
    make_loader,
    output_name,
    pad_to,
    pad_with_border,
    patch_slices,
    preprocess_and_augment_data,
    train_val_split,
    variants,
    write_patches,
)
from sbb_pytorch.exceptions import ConfigError, ImageReadError
from sbb_pytorch.storage.images import imwrite, read_image, read_mask

SIZE = 32


def make_config(**overrides: object) -> SBBConfig:
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": SIZE,
        "input_width": SIZE,
        "augmentation": False,
        **overrides,
    }
    return SBBConfig.model_validate(data)


def write_pair(
    root: Path,
    name: str,
    *,
    size: int = SIZE,
    label_values: tuple[int, ...] = (0, 1, 2),
    image_extension: str = ".png",
) -> Path:
    """One ``images/<name>`` + ``labels/<name>.png`` pair; the label cycles ``label_values``."""
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    imwrite(images / f"{name}{image_extension}", np.stack([gradient] * 3, axis=-1))
    codes = np.tile(np.arange(size, dtype=np.int64) % len(label_values), (size, 1))
    imwrite(labels / f"{name}.png", np.asarray(label_values, dtype=np.uint8)[codes])
    return images / f"{name}{image_extension}"


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A three-pair dataset folder in the layout ``train.py`` used."""
    root = tmp_path / "data"
    for name in ("alpha", "beta", "gamma"):
        write_pair(root, name)
    return root


# --------------------------------------------------------------------- pairing


def test_label_path_follows_the_task_rule_and_keeps_dotted_names(tmp_path: Path) -> None:
    """``im.split('.')[0]`` turned ``page.001.png`` into ``page``, so every variant of a document
    shared one label and overwrote its siblings; a ``.jpg`` image still takes the ``.png`` label."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    jpg, dotted = images / "doc.jpg", images / "page.001.png"
    jpg.write_bytes(b"")
    dotted.write_bytes(b"")
    (labels / "doc.png").write_bytes(b"")
    (labels / "page.001.png").write_bytes(b"")

    assert label_path_for(jpg, labels) == labels / "doc.png"
    assert label_path_for(dotted, labels) == labels / "page.001.png"
    # An enhancement label is an RGB page, so it keeps the source extension.
    (labels / "doc.jpg").write_bytes(b"")
    assert label_path_for(jpg, labels, task="enhancement") == labels / "doc.jpg"


def test_missing_label_raises_instead_of_failing_mid_epoch(tmp_path: Path) -> None:
    with pytest.raises(ImageReadError, match="is missing for the image"):
        label_path_for(tmp_path / "images" / "x.png", tmp_path / "labels")


def test_list_image_pairs_is_case_insensitive_and_drops_unlabelled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``sorted(glob('*.jpg') + glob('*.png'))`` missed ``.JPG``; an unlabelled image is now
    logged and skipped instead of being copied as a blank duplicate."""
    root = tmp_path / "data"
    write_pair(root, "upper", image_extension=".JPG")
    write_pair(root, "jpeg", image_extension=".jpg")
    (root / "images" / "notes.txt").write_text("not an image")

    pairs = list_image_pairs(root / "images", root / "labels")
    assert [p.name for p, _ in pairs] == ["jpeg.jpg", "upper.JPG"]
    assert "notes.txt" not in caplog.text


def test_dataset_rejects_an_empty_or_missing_folder(tmp_path: Path) -> None:
    (tmp_path / "empty" / "images").mkdir(parents=True)
    (tmp_path / "empty" / "labels").mkdir()
    with pytest.raises(ImageReadError, match="yielded no image/label pairs"):
        SegmentationDataset(tmp_path / "empty", n_classes=3)
    with pytest.raises(ValueError, match="pass data_dir, pairs, or config"):
        SegmentationDataset()


# ---------------------------------------------------------------------- loading


def test_getitem_returns_aligned_image_and_mask(dataset_root: Path) -> None:
    """The image and mask must be the same page at the configured size -- the legacy code loaded
    photos as RGB through PIL while inference loaded BGR through cv2, and resized the photo with
    a nearest-neighbour helper that made training images blockier than inference ones."""
    dataset = SegmentationDataset(dataset_root, config=make_config())
    assert len(dataset) == 3
    image, mask = dataset[0]

    assert image.shape == (3, SIZE, SIZE) and image.dtype == torch.float32
    assert mask.shape == (SIZE, SIZE) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) == {0, 1, 2}
    # Channel 0 of the tensor is the file's *red* channel, not its blue one.
    np.testing.assert_allclose(
        image.numpy()[0],
        (read_image(dataset.pairs[0][0])[:, :, 0] / 255.0 - 0.485) / 0.229,
        rtol=1e-4,
        atol=1e-3,
    )


def test_mask_values_outside_the_class_set_are_rejected(dataset_root: Path) -> None:
    """A label written with 0/255 (or an anti-aliased 128) used to train on phantom classes."""
    write_pair(dataset_root, "broken", label_values=(0, 7))
    dataset = SegmentationDataset(dataset_root, config=make_config())
    with pytest.raises(ValueError, match=r"holds class id 7, outside \[0, 3\)"):
        dataset.load_pair(dataset.names.index("broken"))


def test_explicit_mapping_remaps_stored_values(dataset_root: Path) -> None:
    """``{0: 0, 255: 1}`` is what a two-class label PNG actually holds; ``convert('L')`` made it
    ``{10, 190}`` and every pixel matched no class."""
    write_pair(dataset_root, "mapped", label_values=(0, 255))
    dataset = SegmentationDataset(
        dataset_root, n_classes=2, mapping={0: 0, 255: 1}, image_size=(SIZE, SIZE)
    )
    _, mask = dataset[dataset.names.index("mapped")]
    assert set(mask.unique().tolist()) == {0, 1}


def test_page2label_supplies_the_identity_mapping_over_configured_ids(tmp_path: Path) -> None:
    """An SBB label PNG stores the class id in channel 0, so a page-to-label config yields the
    identity map over the ids it lists -- exactly what lets ``remap_mask`` reject a label holding
    an unconfigured value. ``Image.convert('L')`` used to blur ``{1, 2}`` into ``{10, 90}`` and
    every pixel matched no class without anyone noticing."""
    root = tmp_path / "data"
    write_pair(root, "page", label_values=(1, 2))
    page2label = Page2Label(use_case="layout", textregions={"<textlines>": 1}, imageregion=2)
    dataset = SegmentationDataset(root, config=make_config(n_classes=3, page2label=page2label))
    # 0 is not one of the region ids the page2label lists, but it is what every background pixel
    # of a rendered label stores, so the derived mapping covers it (see _resolve_mapping).
    assert dataset._mapping == {0: 0, 1: 1, 2: 2}
    assert set(dataset[0][1].unique().tolist()) == {1, 2}

    # The same folder with a mapping that does not cover the stored values is an error, not a
    # silently all-zero one-hot row.
    unconfigured = Page2Label(use_case="layout", textregions={"<a>": 0})
    with pytest.raises(ValueError, match=r"no class id configured for label values \[1, 2\]"):
        SegmentationDataset(root, config=make_config(page2label=unconfigured))[0]


def test_single_class_config_trains_on_the_class_its_case_label_names(tmp_path: Path) -> None:
    """``{"use_case": "textline", "case_label": 1}`` with ``n_classes: 2`` used to refuse its own
    labels: the derived mapping came from ``class_ids()``, the *region* table, which a single-class
    config leaves empty (or fills with the layout ids, whose every entry past 1 then failed
    ``remap_mask``'s ``[0, n_classes)`` range check). ``sbb train`` on the user's textline dataset
    died on ``no class id configured for label values [1] in 2-class setup``."""
    root = tmp_path / "data"
    write_pair(root, "page", label_values=(0, 1))
    page2label = Page2Label(use_case="textline", case_label=1)
    dataset = SegmentationDataset(root, config=make_config(n_classes=2, page2label=page2label))
    assert dataset._mapping == {0: 0, 1: 1}
    assert set(dataset[0][1].unique().tolist()) == {0, 1}

    # A single-class table that still carries the layout ids -- configs/textline.json's shape --
    # maps the same pair, not the six ids the table lists.
    with_tables = Page2Label(
        use_case="textline",
        case_label=1,
        textregions={"rest_as_paragraph": 1, "header": 2, "heading": 2},
        imageregion=4,
    )
    assert with_tables.legal_class_ids() == {0, 1}


def test_unreadable_pair_raises_with_the_path(dataset_root: Path) -> None:
    """``cv2.imread`` returning ``None`` used to become a ``TypeError`` hundreds of lines later,
    or -- inside a bare ``except:`` -- a silent all-white image."""
    (dataset_root / "images" / "alpha.png").write_bytes(b"not an image")
    dataset = SegmentationDataset(dataset_root, config=make_config())
    with pytest.raises(ImageReadError, match="alpha.png"):
        dataset.load_pair(0)


def test_write_pair_writes_both_files_back(dataset_root: Path) -> None:
    """``write_pair`` is what the patch writer reuses, so its output must read back identical."""
    dataset = SegmentationDataset(dataset_root, config=make_config())
    written = dataset.write_pair(0, dataset_root / "out")
    assert [p.name for p in written] == ["alpha.png", "alpha.png"]
    np.testing.assert_array_equal(read_image(written[0]), read_image(dataset.pairs[0][0]))
    np.testing.assert_array_equal(read_mask(written[1]), read_mask(dataset.pairs[0][1]))


# ------------------------------------------------------------------ the split


@pytest.mark.parametrize("fraction,expected_val", [(0.0, 0), (0.2, 1), (0.5, 2), (0.9, 3)])
def test_train_val_split_is_seeded_order_independent(
    dataset_root: Path, fraction: float, expected_val: int
) -> None:
    """The legacy code had no split at all -- ``dir_eval`` had to exist as its own folder -- and
    its ``sorted(glob(...))`` indexing made the split depend on file names."""
    pairs = list_image_pairs(dataset_root / "images", dataset_root / "labels")
    train, val = train_val_split(pairs, val_fraction=fraction, seed=7)
    train_rev, val_rev = train_val_split(list(reversed(pairs)), val_fraction=fraction, seed=7)
    assert (train, val) == (train_rev, val_rev), "the split depended on argument order"
    assert len(val) == expected_val
    assert len(train) + len(val) == len(pairs)
    assert not ({p[0] for p in train} & {p[0] for p in val}), "a pair was in both splits"


def test_train_val_split_validates_the_fraction(dataset_root: Path) -> None:
    pairs = list_image_pairs(dataset_root / "images", dataset_root / "labels")
    with pytest.raises(ValueError, match=r"val_fraction must be in \[0, 1\)"):
        train_val_split(pairs, val_fraction=1.0)
    assert train_val_split([], val_fraction=0.2) == ([], [])


def test_make_loader_batches_and_sets_pin_memory_from_the_device(dataset_root: Path) -> None:
    """``pin_memory=True`` unconditionally warned on every CPU-only box, and
    ``persistent_workers`` with ``num_workers=0`` is a torch error."""
    dataset = SegmentationDataset(dataset_root, config=make_config())
    loader = make_loader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    images, masks = next(iter(loader))
    assert images.shape == (2, 3, SIZE, SIZE)
    assert masks.shape == (2, SIZE, SIZE)
    assert loader.num_workers == 0 and loader.persistent_workers is False
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        make_loader(dataset, batch_size=0)


# ------------------------------------------------------------------- patches


def test_patch_grid_pads_a_small_image_and_clamps_the_last_patch() -> None:
    """``get_patches`` rounded the grid up and clamped the last index back, so the final row and
    column of patches overlapped their neighbours instead of ending in a sliver."""
    slices = patch_slices(50, 30, 20, 10)
    assert len(slices) == 3 * 3  # ceil(50/20) x ceil(30/10)
    assert slices[-1] == (30, 50, 20, 30), "the last patch must end at the edge, not at 40:60"
    assert all(y1 - y0 == 20 and x1 - x0 == 10 for y0, y1, x0, x1 in slices)
    with pytest.raises(ValueError, match="patch size must be positive"):
        patch_slices(20, 20, 0, 5)


def test_pad_to_centres_on_a_white_canvas_and_zeroes_the_label() -> None:
    """The legacy padding built ``float64`` canvases; both outputs are uint8 now, and the label
    pad value is 0 (background) -- never the image's white."""
    image = np.full((4, 6, 3), 128, np.uint8)
    label = np.full((4, 6), 1, np.uint8)
    padded_image, padded_label = pad_to(image, label, 8, 6)

    assert padded_image.shape == (8, 6, 3) and padded_image.dtype == np.uint8
    assert padded_image[0, 0, 0] == 255 and padded_image[2, 0, 0] == 128
    assert padded_label.shape == (8, 6) and padded_label[0, 0] == 0 and padded_label[2, 1] == 1


def test_pad_with_border_replaces_three_legacy_variants() -> None:
    """``do_padding_white`` / ``do_padding_black`` / ``do_padding_label`` were the same eight
    lines with a different constant and three different dtypes."""
    white = pad_with_border(np.zeros((4, 4, 3), np.uint8), border=2, fill=255)
    black = pad_with_border(np.zeros((4, 4, 3), np.uint8), border=2, fill=0)
    assert white.shape == (8, 8, 3) and white[0, 0, 0] == 255 and white[2, 2, 0] == 0
    assert black[0, 0, 0] == 0
    with pytest.raises(ValueError, match="border must be >= 0"):
        pad_with_border(np.zeros((2, 2), np.uint8), border=-1)


def test_write_patches_names_files_and_returns_the_next_index(tmp_path: Path) -> None:
    image = np.full((SIZE, SIZE, 3), 200, np.uint8)
    label = np.zeros((SIZE, SIZE), np.uint8)
    next_index = write_patches(
        image,
        label,
        images_dir=tmp_path / "i",
        labels_dir=tmp_path / "l",
        patch_height=16,
        patch_width=16,
        start_index=5,
    )
    assert next_index == 9  # four 16x16 patches, numbered from 5
    assert sorted(p.name for p in (tmp_path / "i").iterdir()) == [
        "img_5.png",
        "img_6.png",
        "img_7.png",
        "img_8.png",
    ]


def test_write_patches_with_a_scaler_keeps_the_patch_size(tmp_path: Path) -> None:
    """``get_patches_num_scale_new`` scaled the *page*, not the patch -- so a half-size page
    yields fewer, same-sized patches. The variant that scaled the patch was never called."""
    image = np.full((SIZE, SIZE, 3), 200, np.uint8)
    label = np.zeros((SIZE, SIZE), np.uint8)
    assert (
        write_patches(
            image,
            label,
            images_dir=tmp_path / "s" / "images",
            labels_dir=tmp_path / "s" / "labels",
            patch_height=16,
            patch_width=16,
            scaler=0.5,
        )
        == 1
    ), "a 32x32 page at scale 0.5 is 16x16 -> exactly one patch"
    with pytest.raises(ValueError, match="scaler must be in"):
        write_patches(
            image,
            label,
            images_dir=tmp_path / "x" / "images",
            labels_dir=tmp_path / "x" / "labels",
            patch_height=8,
            patch_width=8,
            scaler=0.0,
        )


def test_patch_files_round_trip_byte_for_byte(tmp_path: Path) -> None:
    """The BGR/RGB contract: a patch written from a canonical RGB array reads back identical."""
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :, 2] = 255  # pure red in RGB order
    label = np.tile(np.arange(8, dtype=np.uint8) % 3, (8, 1))
    assert (
        write_patches(
            image,
            label,
            images_dir=tmp_path / "images",
            labels_dir=tmp_path / "labels",
            patch_height=8,
            patch_width=8,
        )
        == 1
    )
    assert np.array_equal(read_image(tmp_path / "images" / "img_0.png"), image)
    assert np.array_equal(read_mask(tmp_path / "labels" / "img_0.png"), label)

    # One folder for both would put the label on top of the image.
    with pytest.raises(ValueError, match="must be different folders"):
        write_patches(
            image, label, images_dir=tmp_path, labels_dir=tmp_path, patch_height=8, patch_width=8
        )


def test_output_name_uses_the_two_legacy_naming_schemes() -> None:
    """``dataset.py`` wrote ``{name}_flip_1.png``; ``utils.provide_patches`` wrote ``img_{n}.png``.
    Both survive -- the stem form in non-patch mode, the counter in patch mode."""
    assert output_name("page", "base") == "page.png"
    assert output_name("page", "flip_1") == "page_flip_1.png"


def test_variant_names_match_the_legacy_set(tmp_path: Path) -> None:
    """Every flag the legacy branch chain tested still produces its own output name."""
    config = PatchConfig(
        dir_img=str(tmp_path / "data"),
        dir_seg=str(tmp_path / "data"),
        input_height=SIZE,
        input_width=SIZE,
        augmentation=True,
        flip_aug=True,
        flip_index=(1, 0, -1),
        blur_aug=True,
        blur_k=("gauss",),
        brightening=True,
        brightness=(1.5,),
        binarization=True,
        degrading=True,
        degrade_scales=(0.5,),
        rotation=True,
        rotation_not_90=True,
        thetha=(5.0, -5.0),
        channels_shuffling=True,
        shuffle_indexes=((1, 0, 2),),
        scaling=True,
        scales=(0.8,),
        scaling_flip=True,
        scaling_binarization=True,
        scaling_bluring=True,
        scaling_brightness=True,
        shifting=True,
        padding_white=True,
        padding_black=True,
    )
    image, label = np.zeros((SIZE, SIZE, 3), np.uint8), np.zeros((SIZE, SIZE), np.uint8)
    names = [variant.name for variant in variants(image, label, config)]
    assert names[0] == "base"
    for expected in (
        "flip_1",
        "flip_0",
        "flip_-1",
        "blur_gauss",
        "bright_1.5",
        "bin",
        "degrade_0.5",
        "rot90",
        "rot_5.0",
        "rot_-5.0",
        "shuffle_102",
        "scale_0.8",
        "shift_xpos",
        "shift_xmin",
        "shift_ypos",
        "shift_ymin",
        "shift_xypos",
        "shift_xymin",
        "pad_white",
        "pad_black",
        "scale_0.8_flip_1",
        "scale_0.8_bin",
        "scale_0.8_blur_gauss",
        "scale_0.8_bright_1.5",
    ):
        assert expected in names, f"{expected} disappeared"


def test_no_augmentation_means_the_base_pair_only(tmp_path: Path) -> None:
    """The legacy code checked ``augmentation`` once, then never re-checked the sub-flags."""
    config = PatchConfig(augmentation=False, flip_aug=True, binarization=True)
    assert [
        v.name
        for v in variants(
            np.zeros((SIZE, SIZE, 3), np.uint8), np.zeros((SIZE, SIZE), np.uint8), config
        )
    ] == ["base"]


def test_compositing_flags_require_their_source_folders() -> None:
    """``adding_rgb_background`` without a folder crashed mid-loop on a ``None`` image."""
    with pytest.raises(ValueError, match="adding_rgb_background/adding_rgb_foreground needs"):
        PatchConfig(adding_rgb_background=True)
    with pytest.raises(ValueError, match="add_red_textlines needs dir_img_bin"):
        PatchConfig(add_red_textlines=True)


def test_patch_config_from_sbb_copies_every_shared_field() -> None:
    """The legacy call handed ~25 values from a config dict into 25 positional arguments."""
    config = make_config(
        dir_train="/in/train",
        dir_eval="/in/eval",
        dir_output="/out",
        augmentation=True,
        flip_aug=True,
        blur_k=["gauss"],
        scales=(0.5,),
        brightness=(1.2,),
        thetha=(3.0,),
    )
    patch_config = PatchConfig.from_sbb(config)
    assert patch_config.dir_img == "/in/train" and patch_config.dir_seg == "/in/eval"
    assert patch_config.input_height == SIZE and patch_config.task.value == "segmentation"
    assert patch_config.blur_k[0].value == "gauss" and patch_config.scales == (0.5,)
    assert patch_config.brightness == (1.2,) and patch_config.thetha == (3.0,)
    with pytest.raises(ValueError, match="extra_forbidden"):
        PatchConfig.from_sbb(config, n_not_a_field=1)


def test_generate_writes_one_file_per_variant(tmp_path: Path) -> None:
    """``dir_img`` holds the images, ``dir_seg`` the labels -- not both set to the data root."""
    write_pair(tmp_path / "data", "only")
    config = PatchConfig(
        dir_img=str(tmp_path / "data" / "images"),
        dir_seg=str(tmp_path / "data" / "labels"),
        input_height=SIZE,
        input_width=SIZE,
        augmentation=False,
        dir_output=str(tmp_path / "run"),
    )
    assert generate(config) == 2
    assert (tmp_path / "run" / "train" / "images" / "only.png").is_file()
    assert (tmp_path / "run" / "train" / "labels" / "only.png").is_file()


def test_generate_in_patch_mode_writes_every_patch(tmp_path: Path) -> None:
    write_pair(tmp_path / "data", "only")
    config = PatchConfig(
        dir_img=str(tmp_path / "data" / "images"),
        dir_seg=str(tmp_path / "data" / "labels"),
        input_height=16,
        input_width=16,
        augmentation=False,
        patches=True,
        dir_output=str(tmp_path / "run"),
    )
    assert generate(config) == 8  # four patches, image and label each
    assert len(list((tmp_path / "run" / "train" / "images").glob("*.png"))) == 4
    assert sorted(p.name for p in (tmp_path / "run" / "train" / "labels").glob("*.png")) == [
        f"img_{n}.png" for n in range(4)
    ]


def test_generate_dry_run_counts_without_writing(tmp_path: Path) -> None:
    write_pair(tmp_path / "data", "only")
    config = PatchConfig(
        dir_img=str(tmp_path / "data" / "images"),
        dir_seg=str(tmp_path / "data" / "labels"),
        input_height=SIZE,
        input_width=SIZE,
        dir_output=str(tmp_path / "run"),
    )
    assert generate(config, dry_run=True) == 2
    assert not (tmp_path / "run").exists()


def test_generate_rejects_unbalanced_explicit_lists(tmp_path: Path) -> None:
    """``zip(imgs_list_train, segs_list_train)`` silently trained on the shorter of the two."""
    write_pair(tmp_path / "data", "second")
    images, labels = tmp_path / "data" / "images", tmp_path / "data" / "labels"
    config = PatchConfig(
        dir_img=str(images),
        dir_seg=str(labels),
        input_height=SIZE,
        input_width=SIZE,
        imgs_list=(str(images / "second.png"),),
        segs_list=(str(labels / "second.png"),),
    )
    assert generate(config, dry_run=True) == 2
    with pytest.raises(ConfigError, match=r"2 images but 1 labels"):
        generate(
            PatchConfig(
                dir_img=str(images),
                dir_seg=str(labels),
                imgs_list=(str(images / "second.png"), str(images / "second.png")),
                segs_list=(str(labels / "second.png"),),
            ),
            dry_run=True,
        )
    with pytest.raises(ConfigError, match="imgs_list without segs_list needs dir_seg"):
        generate(
            PatchConfig(
                input_height=SIZE, input_width=SIZE, imgs_list=(str(images / "second.png"),)
            ),
            dry_run=True,
        )


def test_preprocess_and_augment_data_writes_both_splits(tmp_path: Path) -> None:
    """``dataset.preprocess_and_augment_data`` wrote ``train/{images,labels}`` and
    ``eval/{images,labels}`` and printed its progress; the layout is unchanged, the prints are
    now log records."""
    source = tmp_path / "source"
    write_pair(source / "train", "train_only")
    write_pair(source / "eval", "eval_only")
    config = make_config(dir_train=str(source / "train"), dir_eval=str(source / "eval"))

    written = preprocess_and_augment_data(source, tmp_path / "out", config)
    assert written == 4
    assert (tmp_path / "out" / "train" / "images" / "train_only.png").is_file()
    assert (tmp_path / "out" / "eval" / "labels" / "eval_only.png").is_file()


def test_preprocess_skips_a_missing_split_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "source"
    write_pair(source / "train", "only")
    config = make_config(dir_train=str(source / "train"))
    assert preprocess_and_augment_data(source, tmp_path / "out", config) == 2
    assert "skipping eval" in caplog.text


def test_a_config_file_on_disk_still_builds_the_pipeline(tmp_path: Path) -> None:
    """End-to-end guard on the repo's own JSON: parse, validate, load, and check the tensors."""
    path = tmp_path / "config.json"
    data = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": SIZE,
        "input_width": SIZE,
        "augmentation": True,
        "flip_aug": True,
        "dir_train": str(tmp_path / "data"),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    config = SBBConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    write_pair(tmp_path / "data", "page")
    image, mask = SegmentationDataset(tmp_path / "data", config=config)[0]
    assert image.shape == (3, SIZE, SIZE) and mask.shape == (SIZE, SIZE)
