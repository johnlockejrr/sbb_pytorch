"""``SBBDataModule`` -- the contract ``training/trainer.py`` uses: ``setup`` plus three loaders.

The legacy ``DataModule`` was a plain class that preprocessed inside ``__init__``, demanded a
separate eval folder, and returned ``None`` from ``val_dataloader()`` when it was missing. Each
of those has a test here, together with the four ways its data can be laid out on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
from torch.utils.data import RandomSampler, SequentialSampler

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.data import SBBDataModule
from sbb_pytorch.data.datamodule import SPLITS_FOR_STAGE, resolve_roots, source_folder
from sbb_pytorch.data.patches import PatchConfig
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.storage.images import imwrite

SIZE = 32


def make_config(**overrides: object) -> SBBConfig:
    data: dict[str, object] = {
        "task": "segmentation",
        "backbone_type": "nontransformer",
        "n_classes": 3,
        "input_height": SIZE,
        "input_width": SIZE,
        "n_batch": 2,
        "num_workers": 0,
        "data_is_provided": True,
        **overrides,
    }
    return SBBConfig.model_validate(data)


def data_root(tmp_path: Path, *, train: int = 5, evals: int = 2) -> Path:
    """``<root>/{train,eval}/{images,labels}`` -- the layout an SBB project ships."""
    root = tmp_path / "data"
    for split, count in (("train", train), ("eval", evals)):
        for index in range(count):
            name = f"{split}{index}"
            gradient = np.tile(np.linspace(0, 255, SIZE, dtype=np.uint8), (SIZE, 1))
            imwrite(root / split / "images" / f"{name}.png", np.stack([gradient] * 3, axis=-1))
            imwrite(
                root / split / "labels" / f"{name}.png",
                np.tile(np.arange(SIZE, dtype=np.uint8) % 3, (SIZE, 1)),
            )
    return root


@pytest.fixture
def module(tmp_path: Path) -> SBBDataModule:
    """A module over a five-train / two-eval dataset, exactly as ``train.py`` had it."""
    root = data_root(tmp_path)
    return SBBDataModule(make_config(dir_train=str(root / "train"), dir_eval=str(root / "eval")))


# --------------------------------------------------------------------- construction


def test_it_is_the_lightning_class_the_legacy_one_was_not(tmp_path: Path) -> None:
    """``dataset.py``'s ``DataModule`` was a plain object, so ``pl.Trainer`` could not take it --
    and ``sbb_pytorch.data`` now exports this one under the name the package promised."""
    module = SBBDataModule(make_config(dir_train=str(data_root(tmp_path) / "train")))
    assert isinstance(module, pl.LightningDataModule)


def test_batch_size_and_workers_come_from_the_config(module: SBBDataModule) -> None:
    """``batch_size=8, num_workers=4`` were arguments to the legacy class while the config said
    otherwise; they are now config fields with a per-call override."""
    assert (module.batch_size, module.num_workers) == (2, 0)
    overridden = SBBDataModule(module.config, batch_size=1, num_workers=0)
    assert (overridden.batch_size, overridden.num_workers) == (1, 0)


def test_val_fraction_is_validated(module: SBBDataModule) -> None:
    with pytest.raises(ValueError, match=r"val_fraction must be in \[0, 1\)"):
        SBBDataModule(module.config, val_fraction=1.0)


# --------------------------------------------------------------------- where the data is


def test_roots_are_the_configured_folders_when_the_data_is_provided(module: SBBDataModule) -> None:
    assert module.roots == {
        "train": Path(module.config.dir_train),
        "val": Path(module.config.dir_eval),
    }


def test_roots_point_at_dir_output_when_the_data_still_has_to_be_built(tmp_path: Path) -> None:
    """The legacy ``__init__`` *mutated* ``train_dir``/``val_dir`` to the output folders after
    preprocessing; asking the same config twice now gives the same answer both times."""
    root = data_root(tmp_path)
    config = make_config(
        dir_train=str(root / "train"),
        dir_eval=str(root / "eval"),
        data_is_provided=False,
        dir_output=str(tmp_path / "out"),
    )
    assert resolve_roots(config) == {
        "train": tmp_path / "out" / "train",
        "val": tmp_path / "out" / "eval",
    }


def test_roots_in_patch_mode_are_the_generated_folder(tmp_path: Path) -> None:
    """Patch mode reads what ``generate`` wrote, not ``dir_train`` -- the pages the patches were
    cut from -- and patch mode has never written an eval folder, so both splits share one."""
    root = data_root(tmp_path)
    config = make_config(
        dir_train=str(root / "train"),
        dir_eval=str(root / "eval"),
        input_height=16,
        input_width=16,
        patches=True,
        dir_output=str(tmp_path / "out"),
    )
    generated = PatchConfig.from_sbb(config).images_output.parent
    assert resolve_roots(config) == {"train": generated, "val": generated}
    assert generated == tmp_path / "out" / "train"


def test_an_unprovided_config_without_dir_train_is_a_config_error(tmp_path: Path) -> None:
    """Before, ``DataLoader`` opened a file and raised ``FileNotFoundError`` mid-epoch."""
    with pytest.raises(ConfigError, match="no data configured"):
        SBBDataModule(make_config(dir_train="", data_is_provided=False))


def test_the_source_folder_is_the_parent_of_a_split_folder(tmp_path: Path) -> None:
    """``preprocess_and_augment_data`` walks ``<root>/{train,eval}/{images,labels}`` while
    ``dir_train`` names one of those folders, so the root it needs is the parent."""
    root = data_root(tmp_path)
    assert source_folder(make_config(dir_train=str(root / "train"))) == root
    flat = tmp_path / "somewhere"
    (flat / "images").mkdir(parents=True)
    assert source_folder(make_config(dir_train=str(flat / "images"))) == flat


# --------------------------------------------------------------------- setup / splits


def test_setup_splits_dir_train_when_dir_eval_is_empty(tmp_path: Path) -> None:
    """The legacy class needed a second folder to validate against, and returned ``None`` from
    ``val_dataloader()`` without one; ``train_val_split`` is the version that lets one folder
    serve both roles -- and it is seeded, so the split does not move between runs."""
    root = data_root(tmp_path, train=5, evals=0)
    config = make_config(dir_train=str(root / "train"), dir_eval="")
    module = SBBDataModule(config, val_fraction=0.4)
    module.setup("fit")
    assert len(module.train_dataset) == 3 and len(module.val_dataset) == 2
    assert not set(module.train_dataset.names) & set(module.val_dataset.names)
    assert SBBDataModule(config, val_fraction=0.4).val_dataset.names == (
        module.val_dataset.names
    ), "the split moved between two builds of the same config"


def test_setup_uses_dir_eval_when_it_is_set(module: SBBDataModule) -> None:
    module.setup("fit")
    assert len(module.train_dataset) == 5 and len(module.val_dataset) == 2
    assert module.val_dataset.names == ["eval0", "eval1"]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (None, {"train", "val"}),
        ("fit", {"train", "val"}),
        ("sanity_check", {"train", "val"}),
        ("train", {"train"}),
        ("validate", {"val"}),
        ("test", {"val"}),
        ("predict", {"val"}),
    ],
)
def test_setup_builds_exactly_the_splits_a_stage_needs(
    tmp_path: Path, stage: str | None, expected: set[str]
) -> None:
    """``Trainer.fit`` asks for train+val and ``Trainer.test`` for the test set; the legacy class
    always built both, and building more than asked means loading images nobody will read."""
    assert set(SPLITS_FOR_STAGE[stage]) == expected
    root = data_root(tmp_path)
    module = SBBDataModule(make_config(dir_train=str(root / "train"), dir_eval=str(root / "eval")))
    module.setup(stage)
    assert set(module._datasets) == expected


def test_setup_rejects_a_stage_lightning_would_not_pass(module: SBBDataModule) -> None:
    with pytest.raises(ConfigError, match="unknown stage 'warmup'"):
        module.setup("warmup")
    assert SPLITS_FOR_STAGE["fit"] == ("train", "val")


def test_setup_is_idempotent(module: SBBDataModule) -> None:
    """Lightning calls ``setup`` on every ``fit``, and a rebuild used to re-list and re-sort the
    folder; the datasets are built once and reused."""
    module.setup("fit")
    first = dict(module._datasets)
    module.setup("fit")
    assert module._datasets == first


def test_explicit_pairs_bypass_the_directories(tmp_path: Path) -> None:
    """``imgs_list``/``segs_list`` were the legacy way to train on a subset; a folder with an
    unlabelled image would otherwise drag that image in (or fail on it)."""
    root = data_root(tmp_path, train=5, evals=2)
    (root / "train" / "images" / "stray.jpg").touch()  # no label: listed, reported, dropped
    pairs = [(root / "train" / "images" / "t0.png", root / "train" / "labels" / "t0.png")]
    module = SBBDataModule(
        make_config(dir_train=str(root / "train"), dir_eval=str(root / "eval")),
        train_pairs=pairs,
        val_pairs=pairs,
    )
    module.setup("fit")
    assert module.train_dataset.names == ["t0"] and module.val_dataset.names == ["t0"]


# --------------------------------------------------------------------- the three loaders


def test_the_three_loaders_agree_on_batching_and_shuffling(module: SBBDataModule) -> None:
    """``pin_memory=True`` unconditionally warned on a CPU box and ``shuffle`` was wrong for the
    eval loader; both now come from ``make_loader``, and the test loader is the val one."""
    module.setup("fit")
    train_loader, val_loader = module.train_dataloader(), module.val_dataloader()
    images, masks = next(iter(train_loader))
    assert images.shape == (2, 3, SIZE, SIZE) and masks.shape == (2, SIZE, SIZE)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert module.test_dataloader().dataset is module.val_dataset
    assert module.predict_dataloader().dataset is module.val_dataset


def test_the_loaders_work_without_a_trainer(module: SBBDataModule) -> None:
    """``setup`` is Lightning's job; calling ``train_dataloader()`` directly used to hit an
    ``AttributeError`` on a class that only built its datasets in ``__init__``."""
    assert module.train_dataloader().dataset is module.train_dataset
    assert module.val_dataloader().dataset is module.val_dataset


# --------------------------------------------------------------------- prepare_data


def test_prepare_data_generates_patches_and_the_loaders_read_them(tmp_path: Path) -> None:
    """``train.py`` ran ``provide_patches`` by hand before training; a ``Trainer`` calls
    ``prepare_data`` instead, and the loaders must then read the generated folder."""
    root = data_root(tmp_path, train=1, evals=1)
    config = make_config(
        dir_train=str(root / "train"),
        dir_eval=str(root / "eval"),
        input_height=16,
        input_width=16,
        patches=True,
        dir_output=str(tmp_path / "out"),
    )
    module = SBBDataModule(config)
    module.prepare_data()
    written = sorted(p.name for p in (tmp_path / "out" / "train" / "images").glob("*.png"))
    assert written == [f"img_{n}.png" for n in range(4)], "four 16x16 patches from one 32x32 page"

    module.prepare_data()  # idempotent: the folder exists, so nothing is generated twice
    assert sorted(p.name for p in (tmp_path / "out" / "train" / "images").glob("*.png")) == written
    module.setup("fit")
    assert len(module.train_dataset) == 4 and len(module.val_dataset) == 4


def test_prepare_data_preprocesses_a_folder_that_is_not_marked_provided(tmp_path: Path) -> None:
    """The ``DataModule`` ran ``preprocess_and_augment_data`` in ``__init__`` -- twice on two
    ranks. It now runs here, once, and its output is what the loaders read."""
    root = data_root(tmp_path, train=2, evals=1)
    config = make_config(
        dir_train=str(root / "train"),
        dir_eval=str(root / "eval"),
        data_is_provided=False,
        dir_output=str(tmp_path / "out"),
    )
    module = SBBDataModule(config)
    module.prepare_data()
    assert (tmp_path / "out" / "train" / "images" / "train0.png").is_file()
    assert (tmp_path / "out" / "eval" / "labels" / "eval0.png").is_file()
    module.setup("fit")
    assert len(module.train_dataset) == 2 and len(module.val_dataset) == 1


def test_prepare_data_does_nothing_when_the_data_is_provided(tmp_path: Path) -> None:
    module = SBBDataModule(make_config(dir_train=str(data_root(tmp_path) / "train")))
    module.prepare_data()  # returns at once; nothing under dir_output is created
    assert not (tmp_path / "out").exists()
