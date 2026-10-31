"""``SBBDataModule`` -- the ``pl.LightningDataModule`` the legacy ``dataset.py`` class became.

The legacy ``DataModule`` was a plain class, not a Lightning one, and it did three things in
``__init__`` that belonged elsewhere: it ran ``preprocess_and_augment_data`` (so constructing it
copied the dataset into ``dir_output``, and two distributed processes wrote the same files at
once), it built two hard-coded ``A.Compose`` pipelines that ignored every augmentation field the
config exposes, and it raised ``FileNotFoundError`` for a missing folder instead of reporting a
configuration problem. Those three now live in :meth:`SBBDataModule.prepare_data`, in
:mod:`sbb_pytorch.data.augmentations`, and in :func:`resolve_roots` respectively.

The rest is the contract ``training/trainer.py`` needs: ``setup(stage)`` plus
``train_dataloader`` / ``val_dataloader`` / ``test_dataloader``. Everything is read from the
:class:`~sbb_pytorch.config.models.SBBConfig` -- ``n_batch``, ``num_workers``, ``n_classes``,
``input_height``/``input_width``, the augmentation flags and the ``patches`` switch -- rather
than the legacy loose arguments, the same rule :mod:`sbb_pytorch.data.dataset` follows: an
argument may override a config field, but the two can never silently disagree about the run.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sbb_pytorch.config.models import SBBConfig
from sbb_pytorch.data.augmentations import build_transform
from sbb_pytorch.data.dataset import (
    SegmentationDataset,
    list_image_pairs,
    make_loader,
    train_val_split,
)
from sbb_pytorch.data.patches import (
    PatchConfig,
    dataset_for,
    generate,
    preprocess_and_augment_data,
)
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.logging import get_logger

__all__ = ["SBBDataModule", "SPLITS_FOR_STAGE", "resolve_roots"]

logger = get_logger("sbb.data")

#: The datasets ``setup(stage)`` builds for each stage name Lightning passes it -- the
#: ``"fit"``/``"validate"``/``"test"``/``"predict"`` strings, ``"sanity_check"``, and ``None``
#: (which means "everything", the way a bare ``setup()`` call does). ``"test"`` and
#: ``"predict"`` name the *validation* set: ``dataset.py`` had no test split and returned
#: ``None`` from ``val_dataloader()`` whenever ``dir_eval`` was missing, and this module always
#: has a validation set (``train_val_split`` supplies one), so there is nothing else to offer.
SPLITS_FOR_STAGE: dict[str | None, tuple[str, ...]] = {
    None: ("train", "val"),
    "fit": ("train", "val"),
    "sanity_check": ("train", "val"),
    "train": ("train",),
    "validate": ("val",),
    "test": ("val",),
    "predict": ("val",),
}


def resolve_roots(config: SBBConfig) -> dict[str, Path]:
    """The ``{split: folder}``` the two datasets read -- one place that knows the on-disk layout.

    Three cases, and all three are the legacy behaviour:

    * ``config.patches`` -- the folder ``provide_patches`` wrote, ``<dir_output>/train``. Patch
      mode has never had an eval folder, so validation reads that same one.
    * ``config.data_is_provided`` -- ``dir_train`` and ``dir_eval`` themselves, which already
      hold the ``images/`` + ``labels/`` layout the loader wants.
    * otherwise -- the ``<dir_output>/{train,eval}`` pair :meth:`SBBDataModule.prepare_data`
      builds from them. ``dataset.py``'s ``__init__`` *mutated* its ``train_dir``/``val_dir``
      arguments to exactly these paths; here it is a derived value, so asking the same config
      twice gives the same answer.

    Called even when the caller passed ``train_pairs``: a config that names no data anywhere is a
    configuration problem, and ``dataset.py`` found out only when ``DataLoader`` opened a file.

    Raises:
        ConfigError: ``data_is_provided`` without a ``dir_train``, or neither field at all.
    """
    if config.patches:
        train = PatchConfig.from_sbb(config).images_output.parent
        return {"train": train, "val": train}
    if not config.dir_train:
        raise ConfigError(
            "no data configured: set dir_train (or data_is_provided, or pass train_pairs) -- "
            "the legacy DataModule raised FileNotFoundError on a missing folder instead"
        )
    if config.data_is_provided:
        return {"train": Path(config.dir_train), "val": Path(config.dir_eval or config.dir_train)}
    output = Path(config.dir_output)
    return {"train": output / "train", "val": output / "eval"}


def source_folder(config: SBBConfig) -> Path:
    """The folder ``preprocess_and_augment_data`` walks for ``config``.

    That function expects ``<root>/{train,eval}/{images,labels}`` -- the layout ``train.py``
    used -- while ``dir_train`` names *one* of those folders, so the root is its parent. A
    ``dir_train`` not named ``train``/``eval``/``images``/``labels`` is used as given, which
    covers a project that points the config straight at a data root.
    """
    path = Path(config.dir_train)
    return path.parent if path.name in {"train", "eval", "images", "labels"} else path


class SBBDataModule(pl.LightningDataModule):
    """Train and validation loaders for an :class:`~sbb_pytorch.config.models.SBBConfig`.

    Args:
        config: The resolved config: directories, ``n_classes``, sizes, ``n_batch``,
            ``num_workers``, ``val_fraction``, the augmentation flags and the ``patches`` switch.
        train_pairs: Pre-paired ``(image, label)`` paths. Given, ``dir_train`` is not listed and
            nothing is generated -- the way ``train.py`` used ``imgs_list``/``segs_list``.
        val_pairs: Same, for validation; given, ``dir_eval`` and ``val_fraction`` are skipped.
        val_fraction: Override of ``config.val_fraction`` -- the share of the training folder used
            for validation when ``dir_eval`` is empty.
        batch_size: Override of ``config.n_batch``.
        num_workers: Override of ``config.num_workers``.
        pin_memory: Override of the device-derived choice
            :func:`~sbb_pytorch.data.dataset.make_loader` otherwise applies.

    Raises:
        ConfigError: no data is reachable (no ``dir_train``, no ``data_is_provided``, no
            ``train_pairs``), or ``setup`` is given a stage it does not know.
    """

    def __init__(
        self,
        config: SBBConfig,
        *,
        train_pairs: list[tuple[Path, Path]] | None = None,
        val_pairs: list[tuple[Path, Path]] | None = None,
        val_fraction: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        self.roots = resolve_roots(config)  # ConfigError when the config describes no data
        fraction = config.val_fraction if val_fraction is None else float(val_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {fraction}")

        super().__init__()
        self.config = config
        self.batch_size = config.n_batch if batch_size is None else int(batch_size)
        self.num_workers = config.num_workers if num_workers is None else int(num_workers)
        self.pin_memory = pin_memory
        self.val_fraction = fraction
        self._pairs: dict[str, list[tuple[Path, Path]] | None] = {
            "train": list(train_pairs) if train_pairs is not None else None,
            "val": list(val_pairs) if val_pairs is not None else None,
        }
        self._datasets: dict[str, SegmentationDataset] = {}

    # ------------------------------------------------------------------- setup

    def prepare_data(self) -> None:
        """Produce the data the datasets read, once (Lightning calls this on rank 0 only).

        The legacy class ran ``preprocess_and_augment_data`` in ``__init__``, so building the
        module twice -- or on two ranks -- rewrote the same files. ``config.patches`` generates
        with :func:`~sbb_pytorch.data.patches.generate`; anything else without
        ``data_is_provided`` goes through
        :func:`~sbb_pytorch.data.patches.preprocess_and_augment_data`; a provided dataset with
        no patches returns immediately, because there is nothing to produce.
        """
        if self._pairs["train"] is not None:
            return  # the pairs were handed in, so there is nothing on disk to build
        if self.config.patches:
            patch_config = PatchConfig.from_sbb(self.config)
            if not patch_config.images_output.is_dir():
                generate(patch_config)
        elif not self.config.data_is_provided and not self.roots["train"].is_dir():
            written = preprocess_and_augment_data(
                source_folder(self.config), self.config.dir_output, self.config
            )
            logger.debug("prepared %d files for %s", written, self.config.dir_output)

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets ``stage`` asks for; idempotent and callable without a Trainer.

        Raises:
            ConfigError: ``stage`` is not one of the names in :data:`SPLITS_FOR_STAGE`.
        """
        if stage is not None and stage not in SPLITS_FOR_STAGE:
            known = ", ".join(repr(name) for name in SPLITS_FOR_STAGE if name is not None)
            raise ConfigError(f"unknown stage {stage!r}; expected one of {known}")
        for split in SPLITS_FOR_STAGE[stage]:
            if split not in self._datasets:
                self._datasets[split] = self._build(split)

    def _build(self, split: str) -> SegmentationDataset:
        """One dataset: in patch mode the folder ``generate`` wrote, otherwise the split's pairs."""
        given = self._pairs[split]
        if self.config.patches:
            # Patch mode's dataset is the generated folder, not ``dir_train`` -- which holds the
            # *pages* the patches were cut from. ``dataset_for`` is the offline path's own
            # resolution of that folder; ``resolve_roots`` and it agree on ``<dir_output>/train``,
            # because patch mode has never written an eval folder.
            if given is not None:
                return SegmentationDataset(config=self.config, pairs=given)
            return dataset_for(
                PatchConfig.from_sbb(self.config),
                n_classes=self.config.n_classes,
                sbb_config=self.config,
            )
        return SegmentationDataset(
            config=self.config,
            pairs=self._pairs_for(split),
            transform=build_transform(self.config, training=split == "train"),
        )

    def _pairs_for(self, split: str) -> list[tuple[Path, Path]]:
        """Pairs for one split: the ones given, or listed, or one half of :func:`train_val_split`.

        ``dataset.py`` insisted on a separate eval folder and raised ``FileNotFoundError``
        without one, so an empty ``dir_eval`` now means "split ``dir_train``" -- the behaviour
        :func:`train_val_split` added. An explicit ``train_pairs``/``val_pairs`` always wins.
        """
        given = self._pairs[split]
        if given is not None:
            return given
        root = self.roots["train"] if not self.config.dir_eval else self.roots[split]
        listed = list_image_pairs(root / "images", root / "labels", task=self.config.task)
        if self.config.dir_eval:
            return listed
        train, val = train_val_split(listed, val_fraction=self.val_fraction)
        return train if split == "train" else val

    # -------------------------------------------------------------- dataloaders

    @property
    def train_dataset(self) -> SegmentationDataset:
        """Set up on first access, so the module also works without a ``Trainer``."""
        self.setup()
        return self._datasets["train"]

    @property
    def val_dataset(self) -> SegmentationDataset:
        self.setup()
        return self._datasets["val"]

    def train_dataloader(self) -> DataLoader:
        """The shuffling training loader, through :func:`~sbb_pytorch.data.make_loader`."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """The evaluation loader, never shuffled."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """The validation loader -- the legacy class offered no test split either."""
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Also the validation loader: predicting needs readable data, not a third split."""
        return self.val_dataloader()

    def _loader(self, dataset: SegmentationDataset, *, shuffle: bool) -> DataLoader:
        return make_loader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
