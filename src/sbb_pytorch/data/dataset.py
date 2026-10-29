"""``SegmentationDataset`` -- the ``dataset.py`` class, rebuilt on top of ``storage``.

The legacy ``dataset.py`` held three things: ``get_dirs_or_files`` + a pile of image helpers
(all now in :mod:`sbb_pytorch.storage.images`), ``preprocess_and_augment_data`` (now in
:mod:`sbb_pytorch.data.patches`), and this class. What it did wrong:

* **Channel order.** It loaded photos through ``PIL.Image.convert('RGB')`` while
  ``inference_pytorch.py`` loaded the same files through ``cv2.imread`` (BGR) and applied the
  RGB ImageNet statistics to them. Training and inference therefore fed the model different
  tensors for the same file. Everything goes through ``storage.read_image`` now.
* **Label loading.** ``Image.open(mask).convert('L')`` collapses a 3-channel SBB label PNG by
  luminance -- ``{1, 2, 3}`` become ``{10, 90, 187}`` -- and the class ids never matched the
  page-to-label mapping. ``storage.read_mask`` reads the channel that holds the ids, and
  ``storage.remap_mask`` turns stored values into class ids before the tensor is built.
* **The size fix-ups.** After ``ToTensorV2`` it still ran ``F.interpolate`` on the image (a
  ``TypeError`` path whenever the transform output was 2-D) and ``mask.squeeze()`` twice for a
  3-D mask. Both existed because nothing guaranteed the transform output size; a single
  ``A.Resize`` in the pipeline does, so the code is now read-transform-return.
* **Silent mismatches.** An image whose label was missing raised ``FileNotFoundError`` at
  ``__getitem__`` time -- mid-epoch, on a random worker -- rather than at construction, when
  pairing can report it. Pairing happens in ``__init__`` now.

The class name and its ``data_dir/images`` + ``data_dir/labels`` layout are unchanged, so an
existing run directory still loads; ``num_classes`` is now ``n_classes``, the name the config
uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sbb_pytorch.config.models import Page2Label, SBBConfig, TaskType
from sbb_pytorch.data.augmentations import build_transform
from sbb_pytorch.exceptions import ImageReadError
from sbb_pytorch.logging import get_logger
from sbb_pytorch.storage.images import (
    imwrite,
    list_images,
    read_image,
    read_mask,
    remap_mask,
    resize,
    resize_mask,
)

__all__ = [
    "SegmentationDataset",
    "label_path_for",
    "list_image_pairs",
    "make_loader",
    "train_val_split",
]

logger = get_logger("sbb.data")


def label_path_for(
    image_path: Path,
    labels_dir: Path,
    *,
    task: TaskType | str = TaskType.SEGMENTATION,
) -> Path:
    """The label file that belongs to ``image_path``.

    ``provide_patches`` had two rules and applied them by string compare: for segmentation and
    binarization the label is ``<stem>.png`` (the image might be a ``.jpg``), for enhancement it
    is the *same filename* including its extension, because an enhancement label is an RGB page.
    The legacy ``im.split('.')[0]`` also truncated ``page.001.png`` to ``page``, so every
    ``page.00N`` variant resolved to one label and overwrote its siblings -- the stem is used
    here, which is what was meant.

    Raises:
        ImageReadError: no label exists at the resolved path (call
            :func:`list_image_pairs` to filter first).
    """
    task = TaskType(task)
    name = image_path.name if task is TaskType.ENHANCEMENT else f"{image_path.stem}.png"
    path = labels_dir / name
    if not path.is_file():
        raise ImageReadError(str(path), reason="is missing for the image it belongs to")
    return path


def list_image_pairs(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    task: TaskType | str = TaskType.SEGMENTATION,
) -> list[tuple[Path, Path]]:
    """``[(image, label)]`` for every image in ``images_dir`` that has a label.

    Images without a label are logged and dropped instead of being written out as blank
    duplicates (the legacy behaviour) or crashing the epoch. Extensions are matched
    case-insensitively over :data:`~sbb_pytorch.storage.images.IMAGE_SUFFIXES`, so a ``.JPG``
    is found -- ``sorted(glob('*.jpg') + glob('*.png'))`` missed it.
    """
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    if not images_root.is_dir():
        raise ImageReadError(str(images_root), reason="is not a directory")
    if not labels_root.is_dir():
        raise ImageReadError(str(labels_root), reason="is not a directory")

    pairs: list[tuple[Path, Path]] = []
    for image_path in list_images(images_root):
        try:
            pairs.append((image_path, label_path_for(image_path, labels_root, task=task)))
        except ImageReadError:
            logger.warning("no label for %s; skipped", image_path.name)
    return pairs


def train_val_split(
    pairs: Sequence[tuple[Path, Path]],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic ``(train, val)`` split of ``pairs``.

    The legacy code had no split at all -- ``dir_eval`` had to exist as a separate directory --
    so a project with one data folder could not validate. Seeded and order-independent: the same
    *set* of pairs always produces the same split, which the pre-refactor
    ``sorted(glob(...))``-order-dependent indexing did not give.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    ordered = sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))
    if not ordered:
        return [], []
    if val_fraction == 0.0:
        return list(ordered), []
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    val = [ordered[i] for i in sorted(permutation[:val_count].tolist())]
    train = [ordered[i] for i in sorted(permutation[val_count:].tolist())]
    return train, val


class SegmentationDataset(Dataset):
    """Image/mask pairs as ``(float32 (3, H, W), int64 (H, W))`` tensors.

    Args:
        data_dir: Folder holding ``images/`` and ``labels/`` (the layout
            ``preprocess_and_augment_data`` writes and ``train.py`` expected).
        config: The resolved config. Supplies the sizes, the class count, the task and the
            page-to-label mapping, and the default pipeline -- pass it instead of repeating its
            fields (the legacy signature took ``image_size``/``num_classes`` as loose arguments
            that could disagree with the config actually in use).
        pairs: Pre-paired ``(image, label)`` paths, e.g. from :func:`train_val_split`; skips
            discovery. ``data_dir`` is then optional.
        transform: Overrides the pipeline built from ``config``.
        image_size: ``(height, width)`` override of ``config.image_size``.
        n_classes: Override of ``config.n_classes``; validates every mask value.
        task: Override of ``config.task`` -- decides how a label file is named and read.
        mapping: ``{stored_value: class_id}`` override of ``config.page2label``.

    Raises:
        ImageReadError: ``data_dir`` has no images, or a pair's file is unreadable.
        ValueError: a label holds a value outside ``[0, n_classes)``.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        config: SBBConfig | None = None,
        pairs: Sequence[tuple[Path, Path]] | None = None,
        transform: A.Compose | None = None,
        image_size: tuple[int, int] | None = None,
        n_classes: int | None = None,
        task: TaskType | str | None = None,
        mapping: dict[int, int] | Page2Label | None = None,
    ) -> None:
        if config is None and data_dir is None and pairs is None:
            raise ValueError("pass data_dir, pairs, or config")

        self.config = config
        self.task = (
            TaskType(task)
            if task is not None
            else (config.task if config is not None else TaskType.SEGMENTATION)
        )
        self.n_classes = int(
            n_classes if n_classes is not None else (config.n_classes if config else 1)
        )
        if self.n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {self.n_classes}")
        size = image_size or (config.image_size if config is not None else (448, 448))
        self.image_size = (int(size[0]), int(size[1]))

        if pairs is None:
            assert data_dir is not None  # checked above
            root = Path(data_dir)
            self.pairs = list_image_pairs(root / "images", root / "labels", task=self.task)
        else:
            self.pairs = list(pairs)
        if not self.pairs:
            raise ImageReadError(str(data_dir or "pairs"), reason="yielded no image/label pairs")

        self._mapping = self._resolve_mapping(mapping, config)
        self.transform = transform if transform is not None else self._default_transform(config)
        logger.info("dataset built: %d pairs at %s", len(self.pairs), data_dir or "pairs")

    @staticmethod
    def _resolve_mapping(
        mapping: dict[int, int] | Page2Label | None, config: SBBConfig | None
    ) -> dict[int, int] | None:
        """``{stored_value: class_id}`` -- from a dict, or derived from a ``Page2Label``.

        An SBB label PNG *stores the class id in channel 0*, so a page-to-label config gives the
        set of valid ids rather than a remap: the derived mapping is the identity over
        :meth:`~sbb_pytorch.config.models.Page2Label.class_ids`, which is exactly what
        ``storage.remap_mask`` needs in order to reject a label holding an unconfigured value.
        An explicit ``{stored_value: class_id}`` dict overrides it (that is the ``0/255`` case).
        """
        if isinstance(mapping, Page2Label):
            return {class_id: class_id for class_id in mapping.class_ids()}
        if mapping is not None:
            return dict(mapping)
        if config is not None and config.page2label is not None:
            return {
                class_id: class_id for class_id in config.page2label.class_ids()
            }
        return None

    def _default_transform(self, config: SBBConfig | None) -> A.Compose:
        """The eval pipeline -- or ``config``'s own one, sized to :attr:`image_size`."""
        if config is not None:
            return build_transform(config, training=False)
        height, width = self.image_size
        from albumentations.pytorch import ToTensorV2

        from sbb_pytorch.storage.images import IMAGENET_MEAN, IMAGENET_STD

        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def names(self) -> list[str]:
        """Stem of every image, in index order -- what ``inference`` writes results next to."""
        return [path.stem for path, _ in self.pairs]

    def load_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read and size-match one pair *before* the tensor transforms (testable on its own).

        The image is resized with area interpolation and the mask with nearest neighbour
        (``storage.resize_mask``) -- the legacy code resized the *photo* with the nearest
        ``resize_image`` helper, which is why its training images looked blocky next to the
        inference ones.
        """
        image_path, label_path = self.pairs[index]
        image = read_image(image_path)
        mask = read_mask(label_path)
        if self._mapping is not None:
            mask = remap_mask(mask, self._mapping, self.n_classes)
        else:
            self._validate_values(mask, label_path)
        return image, mask

    def _validate_values(self, mask: np.ndarray, label_path: Path) -> None:
        """Fail on a label whose values can't be class ids, instead of training on them."""
        highest = int(mask.max(initial=0))
        if highest >= self.n_classes:
            raise ValueError(
                f"label {label_path} holds class id {highest}, outside [0, "
                f"{self.n_classes}); pass a page2label mapping to remap stored values"
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(image, mask)`` -- the tuple ``DataLoader`` collates, exactly as before."""
        image, mask = self.load_pair(index)
        image, mask = self._apply_transform(image, mask)
        return image, mask.long() if mask.dtype != torch.long else mask

    def _apply_transform(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the pipeline, and fall back to a plain resize when the transform refuses."""
        try:
            out = self.transform(image=image, mask=mask)
        except ValueError:
            # A single-channel image (``.png`` saved as grey) makes ``A.Normalize`` raise. The
            # legacy pipeline crashed here; resizing by hand keeps the file usable, which is
            # what ``inference`` did with it before.
            height, width = self.image_size
            return (
                torch.from_numpy(resize(image, height, width).transpose(2, 0, 1)).float() / 255.0,
                torch.from_numpy(resize_mask(mask, height, width)).long(),
            )
        image_out, mask_out = out["image"], out["mask"]
        return (
            image_out if torch.is_tensor(image_out) else torch.from_numpy(np.asarray(image_out)),
            mask_out if torch.is_tensor(mask_out) else torch.from_numpy(np.asarray(mask_out)),
        )

    def write_pair(self, index: int, dest_dir: str | Path, *, suffix: str = "") -> list[Path]:
        """Write the pair's *untransformed* files into ``dest_dir/{images,labels}``.

        The folders are the ones :func:`list_image_pairs` and ``PatchConfig.images_output``
        already agree on. Writing both files straight into ``dest_dir`` gave them the *same*
        ``{name}.png`` -- the label overwrote the image and the pair was one file short from then
        on, which is what ``dataset.py``'s writer did for every augmentation variant it wrote.
        """
        image, mask = self.load_pair(index)
        name = self.names[index] + suffix
        dest = Path(dest_dir)
        return [
            imwrite(dest / "images" / f"{name}.png", image),
            imwrite(dest / "labels" / f"{name}.png", mask),
        ]


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """The ``DataLoader`` ``train.py`` built inline, in one place.

    ``pin_memory`` defaults to ``torch.cuda.is_available()`` -- the legacy code passed
    ``pin_memory=True`` unconditionally, which warns on every CPU-only box. ``persistent_workers``
    is only on when there are workers to persist (passing it with ``num_workers=0`` is an error).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=bool(num_workers),
    )
