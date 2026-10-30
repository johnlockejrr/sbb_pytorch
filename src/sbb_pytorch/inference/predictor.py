"""Run a trained checkpoint on a page: single shot, or a sliding window of tiles.

The legacy ``SBBPredict`` was one class that read the image differently per entry point, resized
its own outputs with the wrong interpolation and had two features that were literally ``pass``
statements (``--patches`` and the entire ``reading_order`` branch). The bugs that mattered, and
what this module does instead:

===============================  ===============================================
Legacy                           Now
===============================  ===============================================
``cv2.imread(path)`` on the      :func:`sbb_pytorch.storage.images.read_image`: RGB, and an
inference path (BGR) while          explicit ``ImageReadError`` instead of a ``None`` reaching
``dataset.py`` trained via PIL      ``cv2.resize`` 200 lines later. The input page is cached on
(RGB), then applied ImageNet's      the prediction so ``--save`` can write the overlay at the
RGB statistics.                     page's own resolution.

Normalised with ImageNet stats    :func:`resize` first (in pixel space), then
*then* resized, and the           :func:`normalize` -- interpolating an already-normalised array
classification path skipped       mixes statistics with pixels, and the classification path
normalisation entirely.           previously skipped it, so it scored a different input
                                  distribution than the one it was trained on.

``resize_image(labels, ...)``     :func:`resize_mask`: nearest-neighbour, and it refuses to
bilinearly resized the label map  invent class ids the label set never contained (bilinear
back to page size, inventing      interpolation of ``[0, 1, 2]`` produces ``[0, 1, 1, 2]`` plus
interpolated class ids.           everything between).

``os.path.join(model_dir,         :func:`latest_checkpoint`, so a run directory of
``'model.pt'`` hard-coded, then   ``epoch_*.pt`` files works, and :class:`Checkpoint`, which
two branches for the two          understands every schema the repo has ever written
checkpoint layouts in the wild.   (``model_state``/``model_state_dict``/``state_dict``/``model``/
                                  bare) instead of only two.

Task-specific behaviour kept: segmentation and binarization produce a class-id ``mask``, enhancement an RGB ``image``,
classification a ``label`` plus probabilities. ``min_area`` and the XML writing for reading order live in
:mod:`sbb_pytorch.gt`, which is where the region ordering itself is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbb_pytorch.config.loaders import load_config
from sbb_pytorch.config.models import SBBConfig, TaskType
from sbb_pytorch.device import resolve_device
from sbb_pytorch.exceptions import ModelBuildError
from sbb_pytorch.inference.visualize import enhancement_image, render, save_images
from sbb_pytorch.models.registry import create_model
from sbb_pytorch.storage.checkpoint import Checkpoint, latest_checkpoint
from sbb_pytorch.storage.images import normalize, read_image, resize, resize_mask

__all__ = ["Prediction", "Predictor"]


@dataclass(slots=True)
class Prediction:
    """One page's worth of model output, in whichever of the three forms its task produces.

    Attributes:
        task: What was predicted, so :meth:`save` knows which renderings exist.
        logits: ``(n_classes, H, W)`` at the model's output size, kept for metrics and
            ``--save-logits`` style debugging. ``None`` only for tasks whose model output is not a
            per-pixel field (classification, reading order).
        mask: ``(page_height, page_width)`` class ids at the *page's* resolution.
        image: The produced picture for the enhancement task (RGB, page resolution).
        label: ``--save-json`` style answer for the classification and reading-order tasks.
        probabilities: ``(class name, probability)`` pairs, highest first, same two tasks.
        path: Where the input page came from, used to name the default output files.
    """

    task: TaskType
    logits: torch.Tensor | None = None
    mask: np.ndarray | None = None
    image: np.ndarray | None = None
    label: str | None = None
    probabilities: tuple[tuple[str, float], ...] = ()
    path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def images(self, source: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Render this prediction (``overlay`` / ``layout`` / ``prediction``) as RGB images.

        Args:
            source: The input page; needed for the segmentation overlay. Defaults to the page
                this prediction was made from when that is known.
        """
        page = source
        if page is None and self.path is not None:
            page = read_image(self.path)
        return render(self.task, mask=self.mask, image=self.image, source=page)

    def save(self, *, save: str | None = None, save_layout: str | None = None) -> list[str]:
        """Write the ``--save`` / ``--save-layout`` outputs; returns the image names written.

        ``--save`` takes the overlay for segmentation and the ``prediction`` image for the other
        tasks, mirroring the legacy script -- which wrote one file for binarization, one for
        enhancement and nothing at all for classification. Asking for a rendering the task did
        not produce is an error now instead of a silently missing file.
        """
        images = self.images()
        primary = "overlay" if self.task is TaskType.SEGMENTATION else "prediction"
        return save_images(images, **{primary: save, "layout": save_layout})


class Predictor:
    """``sbb infer`` engine: build once from a run directory, then predict on any number of pages.

    Args:
        model: A constructed model. Usually obtained through :meth:`from_checkpoint`.
        config: The configuration the model was built from -- it carries the input size, the task
            and the class mapping the prediction needs, so it is not optional here.
        device: ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"`` or ``"cuda:N"``; see
            :func:`sbb_pytorch.device.resolve_device`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SBBConfig,
        *,
        device: str | None = "auto",
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        config: SBBConfig | str | Path | None = None,
        model: str | None = None,
        device: str | None = "auto",
    ) -> Predictor:
        """Restore the weights in ``model_dir`` and the configuration saved next to them.

        Args:
            model_dir: A training run directory (``model.pt``, or the newest ``epoch_*.pt``).
            config: The config used to rebuild the architecture. Omit it to read the
                ``*_config.json`` sidecar that :class:`~sbb_pytorch.storage.checkpoint.Checkpoint`
                writes -- which is what lets ``sbb infer --model runs/textline`` work without
                repeating ``--config``; the legacy script read ``<dir>/config.json`` the same way.
            model: Registry key overriding the config-derived architecture (``--model``).
            device: See :class:`Predictor`.

        Raises:
            FileNotFoundError: no checkpoint in ``model_dir``.
            ModelBuildError: the config names an architecture that cannot be built.
        """
        directory = Path(model_dir)
        checkpoint_file = directory / "model.pt"
        if not checkpoint_file.exists():
            checkpoint_file = latest_checkpoint(directory)
        checkpoint = Checkpoint.load(checkpoint_file)

        resolved = cls._resolve_config(config, checkpoint)
        built = create_model(resolved, model=model, pretrained=False)
        checkpoint.apply_to(built)
        return cls(built, resolved, device=device)

    @staticmethod
    def _resolve_config(config: SBBConfig | str | Path | None, checkpoint: Checkpoint) -> SBBConfig:
        """An explicit config wins; otherwise the one the checkpoint stored, otherwise error."""
        if isinstance(config, SBBConfig):
            return config
        if config is not None:
            return load_config(config)
        if checkpoint.config:
            return SBBConfig.from_dict(checkpoint.config)
        raise ModelBuildError(
            "no configuration available: pass --config or train with a version that records one "
            "beside the checkpoint (the legacy code read <model_dir>/config.json)"
        )

    # ------------------------------------------------------------------ single page

    def predict(
        self, image: np.ndarray | str | Path, *, patches: bool = False, overlap: float = 0.5
    ) -> Prediction:
        """Predict on one page, given as an array or a path.

        Args:
            image: The page. A path is loaded with ``read_image`` (RGB) and remembered on the
                returned prediction so the overlay can be drawn at the page's own resolution.
            patches: Run the sliding-window path instead of downscaling the whole page -- the
                behaviour ``--patches`` was supposed to have and did not (see :meth:`predict_grid`).
            overlap: Sliding-window overlap, forwarded to :meth:`predict_grid`; it applies only
                when ``patches`` is set, which is why ``sbb infer --overlap`` needs ``--patches``.
        """
        path = None if isinstance(image, np.ndarray) else Path(image)
        page = np.asarray(image if path is None else read_image(path))

        # _forward returns a batch axis to index off; predict_grid returns the (n_classes, H, W)
        # field directly. The old code ran the whole-page pass, threw that result away when patches
        # was set, and still indexed [0] off the grid -- the page's first row, not its class axis.
        if patches:
            logits = self.predict_grid(page, overlap=overlap)
        else:
            logits = self._forward(self._to_tensor(page))[0]
        return self._make_prediction(page, logits, path)

    def predict_file(
        self, path: str | Path, *, patches: bool = False, overlap: float = 0.5
    ) -> Prediction:
        """``predict(path)`` -- the name the CLI's ``-i/--input`` loop calls per page."""
        return self.predict(path, patches=patches, overlap=overlap)

    def predict_grid(self, page: np.ndarray, *, overlap: float = 0.5) -> torch.Tensor:
        """Tiled inference: average the logits of overlapping ``input_height x input_width`` tiles.

        The legacy ``--patches`` branch contained only a comment and ``pass``, so asking for
        patch inference on a full-size page silently returned the single downscaled prediction.

        Args:
            page: ``(H, W, 3)`` RGB page, any size.
            overlap: Fraction between adjacent tiles; 0.5 gives each interior pixel the average of
                up to four tiles, which is what removes the seams a non-overlapping grid leaves.

        Returns:
            ``(n_classes, H, W)`` averaged logits at the page's resolution.
        """
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        window_h, window_w = self.config.input_height, self.config.input_width
        height, width = int(page.shape[0]), int(page.shape[1])
        step_h, step_w = (
            max(1, int(round(window_h * (1 - overlap)))),
            max(1, int(round(window_w * (1 - overlap)))),
        )
        rows = list(range(0, max(1, height - window_h + 1), step_h)) or [0]
        cols = list(range(0, max(1, width - window_w + 1), step_w)) or [0]

        accumulator: torch.Tensor | None = None
        weights: np.ndarray | None = None
        n_classes = self.config.n_classes
        for top in rows:
            for left in cols:
                crop = page[top : top + window_h, left : left + window_w]
                tile_logits = self._forward(self._to_tensor(crop))[0]
                tile_h, tile_w = crop.shape[0], crop.shape[1]
                if tuple(tile_logits.shape[-2:]) != (tile_h, tile_w):
                    tile_logits = F.interpolate(
                        tile_logits.unsqueeze(0).float(),
                        size=(tile_h, tile_w),
                        mode="bilinear",
                        align_corners=True,
                    )[0]
                if accumulator is None:
                    accumulator = torch.zeros(n_classes, height, width, dtype=torch.float32)
                    weights = np.zeros((height, width), dtype=np.float32)
                accumulator[:, top : top + tile_h, left : left + tile_w] += (
                    tile_logits.float().cpu()
                )
                weights[top : top + tile_h, left : left + tile_w] += 1.0
        assert accumulator is not None and weights is not None  # the grid always has one cell
        return accumulator / torch.from_numpy(np.maximum(weights, 1.0))[None]

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Resize then normalise the page and add the batch axis, exactly as training did."""
        resized = resize(image, self.config.input_height, self.config.input_width)
        normalized = normalize(resized)
        array = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0).to(self.device)

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tensor.to(self.device))

    def _class_names(self) -> list[str]:
        """Class id -> name, from the config's page2label mapping, else ``class_<id>``.

        The legacy code looked up ``config_params['classification_classes_name'][str(index)]``,
        a key the current schema does not define, so any classification checkpoint raised
        ``KeyError`` at predict time; page2label's own mapping is now the source, with a
        positional fallback so a config without one still answers.
        """
        mapping = self.config.page2label.class_ids() if self.config.page2label else {}
        return [
            ", ".join(sorted(mapping[index])) if mapping.get(index) else f"class_{index}"
            for index in range(self.config.n_classes)
        ]

    def _make_prediction(
        self, page: np.ndarray, logits: torch.Tensor, path: Path | None
    ) -> Prediction:
        """Turn one page's logits into the shape its task defines (mask, image, or label)."""
        task = self.config.task
        assert task is not None  # SBBConfig always carries a task
        if task in (TaskType.SEGMENTATION, TaskType.BINARIZATION):
            # cv2 wants numpy: the legacy code handed it a tensor and got cv2.error from
            # several frames away, which is why this boundary is explicit here.
            mask = resize_mask(logits.argmax(dim=0).cpu().numpy(), page.shape[0], page.shape[1])
            return Prediction(task=task, logits=logits, mask=mask, path=path)
        if task is TaskType.ENHANCEMENT:
            image = enhancement_image(logits.detach().cpu().numpy())
            return Prediction(task=task, logits=logits, image=image, path=path)

        # Classification and reading order answer per page, not per pixel: the logits are already
        # (n_classes,) and the label comes from the config's class mapping.
        probabilities = torch.softmax(logits.float(), dim=0)
        ranked = torch.sort(probabilities, descending=True)  # indices *and* values, together
        names = self._class_names()
        pairs = tuple(
            (
                names[int(index)] if int(index) < len(names) else f"class_{int(index)}",
                float(value),
            )
            for index, value in zip(ranked.indices, ranked.values, strict=True)
        )
        return Prediction(
            task=task,
            logits=logits,
            label=pairs[0][0] if pairs else None,
            probabilities=pairs,
            path=path,
        )
