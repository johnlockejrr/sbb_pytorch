"""Checkpoint persistence -- one schema, plus a reader tolerant of the three legacy layouts.

Before the refactor three layouts were in simultaneous use:

======================  ==========================================  ==========================
writer                  key                                         reader
======================  ==========================================  ==========================
``model_utils.save_model``  ``model_state``                       ``model_utils.load_model``
``train.py`` (final save)   ``model_state_dict``                  ``inference_pytorch.load_model``
``training.ModelCheckpoint``  ``model_state_dict``                -- (crashed before writing)
======================  ==========================================  ==========================

So a model trained with ``train.py`` could not be loaded by ``model_utils.load_model``, and
vice-versa. :meth:`Checkpoint.load` now accepts all three spellings (and a bare state dict),
so existing checkpoints keep working while everything new is written in one format.

Pruning also changed: ``sorted(glob("..._epoch_*.pth"))`` is lexicographic, so ``epoch_10``
sorted before ``epoch_2`` and ``keep_last_n`` deleted the newest files. Ordering is numeric now.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sbb_pytorch import __version__
from sbb_pytorch.exceptions import CheckpointError

__all__ = [
    "Checkpoint",
    "latest_checkpoint",
    "load_checkpoint",
    "load_into_model",
    "prune_checkpoints",
    "save_checkpoint",
]

#: Every key name historically used for the weights blob, in preference order.
_STATE_KEYS = ("model_state", "model_state_dict", "state_dict", "model")

_SCHEME = re.compile(r"(?P<stem>.*?)(?:[._-](?P<epoch>\d+))?(?:\.(?P<ext>pt|pth|ckpt))?$")


def _sidecar(path: Path) -> Path:
    """``model.pt`` -> ``model_config.json``: the JSON mirror written next to a checkpoint."""
    return path.with_name(f"{path.stem}_config.json")


@dataclass(slots=True)
class Checkpoint:
    """A saved training state.

    Attributes:
        model_state: ``model.state_dict()`` -- parameter name to tensor.
        epoch: Last completed epoch.
        step: Last completed optimiser step.
        task: The task the model was trained for (``"segmentation"``, ...), so inference can
            rebuild the right architecture without re-reading a config file from a fixed path.
        config: The resolved configuration, serialised form of
            :class:`sbb_pytorch.config.models.SBBConfig`.
        optimizer_state: ``optimizer.state_dict()`` if one was saved.
        metrics: Final metrics, e.g. ``{"val_loss": 0.31, "val_iou": 0.78}``.
        package_version: Version of this package that produced the file.
        schema_version: On-disk layout version; currently ``1``.
    """

    model_state: dict[str, Tensor]
    epoch: int = 0
    step: int = 0
    task: str | None = None
    config: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    package_version: str = __version__
    schema_version: int = 1

    # -------------------------------------------------------------- construction

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Checkpoint:
        """Build a checkpoint from a loaded dict, accepting the legacy key spellings.

        Accepts ``model_state``, ``model_state_dict``, ``state_dict``, ``model``, or a bare
        ``{name: tensor}`` mapping with no wrapper at all.

        Raises:
            CheckpointError: if no weights can be located in ``raw``.
        """
        state: dict[str, Tensor] | None = None
        for key in _STATE_KEYS:
            value = raw.get(key)
            if isinstance(value, dict):
                state = value
                break
        # A bare state_dict: every value is a tensor, with no wrapper key at all.
        if state is None and raw and all(isinstance(v, Tensor) for v in raw.values()):
            state = dict(raw)
        if state is None:
            raise CheckpointError(
                f"no model weights found; expected one of {list(_STATE_KEYS)} or a bare "
                f"state_dict, got keys {sorted(raw)[:8]}"
            )

        metrics = raw.get("metrics")
        if not isinstance(metrics, dict):
            # train.py used to store val_loss/val_acc as loose top-level scalars.
            metrics = {k: v for k, v in raw.items() if isinstance(v, int | float) and k != "epoch"}

        return cls(
            model_state=state,
            epoch=int(raw.get("epoch") or 0),
            step=int(raw.get("step") or 0),
            task=raw.get("task"),
            config=raw.get("config") if isinstance(raw.get("config"), dict) else None,
            optimizer_state=raw.get("optimizer_state") or raw.get("optimizer_state_dict"),
            metrics={str(k): float(v) for k, v in metrics.items() if isinstance(v, int | float)},
            package_version=str(raw.get("package_version", "unknown")),
            schema_version=int(raw.get("schema_version", 0)),
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs: Any,
    ) -> Checkpoint:
        """Capture the current state of ``model`` (and optionally ``optimizer``)."""
        return cls(model_state=dict(model.state_dict()), **kwargs)

    # ------------------------------------------------------------------ i/o

    def to_dict(self) -> dict[str, Any]:
        """The exact payload written to disk. ``model_state`` is emitted under the canonical key.

        Built by hand rather than with ``dataclasses.asdict``, which deep-copies every tensor
        (doubling peak memory on a multi-gigabyte checkpoint) and recurses inconsistently
        across torch versions.
        """
        return {
            "model_state": self.model_state,
            "epoch": self.epoch,
            "step": self.step,
            "task": self.task,
            "config": self.config,
            "optimizer_state": self.optimizer_state,
            "metrics": self.metrics,
            "package_version": self.package_version,
            "schema_version": self.schema_version,
        }

    def save(self, path: str | Path, *, write_sidecar: bool = True) -> Path:
        """Serialise to ``path`` (``.pt``/``.pth``), creating parent directories.

        When ``write_sidecar`` is set and a config is attached, a ``<stem>_config.json`` mirror
        is written next to the weights (``train.py`` used to write one, and external scripts
        read it without unpickling). Set ``write_sidecar=False`` for per-epoch checkpoints.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.to_dict(), path)
        except (OSError, RuntimeError, pickle.PickleError) as exc:  # pragma: no cover
            raise CheckpointError(f"failed to write {path}: {exc}") from exc
        if write_sidecar and self.config is not None:
            _sidecar(path).write_text(
                json.dumps(self.config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
        """Read a checkpoint written by this package, or by any pre-refactor module.

        ``torch.load`` is first attempted with ``weights_only=True`` (safe against arbitrary
        code execution in untrusted files) and retried with ``weights_only=False`` for legacy
        pickles whose ``config`` holds non-primitive objects.

        Raises:
            CheckpointError: the file is missing, unreadable, or contains no weights.
        """
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"no checkpoint at {path}")
        try:
            raw = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            try:
                raw = torch.load(path, map_location=map_location, weights_only=False)
            except Exception as exc:
                raise CheckpointError(f"could not load {path}: {exc}") from exc

        if isinstance(raw, Checkpoint):
            return raw
        if not isinstance(raw, dict):
            raise CheckpointError(f"{path} contains {type(raw).__name__}, expected a dict")
        return cls.from_dict(raw)

    def apply_to(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        strict: bool = True,
    ) -> None:
        """Load these weights into ``model`` (and ``optimizer``), with a precise error on mismatch.

        Raises:
            CheckpointError: on any key mismatch when ``strict`` -- rather than the bare
                ``RuntimeError`` the legacy callers surfaced, which usually just said
                ``size mismatch for <layer>`` without saying which checkpoint it came from.
        """
        try:
            result = model.load_state_dict(self.model_state, strict=strict)
        except RuntimeError as exc:
            raise CheckpointError(
                f"weights incompatible with {type(model).__name__}: {exc}"
            ) from exc
        # strict=False never raises; report what it skipped so a partial load is still visible.
        if not strict and (result.missing_keys or result.unexpected_keys):
            raise CheckpointError(
                f"{type(model).__name__}: {len(result.missing_keys)} missing and "
                f"{len(result.unexpected_keys)} unexpected keys; "
                f"first missing: {result.missing_keys[:3] or 'none'}"
            )

        if optimizer is not None and self.optimizer_state:
            try:
                optimizer.load_state_dict(self.optimizer_state)
            except (ValueError, KeyError) as exc:
                raise CheckpointError(f"optimizer state incompatible: {exc}") from exc


# ------------------------------------------------------------------ module API


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    task: str | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Build a :class:`Checkpoint` from ``model`` and write it to ``path``."""
    return Checkpoint(
        model_state=dict(model.state_dict()),
        epoch=epoch,
        step=step,
        task=task,
        config=config,
        optimizer_state=None if optimizer is None else optimizer.state_dict(),
        metrics=dict(metrics or {}),
    ).save(path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Read a checkpoint (see :meth:`Checkpoint.load`)."""
    return Checkpoint.load(path, map_location=map_location)


def load_into_model(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Checkpoint:
    """Convenience: read ``path`` and apply it to ``model`` in one call."""
    checkpoint = Checkpoint.load(path, map_location=map_location)
    checkpoint.apply_to(model, optimizer=optimizer, strict=strict)
    return checkpoint


def latest_checkpoint(directory: str | Path, pattern: str | None = None) -> Path | None:
    """Newest checkpoint in ``directory`` by modification time, or ``None`` if there is none.

    With no ``pattern`` all three extensions in circulation are considered (``.pt``, ``.pth``,
    ``.ckpt``); pass one to restrict the search. ``mtime`` is used rather than the legacy
    ``ctime``, which tracks inode changes rather than content changes and reports the wrong
    file after a copy or a ``git checkout``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    patterns = [pattern] if pattern else ["*.pt", "*.pth", "*.ckpt"]
    candidates = {p for glob in patterns for p in directory.glob(glob) if p.is_file()}
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prune_checkpoints(directory: str | Path, keep_last_n: int = 5, prefix: str = "") -> list[Path]:
    """Delete older checkpoints, keeping the ``keep_last_n`` newest. Returns deleted paths.

    Ordering is numeric on the embedded epoch number when one exists (so ``epoch_10`` is newer
    than ``epoch_2``), falling back to modification time. The legacy implementation sorted the
    filenames as strings, which kept ``epoch_1.pth`` and deleted ``epoch_2.pth``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    glob = f"{prefix}*.pt" if prefix else "*.pt"
    files = [p for p in directory.glob(glob) if p.is_file()]
    for extra in (".pth", ".ckpt"):
        files += [p for p in directory.glob(f"{prefix}*{extra}") if p.is_file()]
    files = sorted(set(files))
    if len(files) <= keep_last_n:
        return []

    def sort_key(path: Path) -> tuple[int, float]:
        match = _SCHEME.match(path.name)
        epoch = int(match.group("epoch")) if match and match.group("epoch") else -1
        return epoch, path.stat().st_mtime

    ordered = sorted(files, key=sort_key)
    deleted = ordered[:-keep_last_n]
    for path in deleted:
        path.unlink()
        if _sidecar(path).exists():
            _sidecar(path).unlink()
    return deleted


def write_sidecar_config(path: str | Path, config: dict[str, Any]) -> Path:
    """Write ``config`` as JSON next to a checkpoint, for tooling that inspects it directly.

    The config is also stored *inside* the checkpoint; this sidecar exists because the legacy
    ``train.py`` wrote one and external scripts read it without unpickling the weights.
    """
    sidecar = _sidecar(Path(path))
    sidecar.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar
