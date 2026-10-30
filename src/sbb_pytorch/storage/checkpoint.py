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

A fourth layout comes from pytorch-lightning rather than from the legacy scripts: a ``.ckpt``
holds the module's weights under ``state_dict`` with every key prefixed by the attribute that
holds the model (``model.conv.weight``), the optimiser step under ``global_step``, and the
optimiser states under the plural ``optimizer_states``. :meth:`Checkpoint.from_lightning` reads
that shape as well -- which is what lets ``sbb export best-07-0.9812.ckpt`` write a file
``sbb infer`` will accept. Before it, the export copied the prefixed keys into ``model.pt`` and
the very next load rejected them as ``Unexpected key(s) "model.conv.weight"``.

Pruning also changed: ``sorted(glob("..._epoch_*.pth"))`` is lexicographic, so ``epoch_10``
sorted before ``epoch_2`` and ``keep_last_n`` deleted the newest files. Ordering is numeric now.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Iterable, Mapping
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

#: The key pytorch-lightning stamps on every checkpoint it writes. Ours is spelled
#: ``pytorch_lightning_version``; the hyphen is Lightning's own and is what identifies the file.
_LIGHTNING_VERSION_KEY = "pytorch-lightning_version"


def _is_lightning(raw: dict[str, Any]) -> bool:
    """Whether *raw* is pytorch-lightning's layout, whatever the file happens to be called.

    Content rather than suffix, because ``model.ckpt`` renamed to ``model.pt`` is the ordinary
    thing an old run ends up as and ``torch.load`` cannot tell the two apart afterwards. Either
    marker is enough: Lightning's own dumps always carry ``pytorch-lightning_version``, while a
    trimmed or older file still pairs ``global_step`` with the plural ``optimizer_states``.
    Neither appears in this package's schema, whose fields are ``step`` and ``optimizer_state``.
    """
    return _LIGHTNING_VERSION_KEY in raw or ("global_step" in raw and "optimizer_states" in raw)


def _lightning_state(raw: dict[str, Any]) -> dict[str, Tensor]:
    """The architecture's tensors out of a Lightning ``state_dict``, with the prefix removed.

    Lightning keys everything through the attribute that holds the model, so
    :class:`~sbb_pytorch.training.module.SBBLightningModule`'s parameters come out as
    ``model.conv.weight`` while this package's ``model_state`` holds the *architecture's* own keys
    (``conv.weight``) -- which is exactly what ``fit`` writes into ``model.pt``. Taking
    ``state_dict`` whole is the ``Unexpected key(s) "model.conv.weight"`` that used to make an
    exported checkpoint unloadable, so the ``model.`` group is what gets returned, renamed.

    Order of attempts, each one all-or-nothing so a partial match is never taken quietly: the
    ``model.`` group; one prefix shared by every key, stripped; the ``state_dict`` verbatim.
    """
    state = {k: v for k, v in (raw.get("state_dict") or {}).items() if isinstance(v, Tensor)}
    if not state:
        return {}
    if any(key.startswith("model.") for key in state):
        return {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    heads = {key.split(".", 1)[0] for key in state}
    if len(heads) == 1 and "" not in heads:
        (head,) = heads
        return {k[len(head) + 1 :]: v for k, v in state.items()}
    return dict(state)


def _modules(target: nn.Module, keys: Iterable[str]) -> list[tuple[nn.Module, str]]:
    """Every module *keys* could belong to, with the prefix they would carry.

    ``target`` itself, then each of its child modules under ``"<child_name>."`` -- the shape
    pytorch-lightning writes (``SBBLightningModule.model`` gives ``model.conv.weight``) and the
    shape ``fit`` saves ``model.pt`` from -- then ``target`` again under each leading segment the
    keys themselves use, which is how a ``.ckpt``'s ``model.`` prefix is found when the weights are
    being loaded into a bare architecture rather than into the module that saved them. No
    architecture in :mod:`sbb_pytorch.models` has a ``model`` attribute of its own
    (``grep -rn "self\\.model" src/sbb_pytorch/models/*.py`` is empty), so a prefixed match can
    only fire for a wrapper or for a file that really was written with that prefix.
    """
    found: list[tuple[nn.Module, str]] = [(target, "")]
    found += [(module, f"{name}.") for name, module in target.named_children()]
    heads = sorted({key.split(".", 1)[0] for key in keys if "." in key})
    found += [(target, f"{head}.") for head in heads]
    return found


def _spellings(state: dict[str, Tensor], prefix: str) -> list[dict[str, Tensor]]:
    """The ways *state* can be spelled for *prefix*: as it is, prefixed, or un-prefixed.

    One prefix, all keys at once. Renaming some keys and not others is how a checkpoint belonging
    to another architecture quietly loses weight, so a partial match is never taken.
    """
    if not prefix:
        return [state]
    if all(key.startswith(prefix) for key in state):
        return [state, {key[len(prefix) :]: value for key, value in state.items()}]
    return [state, {f"{prefix}{key}": value for key, value in state.items()}]


def _resolve_target(
    state: Mapping[str, Tensor], target: nn.Module
) -> tuple[nn.Module, dict[str, Tensor], str]:
    """The module *state* belongs in, the keys to hand it, and the rule that decided so.

    One policy, written once, covering every source a :class:`Checkpoint` can come from:

    * ``"exact"`` -- the keys are ``target``'s own, so the weights go where they were written from;
    * ``"prefix"`` -- they are a child module's keys under one prefix, which is the shape a
      pytorch-lightning ``.ckpt`` keeps them in for its ``self.model`` and the shape ``fit``
      writes ``model.pt`` from, so loading either into a ``SBBLightningModule`` resolves here;
    * ``"as-is"`` -- nothing matched, so ``load_state_dict`` receives the original keys and
      produces torch's own missing/unexpected report instead of a guess being made first.
    """
    keys = dict(state)
    for module, prefix in _modules(target, keys):
        for spelling in _spellings(keys, prefix):
            if set(spelling) == set(module.state_dict()):
                return module, spelling, "exact" if not prefix else "prefix"
    return target, keys, "as-is"


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
        pytorch_lightning_version: The pytorch-lightning version a ``.ckpt`` was written by, when
            the checkpoint came from one. ``package_version`` answers for *this* package -- a
            Lightning file was not produced by it, which is why ``sbb export`` re-stamps it -- so
            a foreign producer gets its own field instead of borrowing that one.
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
    pytorch_lightning_version: str | None = None

    # -------------------------------------------------------------- construction

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Checkpoint:
        """Build a checkpoint from a loaded dict, accepting the legacy key spellings.

        Accepts ``model_state``, ``model_state_dict``, ``state_dict``, ``model``, or a bare
        ``{name: tensor}`` mapping with no wrapper at all. A dict that is recognisably
        pytorch-lightning's own layout goes to :meth:`from_lightning` first -- see
        :func:`_is_lightning` for how the two are told apart.

        Raises:
            CheckpointError: if no weights can be located in ``raw``.
        """
        if _is_lightning(raw):
            return cls.from_lightning(raw)
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
        lightning_version = raw.get("pytorch_lightning_version")

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
            pytorch_lightning_version=(
                str(lightning_version) if isinstance(lightning_version, str) else None
            ),
        )

    @classmethod
    def from_lightning(cls, raw: dict[str, Any]) -> Checkpoint:
        """Read pytorch-lightning's own checkpoint layout into this schema.

        A ``.ckpt`` from ``Trainer.save_checkpoint`` or ``ModelCheckpoint`` is not our shape. Its
        ``state_dict`` holds the *LightningModule's* state, so every parameter is named through
        the attribute that holds it (``model.conv.weight``), the step is ``global_step`` rather
        than ``step``, and the optimisers are the plural ``optimizer_states`` -- a list with one
        entry per optimiser -- rather than ``optimizer_state``. Reading it with :meth:`from_dict`
        alone copied those prefixed keys straight into ``model.pt``, which the next
        :meth:`apply_to` then rejected as ``Unexpected key(s) "model.conv.weight"``, so
        ``sbb export best-07-0.9812.ckpt`` produced a file ``sbb infer`` could not load.

        ``task`` and ``config`` stay ``None`` here for the reason they stay ``None`` for any
        pre-schema file: ``SBBLightningModule`` does not call ``save_hyperparameters`` (see
        migration 3.6), so a Lightning checkpoint carries no configuration, and ``sbb infer``
        still needs ``--config`` after such an export. ``metrics`` stays empty rather than
        harvesting loose scalars the way :meth:`from_dict` does for ``train.py``'s files:
        ``global_step`` is not a metric, and Lightning keeps epoch metrics in ``callback_metrics``,
        which never reach the file.
        """
        optimizers = raw.get("optimizer_states")
        if isinstance(optimizers, list | tuple):
            optimizer_state = optimizers[0] if optimizers else None
        elif isinstance(optimizers, dict):  # a trimmed file that kept one state instead of a list
            optimizer_state = optimizers
        else:
            optimizer_state = None
        version = raw.get(_LIGHTNING_VERSION_KEY)
        return cls(
            model_state=_lightning_state(raw),
            epoch=int(raw.get("epoch") or 0),
            step=int(raw.get("step", raw.get("global_step")) or 0),
            optimizer_state=optimizer_state,
            package_version=str(raw.get("package_version", "unknown")),
            schema_version=int(raw.get("schema_version", 1)),
            pytorch_lightning_version=str(version) if isinstance(version, str) else None,
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
            "pytorch_lightning_version": self.pytorch_lightning_version,
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

        The target is chosen by :func:`_resolve_target`: the architecture ``model`` itself when the
        keys are its own, its ``model`` child when they carry the ``model.`` prefix a
        pytorch-lightning checkpoint writes, and ``model`` unchanged so torch's own report is what
        the caller sees when nothing matches. The rule used is named in the error, because "which
        of these two files is this?" is the question a mismatch really asks.

        Raises:
            CheckpointError: on any key mismatch when ``strict`` -- rather than the bare
                ``RuntimeError`` the legacy callers surfaced, which usually just said
                ``size mismatch for <layer>`` without saying which checkpoint it came from.
        """
        module, state, rule = _resolve_target(self.model_state, model)
        detail = "" if rule == "exact" else f" (matched by the {rule} rule)"
        try:
            result = module.load_state_dict(state, strict=strict)
        except RuntimeError as exc:
            raise CheckpointError(
                f"weights incompatible with {type(model).__name__}: {exc}{detail}"
            ) from exc
        # strict=False never raises; report what it skipped so a partial load is still visible.
        if not strict and (result.missing_keys or result.unexpected_keys):
            raise CheckpointError(
                f"{type(model).__name__}: {len(result.missing_keys)} missing and "
                f"{len(result.unexpected_keys)} unexpected keys; "
                f"first missing: {result.missing_keys[:3] or 'none'}{detail}"
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
