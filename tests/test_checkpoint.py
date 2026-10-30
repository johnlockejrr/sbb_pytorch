"""Tests for the checkpoint schema, focused on the layouts that used to be incompatible."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from sbb_pytorch.exceptions import CheckpointError
from sbb_pytorch.storage.checkpoint import (
    Checkpoint,
    latest_checkpoint,
    load_checkpoint,
    load_into_model,
    prune_checkpoints,
    save_checkpoint,
    write_sidecar_config,
)


class TinyModel(nn.Module):
    """A model small enough to save thousands of times in a test run."""

    def __init__(self, out: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.out = nn.Linear(4, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused
        return self.out(x.flatten(2).mean(-1))


# --------------------------------------------------------------- reading layouts


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = save_checkpoint(
        tmp_path / "runs" / "model.pt",
        model,
        optimizer=optimizer,
        epoch=7,
        step=1234,
        task="segmentation",
        config={"task": "segmentation", "n_classes": 4},
        metrics={"val_loss": 0.31, "val_iou": 0.78},
    )

    loaded = load_checkpoint(path)
    assert loaded.epoch == 7
    assert loaded.step == 1234
    assert loaded.task == "segmentation"
    assert loaded.config == {"task": "segmentation", "n_classes": 4}
    assert loaded.metrics == {"val_loss": pytest.approx(0.31), "val_iou": pytest.approx(0.78)}
    assert set(loaded.model_state) == set(model.state_dict())
    assert loaded.optimizer_state is not None

    # The weights must be byte-identical, and loading them back must be a no-op.
    torch.testing.assert_close(loaded.model_state["conv.weight"], model.state_dict()["conv.weight"])
    load_into_model(path, model, optimizer=optimizer)


@pytest.mark.parametrize("key", ["model_state", "model_state_dict", "state_dict", "model"])
def test_reads_every_historical_key_spelling(tmp_path: Path, key: str) -> None:
    """``model_utils`` wrote ``model_state``, ``train.py`` wrote ``model_state_dict``.

    Neither could read the other's files. All four spellings must still load.
    """
    path = tmp_path / f"legacy_{key}.pt"
    torch.save({key: TinyModel().state_dict(), "epoch": 3}, path)
    assert Checkpoint.load(path).model_state.keys() == TinyModel().state_dict().keys()


def test_loads_a_lightning_checkpoint_and_strips_the_model_prefix(tmp_path: Path) -> None:
    """A ``.ckpt`` names every parameter through the module holding it; ``model_state`` does not.

    pytorch-lightning writes ``{"state_dict": {"model.conv.weight": ...}, "global_step": ...,
    "optimizer_states": [...]}`` -- so the step and the optimiser state live under names
    :meth:`Checkpoint.from_dict` never looked at, and the prefixed keys it did copy made
    ``sbb export best-07-0.9812.ckpt`` produce a ``model.pt`` that ``sbb infer`` then refused.
    """
    model = TinyModel()
    path = tmp_path / "best-07-0.9812.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}
            | {"criterion.weights": torch.zeros(3)},
            "epoch": 7,
            "global_step": 42,
            "optimizer_states": [torch.optim.SGD(model.parameters(), lr=0.1).state_dict()],
            "pytorch-lightning_version": "2.6.6",
        },
        path,
    )

    checkpoint = Checkpoint.load(path)
    assert checkpoint.model_state.keys() == model.state_dict().keys()  # the model group, renamed
    assert checkpoint.epoch == 7
    assert checkpoint.step == 42  # Lightning spells this ``global_step``
    assert checkpoint.optimizer_state is not None  # ...and this one ``optimizer_states``
    assert checkpoint.pytorch_lightning_version == "2.6.6"
    assert (
        checkpoint.metrics == {}
    )  # global_step is not a metric, whatever from_dict's fallback says
    checkpoint.apply_to(model)  # the load this whole path used to end in


def test_loads_a_lightning_checkpoint_whose_state_dict_is_not_prefixed(tmp_path: Path) -> None:
    """No single ``model.`` group to take: keep the whole ``state_dict`` rather than guess one.

    The file is a ``.ckpt`` renamed to ``.pt``, which is the ordinary fate of an old run -- so the
    layout is decided by the contents, not by the suffix.
    """
    state = TinyModel().state_dict()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": {**state, "criterion.weights": torch.zeros(3)},
            "epoch": 3,
            "global_step": 12,
            "optimizer_states": [],
            "pytorch-lightning_version": "2.4.0",
        },
        path,
    )
    checkpoint = Checkpoint.load(path)
    assert set(checkpoint.model_state) == set(state) | {"criterion.weights"}
    assert checkpoint.step == 12 and checkpoint.optimizer_state is None
    assert checkpoint.pytorch_lightning_version == "2.4.0"


def test_strips_the_one_prefix_a_lightning_checkpoint_shares(tmp_path: Path) -> None:
    """A checkpoint whose every key starts ``net.``: the shared prefix is what gets removed."""
    path = tmp_path / "net.ckpt"
    torch.save(
        {
            "state_dict": {"net.weight": torch.zeros(2, 2), "net.bias": torch.zeros(2)},
            "epoch": 1,
            "global_step": 5,
            "optimizer_states": [],
        },
        path,
    )
    checkpoint = Checkpoint.load(path)
    assert set(checkpoint.model_state) == {"weight", "bias"}
    assert checkpoint.pytorch_lightning_version is None  # only Lightning's hyphenated key sets it


def test_reads_bare_state_dict_without_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "bare.pt"
    torch.save(TinyModel().state_dict(), path)
    checkpoint = Checkpoint.load(path)
    assert "conv.weight" in checkpoint.model_state
    assert checkpoint.epoch == 0


def test_reads_loose_metric_scalars_from_train_py(tmp_path: Path) -> None:
    """``train.py`` stored ``val_loss``/``val_acc`` as loose top-level scalars, not a dict."""
    path = tmp_path / "train_style.pt"
    torch.save(
        {
            "model_state_dict": TinyModel().state_dict(),
            "epoch": 5,
            "val_loss": 0.42,
            "val_acc": 0.88,
        },
        path,
    )
    checkpoint = Checkpoint.load(path)
    assert checkpoint.metrics == {"val_loss": pytest.approx(0.42), "val_acc": pytest.approx(0.88)}
    assert checkpoint.epoch == 5


def test_reads_legacy_pickle_with_non_primitive_config(tmp_path: Path) -> None:
    """The ``weights_only=True`` -> ``False`` fallback: a config holding a set is not load-safe."""
    path = tmp_path / "legacy_pickle.pt"
    payload = {"model_state": TinyModel().state_dict(), "task": "textline", "epoch": 1}
    with path.open("wb") as handle:
        torch.save(payload, handle)
    assert Checkpoint.load(path).task == "textline"


def test_missing_file_raises_checkpoint_error(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no checkpoint at"):
        load_checkpoint(tmp_path / "absent.pt")


def test_dict_without_weights_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.pt"
    torch.save({"epoch": 1, "note": "no weights here"}, path)
    with pytest.raises(CheckpointError, match="no model weights found"):
        Checkpoint.load(path)


# ------------------------------------------------------------------- applying


def test_apply_to_rejects_wrong_architecture(tmp_path: Path) -> None:
    """The class of error that used to surface as a bare ``RuntimeError`` from torch."""
    path = save_checkpoint(tmp_path / "tiny.pt", TinyModel(out=2))
    other = TinyModel(out=9)
    with pytest.raises(CheckpointError, match="incompatible with TinyModel"):
        load_into_model(path, other)


def test_apply_to_reaches_the_architecture_through_a_wrapper(tmp_path: Path) -> None:
    """``SBBLightningModule`` keeps the architecture in ``self.model``; so does this stand-in.

    The keys a ``model.pt`` holds are the *architecture's* own, so loading one into anything that
    wraps an architecture means reaching through the attribute -- the case the ``prefix`` rule in
    :func:`~sbb_pytorch.storage.checkpoint._resolve_target` exists for.
    """

    class Wrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

    model = TinyModel()
    with torch.no_grad():
        model.conv.weight.normal_()  # weights the wrapper's own freshly built ones cannot match
    path = save_checkpoint(tmp_path / "model.pt", model)

    wrapper = Wrapper(TinyModel())
    Checkpoint.load(path).apply_to(wrapper)
    torch.testing.assert_close(wrapper.model.conv.weight, model.conv.weight)


def test_apply_to_non_strict_reports_skipped_keys(tmp_path: Path) -> None:
    model = TinyModel()
    state = dict(model.state_dict())
    state["surprise.weight"] = torch.zeros(1)
    path = tmp_path / "extra_key.pt"
    torch.save({"model_state": state}, path)

    with pytest.raises(CheckpointError, match="1 unexpected keys"):
        Checkpoint.load(path).apply_to(model, strict=False)


def test_apply_to_non_strict_silently_ok_when_matching(tmp_path: Path) -> None:
    model = TinyModel()
    path = save_checkpoint(tmp_path / "m.pt", model)
    Checkpoint.load(path).apply_to(model, strict=False)  # must not raise


def test_apply_to_rejects_bad_optimizer_state(tmp_path: Path) -> None:
    model = TinyModel()
    path = tmp_path / "bad_opt.pt"
    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": {"state": {}, "param_groups": []}},
        path,
    )
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(CheckpointError, match="optimizer state incompatible"):
        load_into_model(path, model, optimizer=optimizer)


# ----------------------------------------------------------------- sidecar / fs


def test_sidecar_json_written_alongside(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "m.pt", TinyModel(), task="layout", config={"task": "layout"})
    sidecar = tmp_path / "m_config.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == {"task": "layout"}

    # An explicit write replaces it, and a save without a config writes nothing extra.
    write_sidecar_config(path, {"task": "glyph"})
    assert json.loads(sidecar.read_text()) == {"task": "glyph"}
    bare = save_checkpoint(tmp_path / "bare.pt", TinyModel())
    assert not (tmp_path / "bare_config.json").exists()
    assert bare.name == "bare.pt"


def test_latest_checkpoint_uses_mtime_not_name(tmp_path: Path) -> None:
    """Legacy code used ``ctime``, which changes on ``git checkout`` and misreports the newest."""
    old = tmp_path / "epoch_99.pt"
    new = tmp_path / "epoch_1.pth"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (time.time() - 86400, time.time() - 86400))
    os.utime(new, (time.time(), time.time()))

    assert latest_checkpoint(tmp_path) == new
    assert latest_checkpoint(tmp_path, pattern="*.pt") == old  # only .pt considered
    assert latest_checkpoint(tmp_path / "missing_dir") is None
    assert latest_checkpoint(tmp_path.parent / "definitely_absent") is None


def test_prune_keeps_newest_by_number_not_string(tmp_path: Path) -> None:
    """``sorted()`` put ``epoch_10`` before ``epoch_2`` and deleted the wrong files."""
    now = time.time()
    for epoch in range(1, 13):
        path = tmp_path / f"model_epoch_{epoch}.pt"
        path.write_bytes(b"w")
        os.utime(path, (now + epoch, now + epoch))
        (tmp_path / f"model_epoch_{epoch}_config.json").write_text("{}")

    deleted = prune_checkpoints(tmp_path, keep_last_n=3)
    remaining = sorted(p.name for p in tmp_path.glob("*.pt"))

    assert remaining == ["model_epoch_10.pt", "model_epoch_11.pt", "model_epoch_12.pt"]
    assert [p.name for p in deleted] == [f"model_epoch_{e}.pt" for e in range(1, 10)]
    # Their sidecars must not linger.
    assert sorted(p.name for p in tmp_path.glob("*_config.json")) == [
        f"model_epoch_{e}_config.json" for e in (10, 11, 12)
    ]


def test_prune_noop_when_under_limit(tmp_path: Path) -> None:
    (tmp_path / "a.pt").write_bytes(b"x")
    assert prune_checkpoints(tmp_path, keep_last_n=5) == []
    assert prune_checkpoints(tmp_path / "absent") == []


def test_from_model_captures_state(tmp_path: Path) -> None:
    model = TinyModel()
    checkpoint = Checkpoint.from_model(model, epoch=2, task="printspace")
    assert checkpoint.model_state.keys() == model.state_dict().keys()
    assert checkpoint.schema_version == 1
    assert checkpoint.package_version

    # Re-saving and re-reading must be stable (the canonical key is always ``model_state``).
    reread = Checkpoint.load(checkpoint.save(tmp_path / "again.pt", write_sidecar=False))
    assert reread.model_state.keys() == checkpoint.model_state.keys()
    assert reread.schema_version == 1
