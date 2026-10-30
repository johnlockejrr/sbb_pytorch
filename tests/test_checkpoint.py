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
    safetensors_available,
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


# ------------------------------------------------------------------ safetensors

needs_safetensors = pytest.mark.skipif(
    not safetensors_available(), reason="the `safetensors` package is not installed"
)


@needs_safetensors
def test_safetensors_extension_and_metadata_round_trip(tmp_path: Path) -> None:
    """Every field but ``optimizer_state`` travels in the file's ``__metadata__`` string."""
    model = TinyModel()
    checkpoint = Checkpoint(
        model_state=dict(model.state_dict()),
        epoch=7,
        step=42,
        task="segmentation",
        config={"task": "segmentation", "n_classes": 2},
        optimizer_state={"state": {"momentum_buffer": torch.zeros(1)}},
        metrics={"val_loss": 0.31},
        pytorch_lightning_version="2.6.6",
    )
    path = checkpoint.save_safetensors(tmp_path / "model.safetensors")

    import safetensors.torch

    with safetensors.torch.safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    assert metadata["format"] == "pt"
    assert metadata["sbb.epoch"] == "7"
    assert metadata["sbb.step"] == "42"
    assert metadata["sbb.task"] == "segmentation"
    assert json.loads(metadata["sbb.config"]) == {"task": "segmentation", "n_classes": 2}
    assert json.loads(metadata["sbb.metrics"]) == {"val_loss": 0.31}
    assert metadata["sbb.pytorch_lightning_version"] == "2.6.6"

    loaded = Checkpoint.load(path)
    assert loaded.model_state.keys() == model.state_dict().keys()
    assert torch.equal(loaded.model_state["conv.weight"], model.conv.weight)
    assert (loaded.epoch, loaded.step, loaded.task) == (7, 42, "segmentation")
    assert loaded.config == checkpoint.config
    assert loaded.metrics == checkpoint.metrics
    assert loaded.pytorch_lightning_version == "2.6.6"
    # An optimiser state is a tensor tree, not a string: the one field the format cannot carry.
    assert loaded.optimizer_state is None


@needs_safetensors
def test_save_dispatches_on_the_extension(tmp_path: Path) -> None:
    """``save`` writes the format the name asks for, and only a torch file gets a sidecar."""
    checkpoint = Checkpoint(
        model_state=dict(TinyModel().state_dict()),
        task="layout",
        config={"task": "layout"},
    )

    torch_path = checkpoint.save(tmp_path / "nested" / "model.pt")
    assert torch.load(torch_path, weights_only=True)["model_state"].keys()
    assert (torch_path.parent / "model_config.json").exists()  # the mirror external scripts read

    safetensors_path = checkpoint.save(tmp_path / "model.safetensors")
    assert safetensors_path.suffix == ".safetensors"
    assert not (tmp_path / "model_config.json").exists()  # its config travels inside the file
    assert Checkpoint.load(safetensors_path).config == checkpoint.config

    # Asking for the sidecar explicitly still writes it, and a .pth stays a pickle.
    assert checkpoint.save(safetensors_path, write_sidecar=True).with_name("model_config.json")
    assert torch.load(checkpoint.save(tmp_path / "model.pth"), weights_only=True)["model_state"]


@needs_safetensors
def test_safetensors_refuses_an_empty_state_and_reads_a_foreign_file(tmp_path: Path) -> None:
    """A file with no tensors would load as a checkpoint with no weights, so it is never written."""
    import safetensors.torch

    with pytest.raises(CheckpointError, match="empty model_state"):
        Checkpoint(model_state={}).save_safetensors(tmp_path / "empty.safetensors")

    # A file from another tool carries none of our metadata; it loads as unversioned and
    # unattributed rather than being rejected for the missing keys.
    safetensors.torch.save_file({"w": torch.ones(2, 2)}, str(tmp_path / "foreign.safetensors"))
    foreign = Checkpoint.load(tmp_path / "foreign.safetensors")
    assert foreign.schema_version == 0
    assert foreign.package_version == "unknown"
    assert foreign.task is None
    assert foreign.config is None
    assert foreign.metrics == {}
    assert list(foreign.model_state) == ["w"]

    # Tied weights -- two names, one storage -- are legal in a pickle and refused by safetensors.
    shared = torch.zeros(3)
    tied = Checkpoint(model_state={"a": shared, "b": shared}).save_safetensors(
        tmp_path / "tied.safetensors"
    )
    loaded = Checkpoint.load(tied)
    assert {value.data_ptr() for value in loaded.model_state.values()} != ({shared.data_ptr()})
    assert torch.equal(loaded.model_state["a"], loaded.model_state["b"])


def test_save_safetensors_without_the_dependency_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``.safetensors`` request is refused when the package is missing (no skipif)."""
    monkeypatch.setattr("sbb_pytorch.storage.checkpoint.safetensors_available", lambda: False)
    checkpoint = Checkpoint(model_state=dict(TinyModel().state_dict()))
    with pytest.raises(CheckpointError, match="needs the `safetensors` package"):
        checkpoint.save(tmp_path / "model.safetensors")
    # A torch save is unaffected: the guard sits on the safetensors path alone.
    assert checkpoint.save(tmp_path / "model.pt", write_sidecar=False).exists()


def test_latest_checkpoint_ranks_model_pt_above_every_other_extension(tmp_path: Path) -> None:
    """``model.pt`` is the checkpoint; the epoch files and other formats are its history."""
    newest: dict[str, float] = {}
    for name in (
        "model.pt",
        "model.pth",
        "model.ckpt",
        "model.safetensors",
        "run-12.pt",
        "run-13.pth",
        "run-14.ckpt",
        "run-15.safetensors",
    ):
        path = tmp_path / name
        path.write_bytes(b"x")
        newest[name] = time.time() + 100  # every other file is newer than model.pt
    os.utime(tmp_path / "model.pt", (time.time(), time.time()))
    for name, stamp in newest.items():
        if name != "model.pt":
            os.utime(tmp_path / name, (stamp, stamp))

    assert latest_checkpoint(tmp_path) == tmp_path / "model.pt"
    assert latest_checkpoint(tmp_path, pattern="*.safetensors") == tmp_path / "model.safetensors"


def test_latest_checkpoint_honours_prefer(tmp_path: Path) -> None:
    """``prefer`` (what ``sbb infer --format`` passes) promotes one extension above the rest."""
    for name in ("model.pt", "model.safetensors"):
        (tmp_path / name).write_bytes(b"x")

    assert latest_checkpoint(tmp_path) == tmp_path / "model.pt"
    assert latest_checkpoint(tmp_path, prefer="safetensors") == tmp_path / "model.safetensors"
    assert latest_checkpoint(tmp_path, prefer=".pt") == tmp_path / "model.pt"
    # An extension nothing in the directory has changes nothing rather than returning None.
    assert latest_checkpoint(tmp_path, prefer="bin") == tmp_path / "model.pt"


def test_latest_checkpoint_ties_break_on_name_not_glob_order(tmp_path: Path) -> None:
    """Two files written in the same second resolve the same way on every filesystem."""
    stamp = time.time()
    written = [tmp_path / name for name in ("zeta.pt", "alpha.pt", "model.pt")]
    for path in written:
        path.write_bytes(b"x")
        os.utime(path, (stamp, stamp))

    assert latest_checkpoint(tmp_path) == tmp_path / "model.pt"
    (tmp_path / "model.pt").unlink()
    # Same tier, same mtime: the name decides, and it does not depend on glob order.
    assert latest_checkpoint(tmp_path) == tmp_path / "alpha.pt"


def test_prune_adds_the_safetensors_extension_and_spares_the_run_files(tmp_path: Path) -> None:
    """``model.pt`` used to be the first casualty of ``keep_last_n``: no epoch number, no rank."""
    now = time.time()
    for epoch in range(1, 5):
        path = tmp_path / f"run_epoch_{epoch}.pt"
        path.write_bytes(b"w")
        os.utime(path, (now + epoch, now + epoch))
    published = tmp_path / "model.pt"
    published.write_bytes(b"w")
    published_sidecar = published.with_name("model_config.json")
    published_sidecar.write_text("{}")
    exported = tmp_path / "model.safetensors"
    exported.write_bytes(b"w")
    os.utime(exported, (now + 100, now + 100))

    deleted = prune_checkpoints(tmp_path, keep_last_n=3)

    # Six files, keep three: the three oldest *epoch* files go, and the two published ones --
    # model.pt and model.safetensors, neither of which has an epoch number -- are kept even though
    # they are older than every epoch file but one.
    assert sorted(p.name for p in deleted) == ["run_epoch_1.pt", "run_epoch_2.pt", "run_epoch_3.pt"]
    assert {p.name for p in tmp_path.iterdir()} == {
        "model.pt",
        "model_config.json",
        "model.safetensors",
        "run_epoch_4.pt",
    }
    assert published.exists() and exported.exists()
    assert published_sidecar.exists()  # pruned files lose their sidecars; kept files keep theirs


def test_prune_never_deletes_a_safetensors_sidecar(tmp_path: Path) -> None:
    """A ``.safetensors`` holds its config inside, so prune does not own the JSON copy beside it."""
    now = time.time()
    for epoch, name in (
        (1, "run_epoch_1.safetensors"),
        (2, "run_epoch_2.pt"),
        (3, "run_epoch_3.pt"),
    ):
        path = tmp_path / name
        path.write_bytes(b"w")
        os.utime(path, (now + epoch, now + epoch))
    doomed = tmp_path / "run_epoch_1.safetensors"
    sidecar = doomed.with_name("run_epoch_1_config.json")  # written by save(write_sidecar=True)
    sidecar.write_text("{}")

    deleted = prune_checkpoints(tmp_path, keep_last_n=2)

    assert deleted == [doomed]
    assert not doomed.exists()
    assert sidecar.exists()  # a .pt's sidecar would have been unlinked with it


def test_latest_checkpoint_uses_mtime_not_name(tmp_path: Path) -> None:
    """Legacy code used ``ctime``, which changes on ``git checkout`` and misreports the newest."""
    old = tmp_path / "epoch_99.pt"
    new = tmp_path / "epoch_1.pt"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (time.time() - 86400, time.time() - 86400))
    os.utime(new, (time.time(), time.time()))

    # Inside one extension the timestamp answers the question, not the number in the name.
    assert latest_checkpoint(tmp_path) == new
    assert latest_checkpoint(tmp_path, pattern="*.pt") == new  # only .pt considered
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
