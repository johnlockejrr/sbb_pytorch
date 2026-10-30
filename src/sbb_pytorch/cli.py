"""``sbb`` -- one command line for the package, replacing five separate entry scripts.

The legacy repository had no command: it had scripts you edited and ran.

``train.py`` (argparse: ``--config --batch-size --epochs --num-workers --patience --device --resume``)
    :command:`sbb train`. Every one of those flags names a config field (or one of the documented
    aliases -- ``--batch-size`` -> ``n_batch``, ``--epochs`` -> ``n_epochs``, ``--resume`` ->
    ``dir_of_start_model``), so a flag that used to be a one-off now persists in the run's own
    config; precedence is flag, then file, then field default, resolved by
    :func:`~sbb_pytorch.config.loaders.apply_overrides`.
``inference_pytorch.py`` (argparse: ``--input --model --output --save --save-layout ...``)
    :command:`sbb infer`. Its ``--model`` meant "path to the weights" in one script and
    "architecture name" in another; here the path is the argument and the architecture is
    ``--architecture``, because ``Predictor.from_checkpoint`` can read the rest out of the
    checkpoint's own config.
``generate_gt_for_training.py`` (argparse: ``--dir --out --type --printspace --min ...``)
    :command:`sbb gt generate`, plus :command:`sbb gt visualize` for the single-file render that
    used to need a notebook.
``build_model_load_pretrained_weights_and_save.py`` -- a script whose constants you rewrote to
    change anything, then ran to copy weights between files: :command:`sbb export`.
``config_params.json``, documented by a hand-written table that had already drifted from the code:
    :command:`sbb config-schema` (generated from the model, so it cannot drift) and
    :command:`sbb validate`, which reports the cross-config advisories the docs only used to mention.

Two behaviours apply to every subcommand: ``--verbose`` flips the package logger to DEBUG
(:mod:`sbb_pytorch.logging`, the replacement for ~200 stray ``print`` calls), and any
:class:`~sbb_pytorch.exceptions.SBBError` is printed as one line with exit code 1 instead of a
traceback -- ``train.py`` raised ``KeyError('val_acc')`` *after* a whole training run, which is a
poor way to tell someone their run worked.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from sbb_pytorch import __version__
from sbb_pytorch.config import (
    OutputType,
    SBBConfig,
    TaskType,
    UseCase,
    apply_overrides,
    dump_json_schema,
    load_config,
    load_page2label,
    load_scales,
    validate_pair,
)
from sbb_pytorch.data.dataset import label_path_for
from sbb_pytorch.device import resolve_device
from sbb_pytorch.exceptions import CheckpointError, SBBError
from sbb_pytorch.gt import (
    VOCABULARY,
    GroundTruthConfig,
    Page,
    Region,
    RegionKind,
    contour_bbox,
    contours_of_image,
    convex_hull,
    discover_xml_files,
    generate_ground_truth,
    generate_pairs,
    group_contours_into_blocks,
    kind_for_class_name,
    parse_pagexml,
    render_page,
    write_image,
    write_pagexml,
    write_pair,
)
from sbb_pytorch.inference import Prediction, Predictor
from sbb_pytorch.logging import get_logger, set_level
from sbb_pytorch.metrics import accuracy, dice_score, mean_iou
from sbb_pytorch.storage import latest_checkpoint, load_checkpoint
from sbb_pytorch.storage.images import (
    downscale_upscale,
    imwrite,
    list_images,
    read_image,
    read_mask,
    resize_mask,
)
from sbb_pytorch.training import SBBLightningModule, fit
from sbb_pytorch.training.callbacks import build_callbacks, metric_snapshot

__all__ = ["main"]

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

# Where ``sbb config-schema`` writes when ``-o`` is left out. The old default was the relative
# ``docs/config-reference.json``, so the generated reference landed in whatever directory the
# caller happened to stand in. The file it generates is the repository root's
# ``docs/config-reference.json``, and the schema describes this package, so the path is derived
# from the installed package (the ``src`` layout puts ``cli.py`` two levels under the root).
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "config-reference.json"


# ----------------------------------------------------------------------- click plumbing


class SBBGroup(click.Group):
    """A ``click.Group`` that reports the package's own exceptions as one-line errors.

    ``click.ClickException`` is what the CLI prints as ``Error: ...`` with exit code 1; every
    ``sbb_pytorch`` error (:class:`~sbb_pytorch.exceptions.ConfigError`, ``ModelBuildError``,
    ``TrainingError``, ``CheckpointError``, ``PageXMLError``, ...) becomes one here instead of
    dumping a traceback on a user who did nothing wrong but point at a stale config file.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except SBBError as exc:
            raise click.ClickException(str(exc)) from exc


@dataclass
class GlobalState:
    """Per-invocation state on ``click.Context.obj``: the verbosity and the logger behind it."""

    verbose: bool = False
    logger: Any = None


@dataclass
class _Bar:
    """A rich progress bar, or a no-op when stdout is not a terminal.

    :command:`sbb infer` (N files) and :command:`sbb gt generate` (N XML files, reported through a
    callback rather than a loop) both need to advance a bar from an arbitrary place, so the bar is
    an object with an ``advance`` method instead of a ``with Progress()`` wrapped around one loop.
    Without a TTY -- CI, a pipe, the test runner -- every method is a no-op: a progress bar
    written into a log file is noise, and the legacy scripts printed one line per image instead.
    """

    total: int
    description: str
    _progress: Any = None
    _task: Any = None
    seen: int = 0

    def __post_init__(self) -> None:
        if self.total < 1 or not sys.stdout.isatty():
            return
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._task = self._progress.add_task(self.description, total=self.total)

    def advance(self, _item: Any = None) -> None:
        """Mark one unit done. The argument is ignored -- it exists for ``on_progress``."""
        self.seen += 1
        if self._progress is not None:
            self._progress.update(self._task, advance=1)

    def __enter__(self) -> _Bar:
        if self._progress is not None:
            self._progress.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._progress is not None:
            self._progress.stop()


@contextmanager
def _progress(total: int, description: str) -> Iterator[Callable[[Any], None]]:
    """Yield an ``advance(item)`` callback backed by a bar of ``total`` steps."""
    with _Bar(total, description) as bar:
        yield bar.advance


def _echo_problems(problems: Iterable[str]) -> None:
    """Print advisories as yellow lines on stderr, so stdout stays machine-readable."""
    for problem in problems:
        click.secho(f"warning: {problem}", fg="yellow", err=True)


def _device_value(_ctx: Any, _param: Any, value: str | None) -> str | None:
    """Validate ``--device`` at parse time, keeping ``None`` meaning "not given".

    A ``click.Choice`` cannot express ``cuda:1`` without enumerating the host's GPUs, so the value
    is free text checked by :func:`~sbb_pytorch.device.resolve_device` -- the same function
    ``build_trainer`` and ``Predictor`` call, so a flag that parses is a flag that runs. An
    omitted flag stays ``None`` and lets the config decide; the legacy ``train.py`` defaulted its
    ``--device`` to ``cuda`` and crashed on a CPU-only machine that had a config saying ``cpu``.

    The ``ValueError`` becomes a ``click.BadParameter``: a parameter callback runs before any
    command body, so the group's own error handling never sees it and click would otherwise print
    a traceback for what is just a typo in a flag.
    """
    if value is None or not str(value).strip():
        return None
    try:
        return str(resolve_device(value))
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _overlap_value(_ctx: Any, _param: Any, value: float | None) -> float | None:
    """Validate ``--overlap`` at parse time; ``None`` keeps :meth:`Predictor.predict_grid`'s 0.5.

    The window overlap was only reachable through the Python API: ``predict_grid`` accepts
    ``[0, 1)`` and raises ``ValueError`` outside it, which after a checkpoint has been loaded
    costs a model build to learn what click could have said. Same rule as
    :func:`_device_value` -- a flag that parses is a flag that runs.
    """
    if value is None:
        return None
    if not 0.0 <= value < 1.0:
        raise click.BadParameter(f"overlap must be in [0, 1), got {value}")
    return float(value)


def _collect_inputs(paths: Iterable[Path | str], *, limit: int | None = None) -> list[Path]:
    """Expand a mix of files and directories into the images to predict on, in sorted order."""
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(list_images(path)))
        elif path.is_file():
            files.append(path)
        else:  # a symlink click accepted and the filesystem has since forgotten
            raise click.BadParameter(f"{path} is neither a file nor a directory")
    return files[: int(limit)] if limit is not None else files


# ------------------------------------------------------------------------------ the group


@click.group(cls=SBBGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name="sbb")
@click.option("-v", "--verbose", is_flag=True, help="Log at DEBUG instead of INFO.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """SBB pixelwise document-image segmentation: train, infer, ground truth, export.

    \b
      sbb train configs/textline.json            train from a config
      sbb infer runs/textline -i pages/          predict with a run's weights
      sbb gt generate --xml pages/ -o gt/        ground truth from ALTO/PageXML
      sbb gt enhance -i pages/ -o gt/            write the degraded enhancement pages
      sbb gt reading-order --xml pages/           pairwise reading-order samples
      sbb gt visualize page.xml -o page.png      render one XML page
      sbb export runs/textline/model.pt          rewrite a legacy checkpoint
      sbb validate configs/textline.json         check a config without running it
      sbb config-schema -o docs/config-reference.json
    """
    logger = get_logger("sbb", "DEBUG" if verbose else "INFO")
    set_level("DEBUG" if verbose else "INFO")
    ctx.obj = GlobalState(verbose=verbose, logger=logger)


# ------------------------------------------------------------------------------ sbb train


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-m",
    "--model",
    "model_name",
    default=None,
    help="Registry key overriding the architecture the config implies.",
)
@click.option(
    "--pretrained/--no-pretrained",
    "pretrained",
    default=None,
    help="ImageNet weights; default follows config.pretraining (False where unavailable).",
)
@click.option(
    "-d", "--device", default=None, callback=_device_value, help="auto|cpu|cuda|cuda:N|mps."
)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("-e", "--epochs", type=int, default=None)
@click.option("-b", "--batch-size", type=int, default=None)
@click.option("-w", "--num-workers", "workers", type=int, default=None)
@click.option("--lr", "learning_rate", type=float, default=None)
@click.option("--patience", type=int, default=None)
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=True, allow_dash=False, path_type=Path),
    default=None,
    help="Checkpoint to resume from; the config's dir_of_start_model when omitted.",
)
@click.option(
    "--start-weights",
    "start_weights",
    type=click.Path(exists=True, dir_okay=False, allow_dash=False, path_type=Path),
    default=None,
    help="Finetune: .pt/.pth/.ckpt/.safetensors to build the model from, replacing the config's "
    "init (and --pretrained). Starts a new run: the optimiser and epoch counter do not come from it.",
)
@click.option(
    "--page2label", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--scales", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--every-n-steps",
    type=int,
    default=None,
    help="Also save a checkpoint every N batches (the old --save-interval).",
)
@click.option(
    "--callbacks-file",
    "callbacks_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON list of extra callback classes to add to the run's own (see "
    "sbb_pytorch.training.callbacks.load_callbacks_file).",
)
@click.option(
    "--verbose-metrics/--no-verbose-metrics",
    "verbose_metrics",
    default=False,
    help="Print ``Metric val_loss improved. New best score: 0.449`` on every improvement. Off: the "
    "bar carries the same numbers as ``early_stopping/val_loss: 3/10``.",
)
@click.option("--logger/--no-logger", default=True, help="Write TensorBoard (or CSV) logs.")
@click.option("--progress-bar/--no-progress-bar", default=True)
@click.option(
    "--dry-run",
    is_flag=True,
    help="One train and one validation batch, nothing written: checks a config in seconds.",
)
@click.pass_context
def train(
    ctx: click.Context,
    config: Path,
    model_name: str | None,
    pretrained: bool | None,
    device: str | None,
    output_dir: Path | None,
    epochs: int | None,
    batch_size: int | None,
    workers: int | None,
    learning_rate: float | None,
    patience: int | None,
    resume: Path | None,
    start_weights: Path | None,
    page2label: Path | None,
    scales: Path | None,
    every_n_steps: int | None,
    callbacks_file: Path | None,
    verbose_metrics: bool,
    logger: bool,
    progress_bar: bool,
    dry_run: bool,
) -> None:
    """Train from CONFIG; the output directory defaults to the config's ``dir_output``.

    The config is loaded, the flags are applied over it, and the advisories :command:`sbb
    validate` would print are printed *before* the run starts -- they used to be discovered as a
    crash halfway through the first epoch.

    ``--callbacks-file`` adds callback classes to the ones the config implies rather than replacing
    them: a JSON list of dotted import paths or ``{"class": ..., "init": {...}}`` objects, read by
    :func:`~sbb_pytorch.training.callbacks.load_callbacks_file`. The finished list goes to
    ``build_trainer``'s own ``callbacks=`` slot, which is the argument its docstring named the flag
    for (``training/trainer.py:111``); ``every_n_steps`` moves into the same call because
    ``build_trainer`` only derives the built set -- and so only reads that flag -- when it is given
    no callbacks.

    ``--verbose-metrics`` restores the ``Metric val_loss improved. New best score: 0.449`` prints
    the checkpoint and early-stopping callbacks emit on every improvement. It is off by default:
    the progress bar carries the same information as ``early_stopping/val_loss: 3/10``.

    ``--start-weights FILE`` is the finetuning path, and the only one that accepts weights that are
    not a pytorch-lightning ``.ckpt``: a ``model.pt``, a ``model.safetensors`` or a ``.ckpt`` is
    loaded into the model the config builds, and training starts there with a fresh optimiser and
    epoch counter. ``--resume`` cannot take those -- it needs the ``.ckpt`` because that is the only
    file carrying an optimiser state -- so the two flags are mutually exclusive, as are a start file
    and an explicit ``--pretrained`` (see :func:`_resolve_start_weights`). The run then writes
    ``model.pt`` (the final weights, as always) and ``model.safetensors`` (the best ones) when the
    ``safetensors`` package is installed.
    """
    state: GlobalState = ctx.obj
    resolved = load_config(config, page2label_path=page2label, scales_path=scales)
    # --start-weights decides --pretrained (and rejects the combinations that contradict it) before
    # the config is resolved, so the field the run reads matches the weights it is built from.
    pretrained = _resolve_start_weights(resolved, resume, start_weights, pretrained)
    resolved = apply_overrides(
        resolved,
        device=device,
        dir_output=str(output_dir) if output_dir else None,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        lr=learning_rate,
        patience=patience,
        resume=str(resume) if resume else None,
        pretraining=pretrained,
    )
    _echo_problems(_advisories(config, page2label, resolved))
    state.logger.debug("resolved config: %s", resolved.model_dump(mode="json"))

    module = SBBLightningModule(resolved, model_name=model_name, pretrained=pretrained)
    # With a callbacks file the list is built here and handed over whole; without one the trainer
    # builds it, which is the path every existing run takes and stays byte-identical to.
    trainer_kwargs: dict[str, Any] = (
        {"every_n_steps": every_n_steps, "verbose": verbose_metrics}
        if callbacks_file is None
        else {
            "callbacks": build_callbacks(
                resolved,
                module,
                output_dir=output_dir,
                every_n_steps=every_n_steps,
                callbacks_file=callbacks_file,
                verbose=verbose_metrics,
            )
        }
    )
    trainer = fit(
        resolved,
        module=module,
        device=device,
        resume=resume,
        start_weights=start_weights,
        output_dir=output_dir,
        save_final=not dry_run,
        logger="auto" if logger else False,
        enable_progress_bar=progress_bar,
        **trainer_kwargs,
        **({"fast_dev_run": 1} if dry_run else {}),
    )

    checkpoint = Path(output_dir if output_dir is not None else resolved.dir_output) / "model.pt"
    if dry_run:
        click.secho(f"dry run finished; {checkpoint} not written", fg="green")
        return
    click.secho(f"wrote {checkpoint}", fg="green")
    # The best weights, published beside the final ones whenever safetensors is installed.
    if (exported := checkpoint.with_suffix(".safetensors")).exists():
        click.secho(f"wrote {exported}", fg="green")
    for name, value in sorted(metric_snapshot(trainer).items()):
        click.echo(f"  {name:<24} {value:.6f}")


def _resolve_start_weights(
    config: SBBConfig,
    resume: Path | None,
    start_weights: Path | None,
    pretrained: bool | None,
) -> bool | None:
    """The ``--pretrained`` value ``--start-weights`` implies, and the flags it refuses to pair with.

    ``--start-weights`` and ``--resume`` answer different questions -- "train again from this run"
    versus "build a new model out of these weights" -- and pytorch-lightning will not answer both:
    ``trainer.fit(ckpt_path=...)`` restores the optimiser and the epoch counter from the ``.ckpt``
    *after* any weights loaded beforehand, so a start file handed to a resume is loaded and then
    overwritten. Pairing them is therefore a parameter error, and so is pairing a start file with a
    config that asks to continue (``continue_training`` plus ``dir_of_start_model``) or with an
    explicit ``--pretrained``/``--no-pretrained`` -- the download the given file replaces on the next
    line. Everything else forces ``pretrained=False``; without ``--start-weights`` the flag is
    returned untouched, which is every run that existed before this flag did.
    """
    if start_weights is None:
        return pretrained
    if resume is not None:
        raise click.BadParameter(
            "--start-weights and --resume are mutually exclusive: --resume continues the run a "
            "checkpoint came from, --start-weights starts a new one from its weights"
        )
    if config.continue_training:
        raise click.BadParameter(
            f"--start-weights contradicts this config's continue_training=true "
            f"(dir_of_start_model={config.dir_of_start_model}): set continue_training=false, or "
            f"use --resume instead"
        )
    if pretrained is not None:
        raise click.BadParameter(
            f"--{'no-' if pretrained is False else ''}pretrained is redundant next to "
            f"--start-weights {start_weights}: the file is the weights, and --pretrained only "
            f"chooses which ones to download"
        )
    return False


def _advisories(config_path: Path, page2label: Path | None, config: SBBConfig) -> list[str]:
    """The inlined-field warnings plus the main-vs-page2label cross-check, as strings."""
    problems = list(config.warnings())
    if page2label is not None:
        problems.extend(validate_pair(config, load_page2label(page2label)))
    elif config.page2label is not None:
        problems.extend(validate_pair(config, config.page2label))
    del config_path  # the path is in the exception text; the check works on the loaded model
    return problems


# ------------------------------------------------------------------------------ sbb infer


@main.command()
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-i",
    "--input",
    "extra_inputs",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Another file or directory of pages; repeatable.",
)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option(
    "--save/--no-save",
    default=None,
    help="Write the image (or, for label-only tasks, the JSON answer). Default: the config's "
    "save_visualisation, the field that used to be read by nothing.",
)
@click.option("--save-layout/--no-save-layout", default=False, help="Also write the label map.")
@click.option(
    "--save-logits/--no-save-logits",
    "save_logits",
    default=False,
    help="Also write the raw logits as <stem>_logits.pt beside the images -- the tensor "
    "Prediction keeps for exactly this, which nothing used to serialise.",
)
@click.option("--patches/--no-patches", "patches", default=None, help="Default: the config's.")
@click.option(
    "--overlap",
    type=float,
    default=None,
    callback=_overlap_value,
    help="Sliding-window overlap for --patches; default the API's 0.5.",
)
@click.option(
    "--ground-truth",
    "ground_truth",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Label file or directory to score the pages against (the legacy --ground_truth, whose "
    "IoU was computed and thrown away).",
)
@click.option(
    "--xml-file",
    "xml_file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Write each page's answer back as PageXML (the legacy --xml_file, "
    '"the result will be written in the same xml file"). One page: this file. Several: '
    "<its directory>/<page stem>.xml. An existing file is read first so its metadata survives.",
)
@click.option(
    "--created",
    "created",
    default=None,
    help="Metadata/Created stamp for --xml-file (the schema requires one). A page that was read "
    "keeps its own; this names the page that was not. Default: 1970-01-01T00:00:00Z.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config used to rebuild the architecture; default: the checkpoint's sidecar.",
)
@click.option(
    "--architecture",
    default=None,
    help="Registry key overriding the architecture (what --model meant in one legacy script).",
)
@click.option(
    "-d", "--device", default=None, callback=_device_value, help="auto|cpu|cuda|cuda:N|mps."
)
@click.option("--limit", type=int, default=None, help="Stop after N pages.")
@click.pass_context
def infer(
    ctx: click.Context,
    model: Path,
    inputs: tuple[Path, ...],
    extra_inputs: tuple[Path, ...],
    output_dir: Path | None,
    save: bool | None,
    save_layout: bool,
    save_logits: bool,
    patches: bool | None,
    overlap: float | None,
    ground_truth: Path | None,
    xml_file: Path | None,
    created: str | None,
    config_path: Path | None,
    architecture: str | None,
    device: str | None,
    limit: int | None,
) -> None:
    """Run MODEL over INPUT (files or directories) and write the predictions.

    MODEL is a run directory or a checkpoint file. The config is read from the checkpoint unless
    ``-c`` says otherwise, which is what lets ``sbb infer runs/textline -i pages/`` work without
    repeating every option the training run used -- the legacy script needed
    ``--config --model --input --output`` every single time.

    ``--ground-truth`` adds the per-page IoU/accuracy/Dice the legacy script computed and never
    printed; ``--overlap`` tunes the sliding window ``--patches`` builds. Without ``--save`` or
    ``--no-save``, the config's ``save_visualisation`` decides what gets written. ``--xml-file``
    writes each page's answer back as PageXML, the way ``--xml_file`` and ``--page-xml`` did
    upstream; see :func:`_page_xml_for` for which file that is and :func:`_write_page_xml` for what
    goes in it. ``--created`` stamps the pages that have no timestamp of their own -- the schema
    requires one, and a page built from a mask has no file to take it from.
    """
    state: GlobalState = ctx.obj
    checkpoint = _resolve_model_path(model)
    predictor = Predictor.from_checkpoint(
        checkpoint.parent if checkpoint.is_file() else checkpoint,
        config=config_path,
        model=architecture,
        device=device,
    )
    files = _collect_inputs([*inputs, *extra_inputs], limit=limit)
    if not files:
        raise click.BadParameter("no images to predict on (give a file, a directory, or --input)")

    destination = Path(output_dir if output_dir is not None else checkpoint.parent / "predictions")
    destination.mkdir(parents=True, exist_ok=True)
    use_patches = predictor.config.patches if patches is None else bool(patches)
    # The same flag-or-config rule as --patches: save_visualisation decides, --save/--no-save wins.
    should_save = predictor.config.save_visualisation if save is None else bool(save)
    if overlap is not None and not use_patches:
        raise click.BadParameter("--overlap tunes the --patches window; give --patches, or drop it")

    written = 0
    xml_written = 0
    scores: list[tuple[Path, dict[str, float]]] = []
    with _progress(len(files), "predict") as advance:
        for path in files:
            prediction = predictor.predict_file(
                path,
                patches=use_patches,
                **({"overlap": overlap} if overlap is not None else {}),
            )
            written += _save_prediction(
                prediction,
                path,
                destination,
                save=should_save,
                save_layout=save_layout,
                save_logits=save_logits,
            )
            if xml_file is not None:
                xml_written += _write_page_xml(
                    prediction,
                    path,
                    _page_xml_for(path, xml_file, single_page=len(files) == 1),
                    predictor.config,
                    created,
                )
            if ground_truth is not None:
                scores.append((path, _score_prediction(prediction, path, ground_truth)))
            advance(path)
    state.logger.debug("predictor built from %s", checkpoint)
    summary = f"{len(files)} page(s) predicted from {checkpoint}" + (
        f", {written} file(s) written to {destination}" if written else ", nothing saved"
    )
    if xml_written:
        # Its own clause, not part of the image count: ``--xml-file`` often names a file somewhere
        # else, so the number beside ``destination`` would be about a directory it did not write to.
        summary += f", {xml_written} PageXML file(s) written"
    click.secho(summary, fg="green")
    # Same table shape ``sbb train`` prints, one row per scored page.
    for path, values in scores:
        click.echo(
            f"  {path.name:<24} "
            + " ".join(f"{name}={value:.6f}" for name, value in sorted(values.items()))
        )


def _save_prediction(
    prediction: Prediction,
    source: Path,
    destination: Path,
    *,
    save: bool,
    save_layout: bool,
    save_logits: bool = False,
) -> int:
    """Write whatever this prediction actually produced; return the number of files written.

    Segmentation and enhancement produce images, which
    :meth:`~sbb_pytorch.inference.predictor.Prediction.save` writes. Classification and reading
    order produce a label and probabilities and no image at all -- ``Prediction.save`` raises
    rather than pretend otherwise, and the legacy script wrote nothing at all -- so their answer
    goes to ``<stem>_prediction.json``. Without this, ``sbb infer`` on a classification checkpoint
    failed on its own default ``--save``.

    ``--save-logits`` is counted here rather than inside ``Prediction.save`` because the logits are
    not an image: :class:`~sbb_pytorch.inference.predictor.Prediction` documents them as kept "for
    metrics and ``--save-logits`` style debugging", and this is the only place that can write them
    (``torch.save`` of the ``(n_classes, H, W)`` tensor, so a checkpoint's own output size survives
    the resize the image goes through). They are written even when ``--no-save`` suppresses images,
    since asking for the logits *is* the request.
    """
    written = 0
    if save_logits and prediction.logits is not None:
        torch.save(prediction.logits.cpu(), destination / f"{source.stem}_logits.pt")
        written = 1
    if prediction.mask is None and prediction.image is None:
        if not save or prediction.label is None:
            return written
        target = destination / f"{source.stem}_prediction.json"
        target.write_text(
            json.dumps(
                {
                    "task": prediction.task.value,
                    "label": prediction.label,
                    "probabilities": [
                        {"class": name, "probability": round(probability, 6)}
                        for name, probability in prediction.probabilities
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return written + 1
    if not (save or save_layout):
        return written
    return written + len(
        prediction.save(
            save=str(destination / f"{source.stem}_prediction.png") if save else None,
            save_layout=(str(destination / f"{source.stem}_layout.png") if save_layout else None),
        )
    )


def _page_xml_for(source: Path, xml_file: Path, *, single_page: bool) -> Path:
    """Which PageXML file *source*'s answer goes into, given ``--xml-file``.

    The flag names one file, which is what upstream's meant (``inference.py:731-735``: "xml file
    with layout coordinates that reading order detection will be implemented on. The result will be
    written in the same xml file") and is only well defined for one page. Over several pages -- the
    case ``-i/--input`` has always allowed and the sibling ``--page-xml``
    (``inference.py:736-740``, "save PAGE-XML file with detected regions and lines") never resolved
    either -- the file's *directory* plus the page's stem is the pairing rule the port already uses
    for every other per-page artefact (:func:`sbb_pytorch.data.dataset.label_path_for`: same
    directory, same stem, extension from the artefact instead of the task-dependent ``.png`` it
    picks for a label). An omitted ``--xml-file`` writes no PageXML at all, even where a sibling
    exists: overwriting a hand-authored page is not something to do by accident, which is why the
    option -- like ``--save``, whose default is the config's -- has to be asked for.
    """
    return xml_file if single_page else xml_file.parent / f"{source.stem}.xml"


def _class_labels(config: SBBConfig, label: int) -> list[str]:
    """Every name a class id covers -- page2label's, or the ``class_<id>`` fallback it would need.

    The same names :meth:`sbb_pytorch.inference.predictor.Predictor._class_names` assembles for a
    classification answer, kept as a list because the two callers want different things from them:
    :func:`_class_name` joins them into the label a prediction carries, and :func:`_kind_for_class`
    asks each one in turn which element it names.
    """
    mapping = config.page2label.class_ids() if config.page2label else {}
    return sorted(mapping[label]) if mapping.get(label) else [f"class_{label}"]


def _class_name(config: SBBConfig, label: int) -> str:
    """The name a class id answers with: page2label's, else the ``class_<id>`` fallback.

    The same rule :meth:`sbb_pytorch.inference.predictor.Predictor._class_names` applies to a
    classification answer, so a region written from a mask and a ``_prediction.json`` written from a
    label name the same class the same way. What the *element* may say about that name -- ``@type``
    or only ``@custom`` -- is the writer's question, and :mod:`sbb_pytorch.gt.vocabulary` answers it.
    """
    return ", ".join(_class_labels(config, label))


def _kind_for_class(config: SBBConfig, label: int) -> RegionKind:
    """The page element a class id's contours are written as: ``ImageRegion`` for an image class.

    :func:`sbb_pytorch.gt.vocabulary.kind_for_class_name` resolves the name, and this keeps only a
    name whose element can sit on a page -- a class called ``textline`` is not a page-level region to
    write a whole contour as. With nothing to go on the answer is ``TextRegion``, which is what every
    contour was before the vocabulary existed and what a ``class_<id>`` fallback still is.
    """
    for name in _class_labels(config, label):
        kind = kind_for_class_name(name)
        if kind is not None and VOCABULARY[kind].page_level:
            return kind
    return RegionKind.TEXT


def _mask_labels(mask: np.ndarray) -> list[int]:
    """Every class id *mask* carries, background (``0``) excluded, smallest first."""
    return sorted(int(value) for value in np.unique(mask) if int(value) != 0)


#: The use case a mask answer takes its shape from when the config has no ``page2label`` to name one.
#: The three tasks left over (classification, enhancement, reading order) answer with a label rather
#: than a mask, so their entry only matters for a config that produces one anyway -- and the answer
#: then is the layout shape, which is all such a config ever produced.
_MASK_USE_CASE_BY_TASK: dict[TaskType, UseCase] = {
    TaskType.SEGMENTATION: UseCase.LAYOUT,
    TaskType.BINARIZATION: UseCase.BINARIZATION,
    TaskType.CLASSIFICATION: UseCase.LAYOUT,
    TaskType.ENHANCEMENT: UseCase.LAYOUT,
    TaskType.READING_ORDER: UseCase.LAYOUT,
}


def _use_case_for_mask(config: SBBConfig) -> UseCase:
    """The use case *config*'s mask is written as: its ``page2label``'s, else its task's default."""
    if config.page2label is not None:
        return config.page2label.use_case
    return _MASK_USE_CASE_BY_TASK[config.task]


def _regions_for_mask(source: Path, config: SBBConfig, mask: np.ndarray) -> list[Region]:
    """The regions a mask answer describes, in the shape its use case annotates.

    Asking the use case is the whole fix for a textline model whose output looked like a layout page:
    ``_mask_regions`` hardcoded ``kind=RegionKind.TEXT`` for every contour of every class, so
    nothing a mask said could come out as anything but a ``TextRegion`` -- and the lines the model
    actually predicted became 239 of them, with no ``TextLine`` anywhere.

    ==============================  ====================================================
    use case                        the shape
    ==============================  ====================================================
    ``layout``                      one region per class contour (:func:`_flat_regions`)
    ``textline``                    a ``TextRegion`` per block, a ``TextLine`` per contour
    ``printspace``, ``binarization``  no regions -- the answer is ``Page/PrintSpace``
    ``word``, ``glyph``             :class:`click.ClickException`, naming the use case
    ==============================  ====================================================

    Raises:
        click.ClickException: *config*'s use case is ``word`` or ``glyph``, which this port can
            predict but not yet write -- the same voice :func:`_page_from_prediction` uses for a
            prediction it cannot describe.
    """
    use_case = _use_case_for_mask(config)
    if use_case in (UseCase.WORD, UseCase.GLYPH):
        raise click.ClickException(
            f"--xml-file cannot describe {source.name}: use_case={use_case.value} is not implemented "
            f"yet, so its prediction has no PageXML shape to be written as"
        )
    if use_case in (UseCase.PRINTSPACE, UseCase.BINARIZATION):
        return []  # every pixel of the answer goes into Page/PrintSpace; see ``_printspace_from_mask``
    if use_case is UseCase.TEXTLINE:
        return _textline_regions(source, config, mask)
    return _flat_regions(source, config, mask)


def _flat_regions(source: Path, config: SBBConfig, mask: np.ndarray) -> list[Region]:
    """One region per contour of every class on the page -- the layout shape, and the old behaviour.

    v3 built one ``TextRegion`` per class from that class's contours and put each contour in as a
    ``TextLine`` (``inference.py:615-673``); the layout answer this port writes keeps the contours
    as regions of their own, which is what a layout page is: the regions *are* the answer, and they
    are not nested in anything. The area floor is off because
    :data:`~sbb_pytorch.gt.DEFAULT_MIN_AREA_PIXELS` exists to shed JPEG noise, and a predicted
    region one pixel wide has an OpenCV area of zero.
    """
    regions: list[Region] = []
    for label in _mask_labels(mask):
        # One raster per class, because ``cv2.findContours`` cannot segment a labelled field: the
        # port's own gt pipeline masks per class the same way (gt/generator.py builds its label map
        # first and asks geometry for the contours of one region at a time).
        kind = _kind_for_class(config, label)
        name = _class_name(config, label)
        contours = contours_of_image(np.where(mask == label, 255, 0).astype(np.uint8), min_area=0.0)
        for index, contour in enumerate(contours):
            regions.append(
                Region(
                    kind=kind,
                    type_name=name,
                    region_id=f"{source.stem}_{label}_{index}",
                    contour=contour,
                )
            )
    return regions


def _textline_regions(source: Path, config: SBBConfig, mask: np.ndarray) -> list[Region]:
    """``TextRegion`` blocks holding the predicted ``TextLine``\\ s -- the shape lines need.

    A textline model predicts lines and nothing else, so the question is not which regions to write
    but which blocks the lines form, and :func:`sbb_pytorch.gt.geometry.group_contours_into_blocks`
    answers that from geometry alone: each block's outline is the convex hull of the lines inside it,
    which is the quad an annotator draws around a column of lines and not the page-wide box a
    bounding box of everything would give. Grouping happens per class, as it did in v3, so a
    ``header`` line never joins a ``paragraph`` block.
    """
    regions: list[Region] = []
    for label in _mask_labels(mask):
        contours = contours_of_image(np.where(mask == label, 255, 0).astype(np.uint8), min_area=0.0)
        kind = _kind_for_class(config, label)
        name = _class_name(config, label)
        for block_index, group in enumerate(group_contours_into_blocks(contours)):
            block_id = f"{source.stem}_{label}_{block_index}"
            block = Region(
                kind=kind,
                type_name=name,  # @type if the element may say it, @custom always
                region_id=block_id,
                contour=convex_hull(group),
            )
            for line_index, contour in enumerate(group):
                line = Region(
                    kind=RegionKind.TEXTLINE,
                    type_name="",  # a line's class is its element: @custom says text_line
                    region_id=f"{block_id}_{line_index}",
                    contour=contour,
                )
                line.parent = block
                block.children.append(line)
            regions.append(block)
    return regions


def _printspace_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """The ``Page/PrintSpace`` outline a printspace or binarization answer implies.

    Both use cases say one thing about a page -- where the ink is -- and PageXML writes that as the
    page's print space rather than as regions, which is why :func:`_regions_for_mask` returns none for
    them: a ``TextRegion`` per black blob was what the port wrote before, for a task that names no
    regions at all. The outline is the convex hull of the ink, not the ink's own contour: the schema's
    print space is one polygon, and a page of two columns has one.
    """
    contours = contours_of_image(np.where(mask == 0, 0, 255).astype(np.uint8), min_area=0.0)
    hull = convex_hull(contours)
    return hull if hull.size else None


def _ordered_ids(page: Page, label: str) -> dict[str, int]:
    """``Page.region_index`` for a page whose answer is the class name *label*.

    A label names a class, not a sequence of regions, and a PageXML order can only reference regions
    that exist: the regions whose ``@type`` the label covers -- a multi-name class is exactly the
    comma string ``Prediction.label`` carries (``inference/predictor.py:285-289``) -- are ordered
    top-to-bottom then left-to-right, and the regions it does not cover keep no index instead of
    vanishing, which is how :func:`sbb_pytorch.gt.reading_order.ordered_regions` treats the ones a
    min-area filter rejected. The order itself is the page's geometry rather than the model's,
    because a single class label carries no ordering information to keep.
    """
    covered = {name.strip() for name in label.split(",") if name.strip()}
    ranked = sorted(
        (region for region in page.regions if region.region_id and region.type_name in covered),
        key=lambda region: (
            *contour_bbox(region.contour, (page.height, page.width))[:2],
            region.region_id,
        ),
    )
    return {region.region_id: index for index, region in enumerate(ranked)}


def _page_from_prediction(
    source: Path,
    prediction: Prediction,
    config: SBBConfig,
    existing: Page | None,
    created: str | None = None,
) -> Page:
    """The page *source* is now that it has a prediction, keeping what the file already knew.

    Three shapes, the same three :class:`~sbb_pytorch.inference.predictor.Prediction` can hold:

    a ``mask`` (segmentation, binarization)
        The regions come from the prediction, in the shape its use case annotates
        (:func:`_regions_for_mask`) -- the ``--page-xml`` half of upstream -- while the declared
        size, the ``Metadata`` column count, the ``@externalRef`` and the timestamps come from
        *existing* when the file was readable and from the mask itself when it was not. A
        printspace or binarization answer replaces the print space instead of the regions, because
        that is the only thing either task says about a page. The old reading order goes with the
        old regions: its ids named regions that are no longer there, and the new ones name classes.
        *created* is ``--created``: the timestamp to stamp a page with when the file had none.
    a ``label`` (classification, reading order)
        The answer is an ordering, so the page keeps every region it had (``dataclasses.replace``)
        and gains the order :func:`_ordered_ids` builds. With no file to order there is nothing to
        write: ``--xml-file`` then says so in one line rather than writing an empty page that would
        overwrite the description it could not read.
    neither (this port's enhancement page: ``image`` and logits, never a mask, where v3's
        ``--xml_file`` only ever ran the segmentation branch -- ``inference.py:753`` literally
        hardcoded ``task = 'segmentation'``)
        Reported the way ``--ground-truth`` reports the same shape, instead of writing a page whose
        regions came from nowhere.
    """
    if prediction.mask is not None:
        height, width = (
            (existing.height, existing.width)
            if existing is not None
            else (int(prediction.mask.shape[0]), int(prediction.mask.shape[1]))
        )
        use_case = _use_case_for_mask(config)
        # The answer replaces the print space only for the two use cases whose whole answer is one;
        # every other shape keeps the page's own, which is a fact about the page and not about regions.
        printspace = (
            _printspace_from_mask(prediction.mask)
            if use_case in (UseCase.PRINTSPACE, UseCase.BINARIZATION)
            else (existing.printspace_contour if existing else None)
        )
        return Page(
            path=source,
            file_name=source.stem,
            height=height,
            width=width,
            columns_width=existing.columns_width if existing else None,
            regions=_regions_for_mask(source, config, prediction.mask),
            printspace_contour=printspace,
            printspace_bbox=(
                contour_bbox(printspace, (height, width))
                if printspace is not None
                else (existing.printspace_bbox if existing else None)
            ),
            external_ref=existing.external_ref if existing else "",
            creator=existing.creator if existing else None,
            # --created wins over the file's own stamp, and the file's wins over the writer's
            # placeholder, which is what the option exists to replace.
            created=created or (existing.created if existing else None),
            last_change=(existing.last_change if existing else None) or created,
        )
    if prediction.label is not None:
        if existing is None:
            raise click.ClickException(
                f"--xml-file has no page to order: {source.name} has no PageXML beside it, and "
                f"task={prediction.task.value} answers with a label, which is an order stamped onto "
                f"an existing page"
            )
        return replace(existing, region_index=_ordered_ids(existing, prediction.label or ""))
    raise click.ClickException(
        f"--xml-file cannot describe {source.name}: task={prediction.task.value} produced neither a "
        f"mask nor a label, so there are no regions or order to write"
    )


def _write_page_xml(
    prediction: Prediction,
    source: Path,
    target: Path,
    config: SBBConfig,
    created: str | None = None,
) -> int:
    """Fold one page's answer into its PageXML and write the file; returns 1.

    An existing *target* is parsed before anything is written, because upstream's promise was that
    the result goes into *the same file* (``inference.py:731-735``), not that the file is replaced
    with a stub -- and a file that is not readable PageXML raises
    :class:`~sbb_pytorch.exceptions.PageXMLError`, which ``SBBGroup`` prints as one ``Error:`` line,
    rather than being overwritten unread. A *missing* target is not an error for a task that answers
    with regions (that is what ``--page-xml`` created, and why ``exists=False``): the file is built
    from the prediction and written. ``--no-save`` still writes the XML: asking for it *is* the
    request, the way ``--save-logits`` is. *created* is ``--created``, passed through to
    :func:`_page_from_prediction` for the pages that have no timestamp of their own to keep.
    """
    existing = parse_pagexml(target) if target.is_file() else None
    write_pagexml(_page_from_prediction(source, prediction, config, existing, created), target)
    return 1


def _score_prediction(prediction: Prediction, source: Path, ground_truth: Path) -> dict[str, float]:
    """Score one page against its ground-truth label -- the IoU ``inference_pytorch.py`` discarded.

    The legacy script ran ``calculate_iou(gt, pred)`` whenever ``--ground_truth`` was given and
    never printed the number. The metrics come from :mod:`sbb_pytorch.metrics` and the label from
    :func:`~sbb_pytorch.data.dataset.label_path_for` (a label file directly, or the ``<stem>.png``
    in the given directory). Only a task that answers per pixel can be scored: a classification
    page has a label and an enhancement page an image, neither of which the confusion-matrix
    metrics can compare, so those raise instead of printing a number that would mean nothing.
    """
    if prediction.mask is None:
        raise click.ClickException(
            f"--ground-truth scores a per-pixel prediction; {source.name} gave no mask "
            f"for task={prediction.task.value}"
        )
    label_path = (
        ground_truth
        if ground_truth.is_file()
        else label_path_for(source, ground_truth, task=prediction.task)
    )
    try:
        # Both sides at the prediction's own grid: the metric compares the two sample-for-sample,
        # and ``resize_mask`` is the same nearest-neighbour resizer the predictor used for its logits.
        label = resize_mask(read_mask(label_path), *prediction.mask.shape[:2])
        predicted, target = torch.from_numpy(prediction.mask)[None], torch.from_numpy(label)[None]
        n_classes = prediction.logits.shape[0] if prediction.logits is not None else None
        return {
            "miou": mean_iou(predicted, target, n_classes),
            "accuracy": accuracy(predicted, target, n_classes),
            "dice": dice_score(predicted, target, n_classes),
        }
    except ValueError as exc:  # e.g. a label id outside the config's class range
        raise click.ClickException(f"cannot score {source.name}: {exc}") from exc


def _resolve_model_path(model: Path) -> Path:
    """The checkpoint ``sbb infer`` and ``sbb export`` were pointed at.

    A directory means ``model.pt`` if present, else the newest checkpoint in it -- the resolution
    :class:`~sbb_pytorch.inference.predictor.Predictor.from_checkpoint` does itself, done here so
    a missing file is reported before torch is imported and a model is built.
    """
    if model.is_file():
        return model
    found = latest_checkpoint(model)
    if found is None:
        raise CheckpointError(f"no checkpoint found in {model}")
    return found


# ---------------------------------------------------------------------------- sbb gt ...


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--page2label", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--scales", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--strict", is_flag=True, help="Exit 1 when there are advisories, not only when invalid."
)
@click.pass_context
def validate(
    ctx: click.Context, config: Path, page2label: Path | None, scales: Path | None, strict: bool
) -> None:
    """Check CONFIG (and its page2label / scales files) without starting a run.

    A config that fails schema validation exits 1 in every mode: that is the error the legacy
    scripts found at model-build time, after the data had been read. The cross-config advisories
    -- the shipped ``config_params.json`` declares ``n_classes=2`` while
    ``custom_config_page2label.json`` uses ids up to 6 -- are printed as warnings and only fail
    with ``--strict``, because refusing to start would break runs that exist today.
    """
    resolved = load_config(config, page2label_path=page2label, scales_path=scales)
    problems = _advisories(config, page2label, resolved)
    click.secho(f"{config} is a valid SBBConfig ({resolved.task.value})", fg="green")
    _echo_problems(problems)
    if problems and strict:
        raise click.exceptions.Exit(1)
    del ctx  # the group's state carries only logging, which validate does not need


@main.group()
def gt() -> None:
    """Ground truth from ALTO/PageXML: generate, enhance, reading-order, or render one page."""


@gt.command("generate")
@click.option(
    "--xml",
    "xml_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of ALTO/PageXML files.",
)
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GroundTruthConfig JSON; the alternative to the flags below.",
)
@click.option(
    "-u",
    "--use-case",
    "use_case_raw",
    type=click.Choice([member.value for member in UseCase]),
    default=None,
    help="Required unless --config is given.",
)
@click.option(
    "--type-output",
    "output_type",
    type=click.Choice([member.value for member in OutputType]),
    default=None,
    help="Label format: 2d (class id per pixel, replicated across the channels) or 3d (RGB). "
    "Default: the config's own ``type_output``, which ``--config`` used to ignore; 3d when "
    "--use-case is given, as the legacy command wrote it.",
)
@click.option("--printspace/--no-printspace", default=False)
@click.option(
    "--images-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--images-output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option(
    "--min-area",
    type=float,
    default=None,
    help="Smallest contour kept, as a page fraction; overrides the --config value.",
)
@click.option("--limit", type=int, default=None, help="Process only the first N XML files.")
@click.option("--dry-run", is_flag=True, help="List what would be generated and stop.")
@click.pass_context
def gt_generate(
    ctx: click.Context,
    xml_dir: Path,
    output_dir: Path,
    config_path: Path | None,
    use_case_raw: str | None,
    output_type: str | None,
    printspace: bool,
    images_dir: Path | None,
    images_output_dir: Path | None,
    min_area: float | None,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Generate ground-truth images from the XML files in ``--xml`` into ``-o``.

    Give either ``--config`` (a :class:`~sbb_pytorch.gt.GroundTruthConfig` JSON, the old
    ``custom_config_page2label.json`` shape) or ``--use-case`` -- the legacy script made you edit
    the constants in ``generate_gt_for_training.py`` instead.

    ``--type-output`` falls back to the config's own ``type_output`` when it is not given, so
    ``sbb gt generate -c configs/textline.json`` builds the format that file asks for; with
    ``--use-case`` there is no file to ask, and it is ``3d`` as the legacy command wrote it.
    """
    if config_path is not None and use_case_raw is not None:
        raise click.BadParameter("give --config or --use-case, not both")
    if config_path is None and use_case_raw is None:
        raise click.BadParameter("give --config or --use-case")

    # ``--min-area`` next to ``--config`` used to be refused outright ("it is set in the file"),
    # which made a config file the only place to change one number. The field is overridden in
    # place now -- the flag-then-file precedence ``sbb train`` already uses for its own config --
    # and the config is parsed here so an unreadable file fails before the XML folder is walked.
    gt_config = (
        GroundTruthConfig.parse(config_path)
        if config_path is not None
        else GroundTruthConfig(use_case=UseCase(use_case_raw or "layout"))
    )
    if min_area is not None:
        gt_config = replace(gt_config, min_area=min_area)
    # ``--type-output`` and the config's ``type_output`` answer one question, and the flag used to
    # settle it with its own ``3d`` default even when the file said ``2d`` -- the same outranking
    # ``sbb infer --save`` once had over ``save_visualisation``. The flag has no default now, so
    # leaving it out asks the file; naming it still wins. The legacy option had no default either.
    resolved_output_type = OutputType.parse(output_type) if output_type else gt_config.type_output

    xml_files = discover_xml_files(xml_dir)
    if limit is not None:
        xml_files = xml_files[: int(limit)]
    if not xml_files:
        raise click.BadParameter(f"no ALTO/PageXML files in {xml_dir}")
    if dry_run:
        for path in xml_files:
            click.echo(path.name)
        click.secho(f"{len(xml_files)} file(s) in {xml_dir} would produce {output_dir}", fg="green")
        return

    state: GlobalState = ctx.obj
    with _progress(len(xml_files), "ground truth") as advance:
        result = generate_ground_truth(
            gt_config,
            xml_dir,
            output_dir,
            output_type=resolved_output_type,
            printspace=printspace,
            images_dir=images_dir,
            images_output_dir=images_output_dir,
            xml_files=xml_files,
            on_progress=advance,
        )
    state.logger.debug("generation result: %s", result)
    click.secho(f"{output_dir}: {_summarise(result)}", fg="green")


def _summarise(result: object) -> str:
    """One line about a ``GenerationResult``, whatever fields it happens to carry.

    ``GenerationResult`` is a ``slots`` dataclass, so ``vars()`` raises ``TypeError`` on it and
    the plain ``getattr(result, "model_dump", ...)`` fallback never ran -- which is why every
    non-``--dry-run`` ``gt generate`` died here while the dry-run tests passed. Its own
    :meth:`~sbb_pytorch.gt.GenerationResult.summary` is written for exactly this call, so that
    comes first; ``model_dump`` covers a pydantic result and ``vars`` a plain dict-holding one.
    """
    for name in ("summary", "model_dump"):
        render = getattr(result, name, None)
        if not callable(render):
            continue
        values = render()
        if isinstance(values, dict):
            return ", ".join(f"{key}={value}" for key, value in sorted(values.items(), key=str))
        return str(values)
    try:
        values = vars(result)
    except TypeError:  # no __dict__: a slots class without summary()
        return str(result)
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items(), key=str))


@gt.command("visualize")
@click.argument("xml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output",
    type=click.Path(path_type=Path),
    default=None,
    help="Image to write; default: <xml>_<type>.png beside the XML.",
)
@click.option("--size", nargs=2, type=int, default=None, metavar="H W", help="Render at this size.")
@click.option("--type-output", "output_type", type=click.Choice(["2d", "3d"]), default="3d")
def gt_visualize(
    xml: Path, output: Path | None, size: tuple[int, int] | None, output_type: str
) -> None:
    """Render one ALTO/PageXML file as an image (the old notebook's single-page view)."""
    page = parse_pagexml(xml)
    image = render_page(page, size=tuple(size) if size else None, output_type=output_type)
    # Beside the XML, not in the current directory: the render belongs with the page it came from.
    destination = (
        Path(output) if output is not None else xml.with_name(f"{xml.stem}_{output_type}.png")
    )
    written = write_image(destination, image)
    click.secho(f"wrote {written}", fg="green")


@gt.command("enhance")
@click.option(
    "-i",
    "--images",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Source pages: files and/or directories (the legacy ``--dir_imgs``).",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where the degraded pages go (``--dir_out_images``).",
)
@click.option(
    "--labels-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where the originals are written as enhancement labels (``--dir_out_labels``); "
    "default ``<output-dir>_labels``, the sibling the legacy command asked for.",
)
@click.option(
    "-s",
    "--scales",
    "scales_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="``scales_enhancement.json``; its ``scales`` list is the degradation set.",
)
@click.option("--limit", type=int, default=None, help="Stop after N pages.")
@click.option("--dry-run", is_flag=True, help="List the pages and stop.")
def gt_enhance(
    images: tuple[Path, ...],
    output_dir: Path,
    labels_dir: Path | None,
    scales_path: Path,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Write each page degraded to every scale -- the legacy ``image_enhancement`` command.

    ``generate_gt_for_training.py``'s ``image_enhancement`` read ``scales_enhancement.json``,
    resized every page down and back up once per scale, and wrote the pair: the degraded page as
    the image, the original as the label. That is
    :func:`~sbb_pytorch.storage.images.downscale_upscale` (the function the port keeps for it)
    plus :func:`~sbb_pytorch.config.load_scales`, one page and one scale at a time.
    """
    scales = load_scales(scales_path).scales
    pages = _collect_inputs(images, limit=limit)
    if not pages:
        raise click.BadParameter("no images to degrade (give a file, a directory, or --images)")
    labels_out = Path(labels_dir) if labels_dir is not None else Path(f"{output_dir}_labels")
    if dry_run:
        for path in pages:
            click.echo(path.name)
        click.secho(
            f"{len(pages)} page(s) x {len(scales)} scale(s) would produce "
            f"{len(pages) * len(scales)} image/label pair(s) in {output_dir}",
            fg="green",
        )
        return

    written = 0
    with _progress(len(pages), "enhance") as advance:
        for path in pages:
            image = read_image(path)
            for index, scale in enumerate(scales):
                # One pair per scale, named like the legacy ``<stem>_<i>.<ext>``: the loader matches
                # an enhancement image to its label by filename, so both halves need the same name.
                name = f"{path.stem}_{index}{path.suffix}"
                imwrite(output_dir / name, downscale_upscale(image, scale))
                imwrite(labels_out / name, image)
                written += 2
            advance(path)
    click.secho(f"{written} file(s) written to {output_dir} and {labels_out}", fg="green")


@gt.command("reading-order")
@click.option(
    "--xml",
    "xml_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of ALTO/PageXML files (the legacy ``--dir_xml``).",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where the modal images go (``--dir_out_modal_image``).",
)
@click.option(
    "--classes-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where the ``.npy`` class labels go (``--dir_out_classes``); default "
    "``<output-dir>/classes``.",
)
@click.option(
    "--size", nargs=2, type=int, default=None, metavar="H W", help="Rasters at this size."
)
@click.option(
    "--min-area", type=float, default=None, help="Smallest region paired (``--min_area_size``)."
)
@click.option("--max-area", type=float, default=None, help="Largest region paired.")
@click.option(
    "--min-area-early",
    type=float,
    default=None,
    help="The second threshold the legacy command took, ``--min_area_early``: regions under it "
    "are still paired, but marked ``_missed``.",
)
@click.option(
    "--index-start", type=int, default=0, help="First counter value, for appending to a dataset."
)
@click.option("--limit", type=int, default=None, help="Process only the first N XML files.")
@click.option("--dry-run", is_flag=True, help="Count the pairs without writing them.")
def gt_reading_order(
    xml_dir: Path,
    output_dir: Path,
    classes_dir: Path | None,
    size: tuple[int, int] | None,
    min_area: float | None,
    max_area: float | None,
    min_area_early: float | None,
    index_start: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Write the pairwise samples of ``--xml`` -- the legacy ``machine_based_reading_order`` command.

    The old script read the XML inline and emitted one ``.png`` + ``.npy`` pair per ordered region
    pair; :func:`~sbb_pytorch.gt.reading_order.generate_pairs` is that code ported and
    :func:`~sbb_pytorch.gt.reading_order.write_pair` keeps its file contract, ``_missed`` marker
    included.
    """
    xml_files = discover_xml_files(xml_dir)
    if limit is not None:
        xml_files = xml_files[: int(limit)]
    if not xml_files:
        raise click.BadParameter(f"no ALTO/PageXML files in {xml_dir}")
    classes_out = Path(classes_dir) if classes_dir is not None else output_dir / "classes"
    sizes = {"input_height": size[0], "input_width": size[1]} if size else {}
    areas = {
        name: value
        for name, value in (
            ("min_area", min_area),
            ("max_area", max_area),
            ("min_area_early", min_area_early),
        )
        if value is not None
    }

    written = 0
    with _progress(len(xml_files), "reading order") as advance:
        for xml_file in xml_files:
            pairs = generate_pairs(
                parse_pagexml(xml_file), **sizes, **areas, index_start=index_start
            )
            for pair in pairs:
                if not dry_run:
                    write_pair(pair, output_dir, classes_out)
                written += 1
            advance(xml_file)
    verb = "would be written" if dry_run else "were written"
    click.secho(f"{len(xml_files)} page(s) in {xml_dir}: {written} pair(s) {verb}", fg="green")


# ----------------------------------------------------------------------------- sbb export


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--out",
    "dest",
    type=click.Path(dir_okay=True, file_okay=True, allow_dash=False, path_type=Path),
    default=None,
    help="Target file or directory; default: model.pt beside the source.",
)
def export(checkpoint: Path, dest: Path | None) -> None:
    """Rewrite a legacy ``.pth``/``.ckpt`` into the current ``model.pt`` schema.

    ``build_model_load_pretrained_weights_and_save.py`` did this by hand: it built a model, loaded
    weights, and saved them under a key the next script had to guess.
    :func:`~sbb_pytorch.storage.checkpoint.load_checkpoint` normalises every spelling in the wild
    (``model_state_dict``, ``state_dict``, ``model``, a bare state dict) into the
    :class:`~sbb_pytorch.storage.checkpoint.Checkpoint` schema and writes the ``*_config.json``
    sidecar every other reader expects, so an old run becomes something ``sbb infer`` accepts.
    """
    source = checkpoint if checkpoint.is_file() else _resolve_model_path(checkpoint)
    # A pre-schema file loads as ``schema_version=0`` / ``package_version="unknown"`` -- true of
    # the file that was read, false of the one being written. This file *is* schema 1 and this
    # package wrote it, so the two fields are re-stamped before the save.
    payload = replace(load_checkpoint(source), schema_version=1, package_version=__version__)
    target = Path(dest) if dest is not None else source.parent
    if target.is_dir():
        target = target / "model.pt"
    written = payload.save(target)
    click.secho(f"wrote {written}", fg="green")
    click.echo(f"  task             {payload.task or 'unknown'}")
    click.echo(f"  epoch / step     {payload.epoch} / {payload.step}")
    click.echo(f"  parameters       {len(payload.model_state)} tensor(s)")
    click.echo(f"  config embedded  {'yes' if payload.config else 'no'}")
    click.echo(f"  metrics          {len(payload.metrics)} key(s)")


# ------------------------------------------------------------------------ sbb config-schema


@main.command("config-schema")
@click.option(
    "-o",
    "--out",
    "dest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=f"Where to write; default {DEFAULT_SCHEMA_PATH}.",
)
def config_schema(dest: Path | None) -> None:
    """Write the JSON schema of ``SBBConfig`` -- the generated config reference."""
    written = dump_json_schema(Path(dest) if dest is not None else DEFAULT_SCHEMA_PATH)
    click.secho(f"wrote {written}", fg="green")


if __name__ == "__main__":  # pragma: no cover
    main()
