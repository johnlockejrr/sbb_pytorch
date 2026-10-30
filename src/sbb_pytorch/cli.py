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

from sbb_pytorch import __version__
from sbb_pytorch.config import (
    SBBConfig,
    apply_overrides,
    dump_json_schema,
    load_config,
    load_page2label,
    validate_pair,
)
from sbb_pytorch.device import resolve_device
from sbb_pytorch.exceptions import CheckpointError, SBBError
from sbb_pytorch.gt import (
    GroundTruthConfig,
    # gt's own UseCase, not config.models.UseCase: the generator's has seven members
    # (binarization and reading_order among them) and GroundTruthConfig.use_case needs one.
    UseCase,
    discover_xml_files,
    generate_ground_truth,
    parse_pagexml,
    render_page,
    write_image,
)
from sbb_pytorch.inference import Prediction, Predictor
from sbb_pytorch.logging import get_logger, set_level
from sbb_pytorch.storage import latest_checkpoint, load_checkpoint
from sbb_pytorch.storage.images import list_images
from sbb_pytorch.training import SBBLightningModule, fit
from sbb_pytorch.training.callbacks import metric_snapshot

__all__ = ["main"]

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


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
    help="Architecture from the registry, overriding the config's own backbone.",
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
    page2label: Path | None,
    scales: Path | None,
    every_n_steps: int | None,
    logger: bool,
    progress_bar: bool,
    dry_run: bool,
) -> None:
    """Train from CONFIG; the output directory defaults to the config's ``dir_output``.

    The config is loaded, the flags are applied over it, and the advisories :command:`sbb
    validate` would print are printed *before* the run starts -- they used to be discovered as a
    crash halfway through the first epoch.
    """
    state: GlobalState = ctx.obj
    resolved = load_config(config, page2label_path=page2label, scales_path=scales)
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
    trainer = fit(
        resolved,
        module=module,
        device=device,
        resume=resume,
        output_dir=output_dir,
        save_final=not dry_run,
        logger="auto" if logger else False,
        enable_progress_bar=progress_bar,
        every_n_steps=every_n_steps,
        **({"fast_dev_run": 1} if dry_run else {}),
    )

    checkpoint = Path(output_dir if output_dir is not None else resolved.dir_output) / "model.pt"
    if dry_run:
        click.secho(f"dry run finished; {checkpoint} not written", fg="green")
        return
    click.secho(f"wrote {checkpoint}", fg="green")
    for name, value in sorted(metric_snapshot(trainer).items()):
        click.echo(f"  {name:<24} {value:.6f}")


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
    default=True,
    help="Write the image (or, for label-only tasks, the JSON answer).",
)
@click.option("--save-layout/--no-save-layout", default=False, help="Also write the label map.")
@click.option("--patches/--no-patches", "patches", default=None, help="Default: the config's.")
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
    save: bool,
    save_layout: bool,
    patches: bool | None,
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

    written = 0
    with _progress(len(files), "predict") as advance:
        for path in files:
            prediction = predictor.predict_file(path, patches=use_patches)
            written += _save_prediction(
                prediction, path, destination, save=save, save_layout=save_layout
            )
            advance(path)
    state.logger.debug("predictor built from %s", checkpoint)
    click.secho(
        f"{len(files)} page(s) predicted from {checkpoint}"
        + (f", {written} file(s) written to {destination}" if written else ", nothing saved"),
        fg="green",
    )


def _save_prediction(
    prediction: Prediction, source: Path, destination: Path, *, save: bool, save_layout: bool
) -> int:
    """Write whatever this prediction actually produced; return the number of files written.

    Segmentation and enhancement produce images, which
    :meth:`~sbb_pytorch.inference.predictor.Prediction.save` writes. Classification and reading
    order produce a label and probabilities and no image at all -- ``Prediction.save`` raises
    rather than pretend otherwise, and the legacy script wrote nothing at all -- so their answer
    goes to ``<stem>_prediction.json``. Without this, ``sbb infer`` on a classification checkpoint
    failed on its own default ``--save``.
    """
    if prediction.mask is None and prediction.image is None:
        if not save or prediction.label is None:
            return 0
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
        return 1
    if not (save or save_layout):
        return 0
    return len(
        prediction.save(
            save=str(destination / f"{source.stem}_prediction.png") if save else None,
            save_layout=(str(destination / f"{source.stem}_layout.png") if save_layout else None),
        )
    )


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
    """Ground truth: generate training images from ALTO/PageXML, or render one page."""


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
@click.option("--type-output", "output_type", type=click.Choice(["2d", "3d"]), default="3d")
@click.option("--printspace/--no-printspace", default=False)
@click.option(
    "--images-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--images-output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option(
    "--min-area", type=float, default=None, help="Smallest contour kept, as a page fraction."
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
    output_type: str,
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
    """
    if config_path is not None and use_case_raw is not None:
        raise click.BadParameter("give --config or --use-case, not both")
    if config_path is None and use_case_raw is None:
        raise click.BadParameter("give --config or --use-case")

    gt_config: GroundTruthConfig | Path = (
        config_path
        if config_path is not None
        else GroundTruthConfig(
            use_case=UseCase(use_case_raw or "layout"),
            **({"min_area": min_area} if min_area is not None else {}),
        )
    )
    if config_path is not None and min_area is not None:
        raise click.BadParameter("--min-area is set in the --config file")

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
            output_type=output_type,
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
    help="Where to write; default docs/config-reference.json.",
)
def config_schema(dest: Path | None) -> None:
    """Write the JSON schema of ``SBBConfig`` -- the generated config reference."""
    written = dump_json_schema(Path(dest) if dest is not None else "docs/config-reference.json")
    click.secho(f"wrote {written}", fg="green")


if __name__ == "__main__":  # pragma: no cover
    main()
