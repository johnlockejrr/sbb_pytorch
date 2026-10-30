# SBB PyTorch

PyTorch implementation of the SBB (Staatsbibliothek zu Berlin) pixelwise document-image
segmentation models -- layout, textline, binarization, printspace, glyph/word -- plus the two
adjacent tasks that grew out of the same codebase: image enhancement and machine-based reading
order.

Before 0.2.0 this repository was sixteen loose scripts in the root directory (`train.py`,
`inference_pytorch.py`, `generate_gt_for_training.py`, `config_loader.py`, a 49 KB `utils.py`),
each with its own argument parsing, its own config reader and its own copy of the image
utilities. It is now the installable `sbb_pytorch` package with a single command, `sbb`.
[TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md) maps every old file to its replacement, and
[docs/migration.md](docs/migration.md) lists every behaviour that changed -- read it before
re-running a training job you need to compare against a published number.

## What it does

| Task (`SBBConfig.task`) | Architecture (`sbb_pytorch.models.MODEL_REGISTRY`) | Output |
| --- | --- | --- |
| `segmentation` | `resnet50_unet`, `resnet50_unet_light`, `vit_resnet50_unet`, `vit_resnet50_unet_transformer_before_cnn` | per-pixel class mask |
| `binarization` | `resnet50_unet` (the two-class subset of segmentation) | foreground/background mask |
| `classification` | `resnet50_classifier` | one label + per-class probabilities |
| `enhancement` | `enhancement` (`EnhancementModel`) | a rendered image |
| `reading_order` | `machine_based_reading_order` | a region ordering |

The five SBB use cases -- `textline`, `layout`, `printspace`, `glyph`, `word` -- are
`UseCase` members, set in the page-to-label config rather than the task: they decide which ALTO or
PageXML regions become which class id.

## Requirements

Python 3.10 or newer. `torch` and `torchvision` are declared as dependencies, but pip cannot
choose the right PyTorch build for you -- CPU, cu118, cu121 and ROCm builds all satisfy
`torch>=2.0`, and only one of them matches your machine. Install the build first, then the package.

`safetensors>=0.4` is a dependency too: `Checkpoint.save`/`load` write and read `.safetensors`
besides the `.pt` format, and an explicit `.safetensors` request on a machine without the package
names the missing package instead of quietly writing a pickle.

## Installation

With [uv](https://docs.astral.sh/uv/) (the setup this repo is developed with):

```bash
uv venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
```

For a CUDA box, swap the index URL for the one matching the driver
(`https://download.pytorch.org/whl/cu121`, `cu118`, ...) and install that before the editable
install; with plain `python -m venv` the same two commands work unchanged.

Then verify the entry point from [pyproject.toml](pyproject.toml) (`sbb = "sbb_pytorch.cli:main"`)
is on `PATH`:

```bash
sbb --version
```

Without an editable install, every command below can be run as
`python -m sbb_pytorch.cli ...` from the repository root.

## Quick start

```bash
sbb validate configs/textline.json          # check the config before spending an hour on it
sbb train configs/textline.json --dry-run   # one train + one validation batch, nothing written
sbb train configs/textline.json -o runs/textline
sbb infer runs/textline -i pages/           # writes pages' predictions next to the run
```

## The `sbb` command

Every subcommand reads the same [pydantic](https://docs.pydantic.dev) model, `SBBConfig`; nothing
on the command line is a second source of truth. The options that overlap the config file
(`--epochs`, `--batch-size`, `--device`, `--patience`, `--resume`) are resolved by
`sbb_pytorch.config.apply_overrides`: **flag, then file, then field default**. The legacy
`train.py` had the same flags but a separate `get_arg_with_priority` helper for them, and several
of them -- `--resume`, `--num-workers` for some tasks -- had no config field to write back to.

### `sbb train CONFIG`

Builds the model, the Lightning module, the trainer and the callbacks from one file, runs
`trainer.fit`, then writes `model.pt` -- and beside it `model.safetensors`, the run's best weights,
when the `safetensors` dependency is installed.

| Option | Meaning |
| --- | --- |
| `CONFIG` | Config file, e.g. `configs/textline.json` (see [Configurations](#configurations)) |
| `-m, --model` | Registry key overriding the architecture the config implies |
| `--pretrained / --no-pretrained` | ImageNet weights; default follows `config.pretraining` |
| `-d, --device` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps` -- validated at parse time by `sbb_pytorch.device.resolve_device`, so a device that cannot run is refused before a model is built (a `click.Choice` could not express `cuda:1`; a hard-coded `cuda` default crashed CPU-only boxes) |
| `-o, --output-dir` | Run directory; default `config.dir_output` |
| `-e, --epochs`, `-b, --batch-size`, `-w, --num-workers`, `--lr`, `--patience` | Field overrides (`n_epochs`, `n_batch`, `num_workers`, `learning_rate`, `patience`) |
| `--resume` | Checkpoint to resume from; the config's `dir_of_start_model` when omitted -- in `train.py` the flag existed and the value was never passed to `fit(ckpt_path=...)` |
| `--start-weights` | Finetune: a `.pt`, `.pth`, `.ckpt` or `.safetensors` whose weights the model is built from, replacing the config's `init_method` and any ImageNet download. A *new* run -- the optimiser and epoch counter do not come from the file -- so it is refused next to `--resume`, next to an explicit `--pretrained`/`--no-pretrained`, and next to a config whose `continue_training` is true |
| `--page2label`, `--scales` | Attach a page-to-label / enhancement-scales file (or use a config with them inlined) |
| `--every-n-steps` | Also save a mid-epoch checkpoint every N batches (the old `--save-interval`, whose callback never fired) |
| `--logger / --no-logger` | TensorBoard logs when tensorboard is importable, CSV logs otherwise |
| `--progress-bar / --no-progress-bar` | The `rich` bar pytorch-lightning uses when `rich` is installed |
| `--verbose-metrics / --no-verbose-metrics` | Print `Metric val_loss improved. New best score: 0.449` on every improvement, as `training.py` did. Off by default: the bar carries the same numbers as `early_stopping/val_loss: 3/10` |
| `--dry-run` | One train and one validation batch, no `model.pt`: turns a bad config into a two-second failure |

### `sbb infer MODEL [INPUT...]`

| Option | Meaning |
| --- | --- |
| `MODEL` | Run directory (its `model.pt`, else the newest checkpoint) or a checkpoint file |
| `INPUT` / `-i, --input` | Pages to predict on: files and/or directories, repeatable |
| `-o, --output-dir` | Default `predictions/` beside the checkpoint |
| `--save / --no-save`, `--save-layout / --no-save-layout` | Which renderings to write; asking for one the task did not produce is an error, not a silently missing file |
| `--patches / --no-patches` | Sliding-window inference (`--patches` was a `pass` statement in `inference_pytorch.py`) |
| `-c, --config` | Config used to rebuild the architecture; default is the checkpoint's own `model_config.json`, which is what lets `sbb infer runs/textline -i pages/` work without repeating every training option |
| `--architecture` | Registry key override (in one legacy script `--model` meant this, in another it meant the weights path) |
| `-d, --device`, `--limit` | Device as above; stop after N pages |

### `sbb gt generate` and `sbb gt visualize`

`sbb gt generate --xml DIR -o DIR` replaces `generate_gt_for_training.py`: give it either
`--config` (a `GroundTruthConfig` JSON, the shape of `custom_config_page2label.json`) or
`--use-case`, which the old script made you edit as a constant. `--type-output` is `2d` or `3d`
(`--type` before), `--printspace` and `--min-area` carry over, and `--dry-run` lists the XML files
it found without writing anything. A `rich` bar tracks the files when stdout is a terminal.

`sbb gt visualize XML [-o OUT] [--size H W] [--type-output 2d|3d]` renders a single ALTO/PageXML
page -- the view that previously required opening the notebook.

### `sbb export CHECKPOINT`

Rewrites a checkpoint into the current schema. The repo had three layouts in simultaneous use
(`model_state`, `model_state_dict`, bare state dicts) and each module could read only one of them;
`Checkpoint.load` accepts all three and writes `model.pt` plus the `model_config.json` sidecar that
`sbb infer` reads. This is what `build_model_load_pretrained_weights_and_save.py` -- a Keras script
in a PyTorch repo -- used to do by hand. `-o` defaults to `model.pt` beside the source.

### `sbb validate CONFIG` and `sbb config-schema`

`validate` loads the config, prints `is a valid SBBConfig (task)` and then the advisories: the
shipped pair `config_params.json` + `custom_config_page2label.json` declares `n_classes=2` while
its mapping uses ids up to 6. Those are printed, not raised -- refusing to start would break runs
that exist today -- unless `--strict`, which exits 1 so CI can fail on them. A config that fails
*schema* validation always exits 1: that is the error `train.py` found at model-build time, after
the data had already been read.

`config-schema -o docs/config-reference.json` writes `SBBConfig.model_json_schema()`. The
hand-written option table it replaces had already drifted from the code; a generated file cannot.

## Configurations

Six ready-made files in [configs/](configs/), each loadable by `sbb_pytorch.config.load_config`:

| File | Notes |
| --- | --- |
| [configs/textline.json](configs/textline.json) | from `config_train_textline.json`, plus the ten keys that file omitted and `ConfigLoader` rebuilt by hand (their absence was a `KeyError`); `n_classes` is `7`, the count its `page2label` ids (1-6, background included) need -- the legacy pair said `3` against ids up to `6`, which `sbb validate` reported and the legacy code trained on anyway |
| [configs/layout.json](configs/layout.json) | from `config_params.json`; `num_patches_xy` corrected to `[112, 112]` for a 448 input; `n_classes` is `7` for the same reason, and `n_epochs` is `50` -- the legacy `0` asks Lightning for a zero-epoch run, which trains nothing rather than "validating only" |
| [configs/binarization.json](configs/binarization.json), [configs/enhancement.json](configs/enhancement.json), [configs/classification.json](configs/classification.json) | new; no legacy file existed for these tasks, so their numbers are chosen and documented, not ported |
| [configs/reading_order.json](configs/reading_order.json) | new, and **inference-only**: the port has no trainable reading-order loss, so `sbb train` on it raises `NotImplementedError` by design -- the order comes from `sbb gt` and `sbb infer` |

The data directories in all six are placeholders (`data/<task>/{train,eval}`, `runs/<task>`) --
point them at real data or pass `-o`. Every value that differs from the legacy file, and why, is
listed in [docs/migration.md](docs/migration.md) and
[TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md). The full field reference is
[docs/config-reference.json](docs/config-reference.json), generated by `sbb config-schema`.

## Outputs and checkpoints

A training run in `runs/textline/` contains:

```
runs/textline/
  model.pt                  final weights, config and metrics in one schema
  model.safetensors         the *best* weights, in the format with no pickle in it
  model_config.json         the same config as JSON, for tooling that will not unpickle
  model-07-0.3141.ckpt      epoch checkpoints, best-by-val_loss   (top_k=3 kept)
  best-07-0.9812.ckpt       epoch checkpoints, best-by-val_accuracy
  steps/model_step_1200.pt  mid-epoch checkpoints, when --every-n-steps is used
  lightning_logs/           TensorBoard, or CSV when tensorboard is not installed
```

`model.pt` holds the **last** epoch and `model.safetensors` the **best** weights, read back from the
`best-*.ckpt` the run's own callbacks chose; two files, because putting the best weights into
`model.pt` would move a file every existing reader opens without telling them which of the two they
got. Either one -- or another run's `best-07-0.9812.ckpt` -- can start a new run through
`sbb train --start-weights`.

`train.py` wrote `model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.ckpt` and then read
`callback_metrics["val_acc"]` to build `model.pt`, so any task that never logs `val_acc`
(enhancement, reading order) raised `KeyError` *after* a full run. The metrics are now named
`val_loss`, `val_accuracy`, `val_miou`, `val_dice_macro`, `val_dice_micro` -- the rename is a
breaking change for anyone reading old TensorBoard runs, and `SBBLightningModule.logged_val_keys`
is the single source of the names a run produces.

## Logging

`sbb_pytorch.logging.get_logger` returns one cached logger per name with a console handler (a
`rich` handler when importable) and, if asked, one timestamped file handler -- calling it twice no
longer duplicates every subsequent line, which is what the old `train.py` logger did. `-v` on the
group raises the level to DEBUG; `inference_pytorch.py` and `utils.py` used `print()` instead
(~200 calls between them, several in `except:` blocks that also swallowed the error).

## Development

```bash
.venv/bin/python -m pytest                      # the suite in tests/
~/.local/bin/ruff check src tests               # lint (config in pyproject.toml)
~/.local/bin/ruff format --check src            # formatting, line length 100
.venv/bin/python -m mypy                        # type checks, pydantic plugin
```

`src/sbb_pytorch/py.typed` marks the package as typed; `mypy` runs over `src` with
`strict = false` and the pydantic plugin, per [pyproject.toml](pyproject.toml).

Contributions are welcome: open a pull request against `main`, and if the change moves a number,
add the entry to `docs/migration.md` in the same commit.

## License

Apache-2.0 -- see [LICENSE](LICENSE). The `LICENSE` file has always been Apache-2.0; the previous
README said MIT, which was simply wrong, and `pyproject.toml` now declares
`license = "Apache-2.0"` with `license-files = ["LICENSE"]`.

## How to cite

If you find this tool useful in your work, please consider citing our paper:

```bibtex
@inproceedings{hip23rezanezhad,
  title     = {Document Layout Analysis with Deep Learning and Heuristics},
  author    = {Rezanezhad, Vahid and Baierer, Konstantin and Gerber, Mike and Labusch, Kai and Neudecker, Clemens},
  booktitle = {Proceedings of the 7th International Workshop on Historical Document Imaging and Processing {HIP} 2023,
               San Jos{\'e}, CA, USA, August 25-26, 2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2023},
  pages     = {73--78},
  url       = {https://doi.org/10.1145/3604951.3605513}
}
```

The reference implementation this port derives from is
[qbayee/sbb_pixelwise_segmentation](https://github.com/qbayee/sbb_pixelwise_segmentation)
(TensorFlow); the PyTorch port lives at
[cneud/sbb_pytorch](https://github.com/cneud/sbb_pytorch).
