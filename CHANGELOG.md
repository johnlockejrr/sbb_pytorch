# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[semantic versioning](https://semver.org/spec/v2.0.0.html).

The version history has two entries. `0.1.0` is the state of the repository before the refactor:
sixteen loose scripts in the root directory, the TensorFlow-to-PyTorch port. `0.2.0` is the
packaged rewrite on `refactor/packaged-rewrite`, and its entry describes that change -- the older
changelog text this file used to hold is summarised under 0.1.0 and, in full, in git.

## [0.2.0] - 2026-10-29

The refactor from root-level scripts to the installable `sbb_pytorch` package. The rule the whole
rewrite was written under was *behaviour preservation outranks elegance*: every change that moves a
number is listed in [docs/migration.md](docs/migration.md), which is the sign-off sheet for this
release, and [TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md) maps each old file to its
replacement.

### Added

- The `sbb_pytorch` package under `src/` with a hatchling build, `py.typed`, and the `sbb` console
  script (`sbb = "sbb_pytorch.cli:main"`). `pip install -e .` makes the package importable from
  anywhere; the empty root `__init__.py` that made the source tree an accidental module is gone.
- `sbb`, one `click` group, replacing three argument-parsing styles (`argparse` in `train.py`, a
  bare `click.command` in `inference_pytorch.py`, its own `click.group` in
  `generate_gt_for_training.py`): `train`, `infer`, `gt generate`, `gt visualize`, `gt enhance`,
  `gt reading-order`, `export`, `validate`, `config-schema` -- every command the legacy scripts
  had, including `image_enhancement` and `machine_based_reading_order`, which were ported as code
  first and given flags in this batch.
- `SBBConfig`, `Page2Label` and `ScalesConfig` as pydantic v2 models, replacing eight dataclasses
  and the stateful `ConfigLoader` class. Every field has a type, a default and, where meaningful,
  a constraint; `dump_json_schema` writes `docs/config-reference.json`, replacing the
  hand-written option table that had already drifted from the code.
- `Checkpoint` in `storage/checkpoint.py`: one schema for the three checkpoint layouts that were in
  simultaneous use (`model_state`, `model_state_dict`, bare state dicts), plus
  `load_checkpoint`, `save_checkpoint`, `latest_checkpoint`, `prune_checkpoints` and the
  `model_config.json` sidecar.
- `SBBDataModule` with a real `prepare_data`/`setup` split, `SegmentationDataset`,
  `PatchConfig`/`preprocess_and_augment_data`, and one augmentation factory gated on the config.
- `SBBLightningModule` -- one module for every task, replacing the single `SegmentationModel` that
  `train.py` used for all six. `training/factory.py` is the only optimizer/initializer/scheduler
  builder; `training/callbacks.py` derives its monitored metrics from what the module actually
  logs; `training/trainer.py` assembles `pl.Trainer` and the `fit` the CLI calls.
- `models/base.py`: one `UNetDecoder` plus four thin assemblies and a `MODEL_REGISTRY` of seven
  architectures, replacing seven classes built from five copied decoders. It builds the one skip
  pairing the upstream model uses -- each block with the encoder level of the same name,
  `[f5, f4, f3, f2, f1]` -- rather than the port's `[f5, f3, f2, f1, f1]`.
- `metrics/`: one `Metric` interface over `functional.py`, `segmentation.py` and `enhance.py`,
  replacing five ways of computing two numbers. `losses.py` is now a thin `nn.Module` layer over
  it, and the losses that existed only as plain functions (`SoftDiceLoss`, `TverskyLoss`,
  `FocalTverskyLoss`, `GeneralizedDiceLoss`, `JaccardLoss`) are constructible modules.
- `gt/`: one ALTO/PageXML parser, one contour-geometry module, one renderer, one generator and the
  reading-order pair builder, replacing five near-identical XML readers (~1000 duplicated lines)
  and five rasterising functions.
- `sbb_pytorch.device` (`resolve_device`, `available_devices`) and `sbb_pytorch.exceptions`
  (`SBBError` and its typed subclasses), so a bad device string or a stale config is one
  explanatory line instead of a traceback.
- `sbb_pytorch.logging.get_logger`: one cached logger per name, console plus optional timestamped
  file handler, replacing the `train.py` logger that appended a fresh `FileHandler` on every call
  and the ~200 `print()` calls spread across the old modules.
- The `tests/` suite: fourteen modules covering config, models, data, augmentations, metrics,
  losses, storage, inference, training (plus the training factory) and ground truth -- 816
  collected cases as of this batch.
  `tests/test_gt.py` is the last to land and the largest -- 2300 lines, 116 test functions, 252
  collected cases over `parser`, `geometry`, `render`, `reading_order`, `generator` and `writer`,
  none of which had ever been tested outside the `--dry-run` CLI paths of `tests/test_cli.py`
  (the two `--type-output` cases below are the first to run `sbb gt generate` for real there). Its
  expected values are derived from the geometry or from the legacy contract each docstring states
  rather than read back off the implementation. Eleven of its assertions disagreed with the code
  they covered; each was first pinned with a `# TODO(file.py:line)` marker and is now an ordinary
  assertion, and the eleven fixes are listed in
  [docs/migration.md](docs/migration.md) section 3.8 and under **Fixed** below.
- `configs/`: six ready-made configs (textline, layout, binarization, enhancement,
  classification, reading_order). The first two carry the values of `config_train_textline.json`
  and `config_params.json`; the other four tasks had no legacy config at all.
- `gt/writer.py`: `to_pagexml` / `write_pagexml` / `points_attribute`, the counterpart of
  `gt/parser.py` that the rewrite had never needed. It serialises exactly what `Page` models, in
  the order the parser reads it, so parse → write → parse is byte-stable, and it is deterministic
  on purpose: no `Created`/`LastChange` timestamp and no `uuid4()` ids, both of which the two
  upstream writers stamped on every file.
- `training.callbacks.load_callbacks_file` and `sbb train --callbacks-file`: a JSON list of dotted
  callback paths (or `{"class": ..., "init": {...}}` objects) imported, validated as
  `pl.Callback` subclasses and *appended* to the run's own set. Every legacy script hard-coded its
  callback list, so adding one meant editing the file.
- Four `sbb infer` flags the code behind them mostly had: `--save-logits/--no-save-logits` (the
  `.pt` form of the logits every `Prediction` already carries), `--overlap` (the sliding-window
  overlap, previously reachable only through the Python API), `--ground-truth` (the legacy
  `--ground_truth`, whose IoU the old script computed and threw away) and `--xml-file` (writes the
  answer back as PageXML through `gt.writer`, the legacy `--xml_file`).
- `sbb train --verbose-metrics/--no-verbose-metrics`, the flag that sets the `verbose` field of
  every `ModelCheckpoint` and `EarlyStopping` a run builds, plus the two callbacks that make a
  quiet run still readable: `training.callbacks.SBBProgressBar` (the rich bar plus the
  `early_stopping/val_loss: 3/10` counters `early_stopping_fields` reads off the trainer) and
  `training.callbacks.SBBModelSummary` (the same summary table without its `FLOPs` column and the
  `Total FLOPs: 0` line under it).
- **`.safetensors` checkpoints: `Checkpoint.save_safetensors` and `Checkpoint.load_safetensors`.**
  `save()` and `load()` pick the format from the extension now, so a run can publish
  `model.safetensors` beside `model.pt` without a second schema. The format holds tensors plus one
  JSON string, so every field except `optimizer_state` moves into `__metadata__` as flat `sbb.*`
  keys (`format`, `sbb.schema_version`, `sbb.package_version`, `sbb.task`, `sbb.epoch`, `sbb.step`,
  `sbb.config`, `sbb.metrics`, `sbb.pytorch_lightning_version`) -- which is why a `.safetensors`
  start means a *fresh* optimizer rather than a resumed one, why a file written by another tool
  still loads (as `schema_version=0`, `package_version="unknown"`) instead of being refused for the
  keys it never had, and why a `.safetensors` needs no `_config.json` beside it. `safetensors>=0.4`
  joins the runtime dependencies behind a `safetensors_available()` guard -- the
  `tensorboard_available()` shape, so a machine without it is tested by patching one function -- and
  an explicit `.safetensors` save there is a `CheckpointError` rather than a silent pickle. Writing
  goes through `detach().cpu().contiguous()` with duplicates found by `data_ptr()` cloned, because
  safetensors refuses the shared storage (`Some tensors share memory`) and the non-contiguous views
  (`torch.save` accepts both) -- and an empty `model_state` is refused up front instead of becoming
  a weightless checkpoint.
- **`sbb train --start-weights FILE`, the finetuning path for weights that are not a
  pytorch-lightning `.ckpt`.** `training.apply_start_weights` loads a `.pt`, `.pth`, `.ckpt` or
  `.safetensors` into the module *before* the trainer is built and returns the `Checkpoint` it read.
  Only the weights move -- the optimiser and the epoch counter start fresh, which is the point of
  finetuning and the thing `--resume` never offered, since `trainer.fit(ckpt_path=...)` wants
  Lightning's own file and restores its optimiser, and `--pretrained` cannot express it, since that
  flag only chooses which ImageNet weights to download. The CLI refuses every pairing that answers
  one question twice -- `--resume`, an explicit `--pretrained`/`--no-pretrained`, a config whose
  `continue_training` is true -- and `fit` refuses `start_weights` alongside `resume` for the reason
  Lightning itself gives: the start file gets loaded and then overwritten.
- **`--format {pt,safetensors}` on `sbb infer` and `sbb export`.** On `infer` it names which
  checkpoint a *directory* means -- `pt` (the default: `model.pt`, then the other `.pt`/`.pth`, then
  `.ckpt`, then `.safetensors`) or `safetensors` (the exported file) -- and is ignored when `MODEL`
  is a file, because that file is the answer. On `export` it is the format written:
  `model.safetensors` beside the source instead of `model.pt`, with no `model_config.json` beside it
  because a safetensors file carries its own config. An `-o` that names a file keeps that file's
  extension whatever `--format` says.
- **`val_fraction` is a config field, and `sbb train --val-fraction` is its flag.** The share of
  `dir_train` handed to validation when `dir_eval` is empty used to exist only as
  `SBBDataModule(val_fraction=0.2)` -- an argument no config file could carry, so every run validated
  the default fifth of its training folder: 100 pages of eval on a 500-page dataset, scored every
  epoch for a number that arrives after the epoch it describes. `0.2` stays the default,
  `"val_fraction": 0.1` is the tenth the same run wanted, `--val-fraction 0.1` overrides it per run,
  and `0.0` (legal, `ge=0.0, lt=1.0`) hands everything to training.

### Changed

- Every legacy script is replaced by a module: see [TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md)
  for the file-by-file table, including the four root JSON configs (now `configs/*.json`),
  `requirements.txt` (now `pyproject.toml`), and `train.md` (folded into the README and the
  changelogs).
- **`--xml` folders are read recursively, and one page gets one label.** A folder whose
  children are datasets -- `smp-private/` holding one folder per export -- has no XML file of its
  own, and `discover_xml_files` listed only the files directly inside the folder it was given, so
  such a parent answered `no ALTO/PageXML files in ...` and a command that looked like it covered
  three datasets had read none of them. `sbb gt generate` and `sbb gt reading-order` now descend by
  default (`--recursive`, the new default, is what `discover_xml_files` does; `--no-recursive`
  restricts it to the folder itself), `--images-dir` is searched the same way so a page's photo is
  found in the folder its XML was found in, and the dry-run prints each path relative to `--xml`
  because the folder is now part of the answer.
- **A page filed in both formats is labelled once, from its PageXML** (`--dedupe-pages`, the
  default). `000123456-1800.xml` and its `000123456-1800.alto.xml` are one page written twice --
  the annotation and the OCR markup it was made on -- and the previous behaviour labelled the ALTO
  as a second page: two PNGs for one page, with the ALTO taking the legacy name `1800` (it sorts
  first) and the PageXML renamed `000123456-1800`. `parser.select_page_files` keeps the PageXML,
  and `parser.page_identity` names the page a file describes so that a folder holding *only* the
  ALTO of a page still yields it. `--no-dedupe-pages` labels every file, as before.
- **Outputs are named after the file that produced them, folders included.** A label was named by
  the upstream rule `name.split('-')[1].split('.')[0]`, which answers `039` for
  `901234567-039.xml` and `1800` for `000123456-1800.xml` -- a shortening two documents holding a
  page `039` cannot both use, which is why 2455 XMLs once became 476 labels. `gt.generator.file_stem`
  is the name now: `901234567-039.xml` → `901234567-039.png`, and a page found inside a subfolder
  of `--xml` carries that folder in the name (`mss-021/page-000.xml` → `mss-021_page-000.png`) so
  the four children of a parent dataset keep their pages apart in the flat `-o`. A file sitting
  directly inside `--xml` has no folder to carry and keeps its plain name. `--legacy-stem`
  (`gt.generator.output_stem`) writes the old shortening for a run that must reproduce names an
  earlier one wrote. The copied source image gets the same stem as its label, so the pair the
  training loader matches by name stays a pair.
- `--device` defaults to `auto` and is validated at parse time. `train.py` hard-coded `cuda` and
  called `pl.Trainer(accelerator=device, devices=1)`, which rejected the valid spec `cuda:1` and
  made the second GPU of a machine unreachable.
- Config flags and config fields are the same values. `train.py`'s `get_arg_with_priority` helper
  became `config.loaders.apply_overrides` (flag, then file, then default), and `--resume` --
  accepted and never forwarded to `fit(ckpt_path=...)` -- is a real field.
- Checkpoint file names: `model-{epoch:02d}-{val_loss:.4f}.ckpt` and
  `best-{epoch:02d}-{val_accuracy:.4f}.ckpt`, `top_k=3` each; mid-epoch checkpoints go in a
  `steps/` subdirectory instead of one directory per step.
- The `Metric val_loss improved. New best score: 0.449` lines are off by default, and the bar now
  carries what they said. Both came from `training.py`'s hard-coded `EarlyStopping(verbose=True)` --
  one line per monitored metric per epoch, `Metric ... improved by 0.047 >= min_delta = 0.0. New
  best score: 0.402` after the first -- and `early_stopping_for` takes `verbose` now, default
  `False` like the `ModelCheckpoint` beside it, both set together by `sbb train --verbose-metrics`.
  The same numbers sit on the progress bar instead: `val_loss: 0.281 train_loss: 0.306
  val_accuracy: 0.892 early_stopping/val_loss: 3/10 early_stopping/val_accuracy: 0/10`.
- `Total FLOPs: 0` is gone from the model summary, the table above it unchanged. Lightning can only
  count FLOPs when the module exposes `example_input_array`, which `SBBLightningModule` has never
  had, so the line reported `0` for a model with real convolutions on every run; `SBBModelSummary`
  renders the same table without the `FLOPs` column and is handed to `pl.Trainer` together with
  `enable_model_summary=False`, which a summary callback and the bool both being set would
  otherwise have made Lightning warn about.
- Metric names: `val_acc` becomes `val_accuracy`, and the accuracy/IoU/Dice set is replaced by one
  confusion matrix reporting `val_loss`, `val_accuracy`, `val_miou`, `val_dice_macro`,
  `val_dice_micro` and the per-class keys. Reading an old TensorBoard or CSV run by the old keys
  needs the new ones; **there is no flag that gets the old numbers back.**
- Metrics are computed once per epoch from pooled counts instead of `compute()` on every batch, so
  a logged epoch value no longer depends on batch size and order.
- Every model returns raw logits. `ResNet50UNet` returned logits for segmentation and `sigmoid`
  otherwise while four other classes returned `softmax`, so `CrossEntropyLoss` normalised twice on
  those and they trained on a flattened loss surface.
- `configs/layout.json`: `num_patches_xy` `[56, 56]` (the legacy `transformer_num_patches_xy`, a
  224x224 grid behind a 448 input) becomes `[112, 112]`; `"blur_k": ["blur", "guass", "median"]`
  is corrected to `"gauss"`; `page2label.use_case` in `configs/layout.json` becomes `layout`.
- `requirements.txt` → `pyproject.toml`: `sacred` dropped (only the Keras build script used it),
  and with it the seven declared packages `src/` never imports -- `shapely`, `matplotlib`,
  `seaborn`, `scikit-learn`, `scipy`, `tqdm`, `Pillow`. Three of those still install through other
  dependencies (`scipy` via `albumentations`, `tqdm` via `pytorch_lightning`, `Pillow` via
  `torchvision`), so the install actually shrinks by `shapely`, `matplotlib`, `seaborn` and
  `scikit-learn`. Floors raised where the code requires it (`numpy>=1.24,<3`,
  `albumentations>=1.4.5`), `pydantic`, `click`, `rich` and `tensorboard` added.
  `torch`/`torchvision` are declared but the active build still has to be chosen before install.
- `ruff format` now passes across `src/` and `tests/` (eleven files were reformatted; `ruff check`
  had been clean the whole time, which is why a lint-only gate never showed it), the two builtin
  `callable` annotations in `training/factory.py` are `Callable[..., Any]`, and `exceptions.py`
  gained the `__all__` every other root module has. `pyproject.toml`'s `[project.urls]` point at
  the repository the package is published from.
- README license section corrected from MIT to Apache-2.0, which is what `LICENSE` has always
  contained, and `pyproject.toml` declares `license = "Apache-2.0"`.
- **`UseCase` and `OutputType` are declared once each.** `config/models.py` and `gt/generator.py`
  both defined a `UseCase` -- five members in the schema, seven in the generator, so
  `sbb gt generate --use-case binarization` was refused by the schema the same command reads --
  and `type_output` was a `Literal["2d", "3d"]` in the schema with a second, equivalent enum in
  `gt/render.py`. `config.models.UseCase` and `config.models.OutputType` are now the only
  definitions (with `TypeOutput` kept as an alias and `gt.render.OutputType` still importable), and
  `SBBConfig.json_schema()` -- a wrapper around `model_json_schema()` whose docstring named a
  `docs/config-reference.md` that had never existed -- is gone. `docs/config-reference.json` is
  regenerated from the schema, as `sbb config-schema` has always written it.
- **`configs/layout.json` sets `adding_rgb_background` and `adding_rgb_foreground` to `false`.**
  They were `true` on a config that names no `dir_rgb_backgrounds` or `dir_rgb_foregrounds`. The
  consequence is in [docs/migration.md](docs/migration.md) section 3.4: layout pages stop being
  composited onto random RGB backgrounds, and setting the two directories plus the flags back to
  `true` restores them.
- **`sbb infer --save/--no-save` now consults `save_visualisation`** when neither side of the flag
  is given. The field existed on `SBBConfig` and was read by nothing, so a config asking not to
  save images saved them anyway; passing `--save` or `--no-save` still wins, as it does for
  `--patches`.
- **`configs/textline.json` says `n_classes: 2`, the count upstream's textline pairing carries.** It
  had been moved to 7 -- `configs/layout.json`'s number, from the migration row that covered both
  configs with one figure -- and 7 is a layout table's count, not this one's: a single-class
  textline model has background plus the one class it predicts, so the head carried five channels
  nothing supervised and they argmaxed onto every page as `marginalia`/`<imageregion>`/
  `rest_as_decoration` speckle over the lines the model was trained to find. The head width is a
  function of the field, so the textline checkpoint trained under 7 stops loading and that run needs
  a retrain. `Page2Label` gained the five fields `custom_config_page2label.json` carries and the
  model lacked (`case_label`, `artificial_class_label`, `printspace_as_class_in_layout`,
  `tableregion`, `noiseregion`) so a single-class config can name its own class instead of borrowing
  the layout ids, and a flat legacy file with `case_label` beside its `use_case` now folds into
  `page2label` instead of being left at the top level for `extra="forbid"` to reject. The layout
  tables stay in the file beside them, so `sbb validate` reports the ids above 1 -- the state
  upstream ships; clearing them is the follow-up. `docs/config-reference.json` is regenerated.
- **`latest_checkpoint` ranks by name and `prune_checkpoints` stops eating `model.pt`.** Ranking one
  directory by `mtime` answered "which file was touched last" where the question is "which
  checkpoint is this": `model.pt` is the published file and the epoch files are its history. The
  order is `model.<ext>`, then the other `.pt`/`.pth`, then `.ckpt`, then `.safetensors`, with
  `mtime` ordering ties and the filename breaking ties in `mtime` -- and a `prefer=` keyword (which
  `sbb infer --format` will pass) promotes one extension. `prune_checkpoints` learns
  `.safetensors` and sorts epoch-less files after every epoch file: `model.pt` has no epoch number,
  the old key scored it `-1` -- the oldest thing present -- and `keep_last_n` deleted the published
  checkpoint before the history it was written from. `Checkpoint.save`'s `write_sidecar` becomes
  `bool | None = None` -- yes for a torch file, no for a safetensors one, whose config is already
  inside it -- so every existing call site keeps its meaning.
- **`sbb train` writes two files where it wrote one.** `model.pt` holds the last epoch, exactly as
  before, and `model.safetensors` -- beside it, whenever the `safetensors` package is importable,
  with `fit(save_safetensors=)` overriding -- holds the **best** weights, read back through
  `Checkpoint.load` from the `.ckpt` the run's own `ModelCheckpoint` callbacks call best
  (`training.best_checkpoint_path`: the `val_accuracy` file when there is one, since that is the
  metric `train.py` optimised and the file people point `sbb export` at; the `val_loss` file
  otherwise). `training.final_checkpoint` and `training.best_checkpoint` are the two answers, the
  first extracted from the block `fit` used to build inline. Publishing the best weights *through*
  `model.pt` was the alternative, and this is why it was refused: a reader could not tell which of
  the two it had been given, and every tool reading that file today would get a different answer
  without being asked. The CLI prints both paths.

### Fixed

Crashes and silently wrong values, grouped where they lived; the annotated list with the
breaking/fix/api markers is [docs/migration.md](docs/migration.md).

- **Models.** Four of the five copied decoders could not complete a forward pass (3072 channels fed
  into a `Conv2d(2048, 1024)`), and their skip pairing was off by one (`f4` unused, `f1` twice).
  Both ViT variants raised `RuntimeError` on their first batch. `TaskType.BINARIZATION` did not
  exist as an enum member, so a binarization config failed at parse time.
- **The head's missing stage.** The port kept the `Conv2D(n_classes, 1x1)` and dropped the stage in
  front of it -- the upstream head concatenates the input image onto the upsampled decoder output
  and runs a `Conv2D(32, 3x3) + BatchNormalization + ReLU` at full resolution, then normalises the
  logits on segmentation tasks -- so a "full resolution" prediction was an interpolation of a
  prediction made at `H/4`. `models.base.SegmentationHead` is that stage, composed by every
  segmentation model -- so the head's parameters are `head.fusion.*`, `head.classify.weight` and
  `head.norm.*` where the port had only `head.weight`/`head.bias` -- and
  `LIGHT_DECODER_CHANNELS` takes the widths the light decoder really has now that its 32-channel
  conv has moved back into the head.
- **The encoder's tap points.** `ResNetEncoder` takes `f1` from the **raw `conv1`** at `H/2` --
  upstream reads `f1 = x` between the conv and its own `BatchNormalization`, so the skip upstream
  hands the decoder is neither normalised nor rectified -- and lets `bn1 -> relu -> maxpool ->
  layer1` be level 1, which is what lets the head's upsample be a real 2x. `layer1`'s parameters
  move from `levels.1.*` to `levels.1.3.*`; an earlier version of this entry said `levels.1.1.*`,
  which described the pool-in-stem tap that came before it, and no checkpoint of ours exists for
  either.
- **Training.** `torch.save({... "val_acc": trainer.callback_metrics["val_acc"]})` raised `KeyError`
  *after* a full run for any task that does not log it, throwing away the result.
  `config.precision`, `config.accumulate_grad_batches` and `config.gradient_clip_val` were read by
  nothing. `ReduceLROnPlateau` was imported and never constructed, so no scheduler ever ran;
  `torch.optim.Adadelta` was built without `weight_decay`; `momentum` was passed to optimizers that
  do not take it; pretrained ImageNet weights were re-initialised with kaiming one line after being
  downloaded. `device.available_devices` called `torch.device_count()` -- an attribute that does not
  exist, on the one branch that only runs when CUDA is present, so it passed every test here and
  would have raised `AttributeError` on the machine `--device auto` is meant for.
- **Checkpoints.** `Checkpoint.load` could not read a pytorch-lightning `.ckpt`: it copied
  `state_dict` verbatim, where Lightning keys every parameter through the attribute holding it
  (`model.conv.weight`), so the `apply_to` immediately after the load raised
  `Unexpected key(s) "model.conv.weight"` -- which is exactly what `sbb export
  best-07-0.9812.ckpt` then handed to `sbb infer`. `Checkpoint.from_lightning` reads that layout
  now, with the `model.` prefix stripped, `step` taken from `global_step`, `optimizer_states[0]` as
  `optimizer_state`, and the Lightning version kept in a new `pytorch_lightning_version` field
  instead of `package_version`. Detection is by contents, not by suffix, so a `.ckpt` renamed to
  `.pt` is still recognised.
- **Callbacks.** `SaveWeightsAfterSteps` declared a signature pytorch-lightning never calls, so it
  never fired; `training.py`'s `ModelCheckpoint.save_model` read attributes its own `__init__`
  never assigned, so the first save raised `AttributeError`; pruning sorted `epoch_10` before
  `epoch_2` and deleted the newest files. Two checkpoints and two early-stopping callbacks were
  registered unconditionally and raised on any task that never logs `val_acc`.
- **Losses and metrics.** Tversky's `alpha`/`beta` were swapped (penalising false negatives where
  the config asked for false positives); focal loss used `alpha=4.0` in `metrics.py` and `0.25` in
  `losses.py` and diluted the loss by `1/n_classes`; focal-Tversky used an unpublished formula;
  Jaccard loss was multiplied by its smoothing constant; generalized dice merged every image in the
  batch; `ignore_index` filtered the result vector instead of the labels; `losses.SSIMLoss` raised
  `NameError` on construction, which made the only enhancement loss unreachable.
- **Data and images.** Training and inference fed the model different tensors for the same file
  (RGB via PIL, BGR via cv2 with RGB statistics); label loading collapsed 3-channel SBB masks by
  luminance, turning class ids `{1,2,3}` into `{10,90,187}`; `im.split('.')[0]` truncated
  `page.001.png` to `page`; `augmentation: false` was ignored, because the module that read the
  config was dead code; `preprocess_and_augment_data` ran in `DataModule.__init__`, so two
  processes wrote the same files at once; a missing image folder raised from a worker mid-epoch
  instead of at construction. `SegmentationDataset` added two faults of its own: it gave every task
  the same target -- an `int64 (H, W)` label map -- which neither the classifier (`(B, C)` logits) nor the enhancement loss (a float image
  target) accepts, so two of the six shipped configs failed on their first batch; and the
  page2label-derived remap omitted class id `0`, the value every background pixel of a rendered
  label carries, so a generated label image was rejected as "no class id configured".
- **Inference.** The output image was a thumbnail: the page was downsampled to the model's
  448x672 output instead of the mask being upscaled; a bilinear resize of the label map invented
  class ids; normalisation was applied before resizing and skipped entirely for classification;
  `--save` wrote nothing at all for classification; `--patches` and the whole `reading_order`
  branch were `pass` statements; `--ground_truth` was documented as a directory, read as a file,
  and its IoU computed and thrown away; `color_images` hard-coded three colours, rendering every
  class from 3 up black.
- **`sbb infer` loaded a different file than the one it was given.** `cli._resolve_model_path`
  resolved a checkpoint and then handed its *parent* to `Predictor.from_checkpoint`, which re-derived
  `directory / "model.pt"` from that -- so `sbb infer run/best-07-0.9812.ckpt page.png` predicted
  with `run/model.pt`, and a `model.safetensors` could not be used at all because the Predictor went
  looking for a `model_config.json` beside a file whose config is inside it. A directory with no
  checkpoint raised `TypeError: argument should be a str or an os.PathLike object` -- from
  `Checkpoint.load(None)`, which is what `latest_checkpoint` answers -- instead of naming the
  directory. `from_checkpoint` now takes the file as given, uses `latest_checkpoint` for a directory
  exactly as the CLI does so the two cannot disagree, and reads the config a `.safetensors` carries
  in its own metadata.
- **Ground truth.** The XML namespace prefix was derived as `"PcGts}"` for a namespace-less
  document, leaving `imageHeight`/`imageWidth` unbound; `config_params=None` crashed the layout
  branch although `--config` was documented as ignorable; `case_label` was applied to every region
  whatever the use case; the overlay blended with weights summing to 0.6, dimming the result.
  Five more, all found after the rewrite and before any release: `sbb gt generate` died in its own
  result summary (`vars()` on a `slots` dataclass) on every run that was not `--dry-run`;
  `--config` could not read a `configs/*.json` file because `GroundTruthConfig.from_dict` did not
  look inside its `page2label` section; `label_for` queried the raw-keyed region table with a
  normalised type name, so a `type="paragraph"` region under a `rest_as_paragraph` config labelled
  nothing and the page produced no output; `columns_width` keys were stringified while the lookup
  used an `int`, so the column-normalising resize never ran; and `--use-case` offered
  `config.models.UseCase`'s five members where the generator's own enum has seven, rejecting
  `binarization` and `reading_order`.
- **Eleven more, found by `tests/test_gt.py` and fixed in the same commit that retired its
  `TODO` markers.** `generate_pairs` could not build a single sample for any input size other than
  the page's own -- `np.stack` refused the page-sized label maps beside the resampled layout
  channel, so a 448×448 config on a 60×60 page raised `ValueError: all input arrays must have the
  same shape`; `_is_header` matched the literal `"header"` and gave no separator band to a
  `heading` or `rest_as_header` region; `write_pair` wrote `pair.name` and ignored the `suffix`
  property it documents; `label_image` range-checked its entry labels but passed `background`
  straight to `np.full`, where `999` raised `OverflowError`; a 1-pixel `outline_color` was
  swallowed by the fill it sat on, so it only appeared once the stroke grew past the fill; a
  digit-less `@ID` took its class id from `hash((kind, type)) % 19`, which CPython re-randomises on
  every interpreter start, so the same page rendered different colours on consecutive runs;
  `crop_to` let numpy clip the slice, returning a smaller crop than the box asked for; `from_dict`
  defaulted `case_label` to `artificial_class_label` before `1`; `_layout_map` painted regions in
  document order instead of the documented big-over-small, burying a header inside its own
  paragraph; `_find_image` preferred a page's `.xml` to its `.png` when both shared a stem; and
  `output_stem` produced `b-c` and `1800.tar` where the ported rule writes `b` and `1800`, so
  regenerating a dataset wrote a second file beside the existing one rather than overwriting it.
  The ported rule itself turned out to be the next defect: `split('-')[1]` is not *unique*, so a
  mixed folder -- two documents' page `0113`, a page's PageXML and its ALTO, which is what
  `sam_44_mss_mixed` is -- wrote 2455 XMLs as 476 label images, each file overwritten by the next
  one that wanted its name. `unique_output_stem` keeps the legacy answer for the first file to
  claim a stem (so a dataset generated before still matches what the next run writes) and
  disambiguates the rest to the whole file stem, then `_1`; the label and its copied image take
  one stem, which is how the dataset loader pairs them.
- **Config.** Any key a file omitted raised `KeyError` even when the value was irrelevant --
  `config_train_textline.json` was loadable through exactly one code path -- and a `" "`
  `dir_of_start_model` (the shipped way of writing "unset") is now `None`. `validate_configs`'
  three hand-written `if` checks become `SBBConfig.warnings()` and `validate_pair()`, which report
  rather than raise; `sbb validate --strict` turns the reports into an exit code. Two of the six
  shipped configs also had their numbers corrected after review: `configs/textline.json` and
  `configs/layout.json` now say `n_classes: 7`, the count their `page2label` ids (1-6, background
  included) require -- the legacy files they were built from said `3` and `2` against a mapping
  whose ids reach `6` -- and `configs/layout.json`'s `n_epochs` is `50` instead of the ported `0`,
  which asks Lightning for a run that trains nothing.
- **A second pass over the same code, found by reading it against the upstream sources rather than
  by the tests.** Every row is in [docs/migration.md](docs/migration.md) section 3; the headings
  repeat the ones above so the two documents line up.
  - *Models*: `ResNet50UNetLight` fused an unprojected `f4` where upstream runs a `Conv2D(512,
    (1,1))` first, repeating the width error that stopped that class from running at all
    (`UNetDecoder` gained `skip_projections`); `_MLPHead` built `hidden_units[0]` and discarded the
    rest, so the reading-order model's two 256-unit dense layers became one; both ViT classes
    stored `mlp_head_units` and built the MLP from `mlp_ratio` instead; and
    `ViTResNet50UNetTransformerBeforeCNN` showed its head a *reconstruction* of the page where
    upstream shows it the page.
  - *Inference*: `Predictor.predict(patches=True)` computed a whole-page forward pass, threw the
    result away and then indexed `[0]` off the tiled field -- which has no batch axis, so a patched
    prediction was one class wide instead of a page.
  - *Data*: `PatchConfig.from_sbb` forwarded eight of the eleven flags it has, and
    `augmentations.geometric_transforms` carried a branch that read `scaling_flip` and
    `scaling_bluring` and did `pass`.
  - *Ground truth*: `Coords/@points` raised on the fractional coordinates its own `<Point>` twin
    truncated; `ET.XMLParser(encoding=...)` overrode each document's XML declaration and decoded
    UTF-8 pages as iso-8859-5; `Metadata/Comments` turned the tool's `"#num_col0456, num_col1500"`
    into the width `4561500`; `float64` contours raised in `cv2.contourArea` although the module
    claimed to widen them; a scalar `fillPoly` colour painted one channel of a three-channel
    canvas; and `rotated_rect_with_max_area` had been collapsed to identities that only held at
    right angles.
- **`sbb gt generate --type-output` overruled the config it was handed, and the config could not
  answer it anyway.** The flag's own `3d` default went straight to `generate_ground_truth` while
  `GroundTruthConfig` had no `type_output` field at all, so a config asking for `2d` -- which is
  what `configs/textline.json` does -- still wrote the palette rendering, and `sbb train` on the
  result failed with `no class id configured for label values [255] in 7-class setup`: the label
  reader takes channel 0 for a class id and found class 1's colour there. `GroundTruthConfig`
  carries the field now, the flag defaults to `None` and falls back to it (naming the flag still
  wins, the precedence `sbb infer --save` set for `save_visualisation`), and the first tests to run
  `sbb gt generate` without `--dry-run` pin both directions. The three-channel shape of a `2d`
  label is upstream's own -- `gt_gen_utils.py:289` fills a `(H, W, 3)` array on that branch and
  `generate_gt_for_training.py:45` says one channel would do -- so `render.to_label_map` keeps it
  and now cites both.
- **The PageXML `sbb infer --xml-file` wrote could not be opened by anything else.** It failed
  `xmllint --schema pagecontent/2019-07-15` on six counts, and the two structural ones came from
  the parser and the writer agreeing with each other instead of with the schema. `gt.parser.Region`
  modelled five fields with no parent and no children and `_iter_regions` walked the document flat,
  so a `<TextLine>` nested inside a `<TextRegion>` came back as the *page's* child and went back out
  the same way -- `Element 'TextLine': This element is not expected`, once per line, which is why one
  annotated page whose reference holds **38 lines inside one `TextRegion`** came out of here as 239
  `TextRegion`s and no lines at all. `@custom`, `<TextEquiv @conf>` and `<Baseline>` were never read,
  and the `Metadata` the parser threw away took the annotation's `@externalRef`, `Created` and
  `LastChange` with it -- so a rewrite erased the provenance line the schema requires and the writer
  supplied none of it. `Region` has `children`/`parent` now (`compare=False` on `parent`, so
  `__eq__` does not recurse), `_iter_regions` walks pre-order through an `_ALLOWED_CHILDREN` table
  while `Page.regions` stays the flat list every existing caller reads, and an illegal pairing (a
  loose `TextLine`, our own old output) stays an orphan rather than being silently re-parented, so it
  survives the round trip. The writer emits `Page`'s children in the order `PageType` declares
  (`Border?`, `PrintSpace?`, `ReadingOrder?`, then the regions), writes the `@id` the schema requires
  even when the parsed one was empty (`<Element>_<n>`, counted per element name, which keeps the
  output byte-stable), gives the `<OrderedGroup>` the `@id` it had left off while its own
  `_ORDERED_GROUP_ID` sat unused in the same file, and writes `Created`/`LastChange` from the parsed
  page -- falling back to `--created`, and without that to `writer.timestamp_now()`: the current UTC
  second. The epoch (`1970-01-01T00:00:00Z`, `writer.DEFAULT_TIMESTAMP`, now gone) was a legal
  `xs:dateTime` that every tool opened and displayed as 1970, which is the difference between a date
  the schema accepts and a date the person opening the file believes.
- **`gt/vocabulary.py` is the PageXML element and `@type` table, and the schema is its authority.**
  Three places spelled these names before, and all three disagreed with the xsd: `gt.parser` kept ten
  text types and four graphic ones where the schema has eighteen and eleven (so a `caption`, `footer`
  or `credit` region read as untyped), `gt.writer` kept its own element-per-kind table, and
  `cli._class_name` joined *every* name a class id stood for into `@type="header, heading"` -- a
  value no version of this schema accepts, and invalid twice over on the elements that carry no
  `@type` at all (`ImageRegion`, `SeparatorRegion`, `TableRegion`, `NoiseRegion`, `Word`, `Glyph`,
  `Border`, `PrintSpace`, where `<TextRegion type="<imageregion>">` was the old output). The parser's
  `TEXT_REGION_TYPES`, `GRAPHIC_REGION_TYPES` and `REGION_TAGS` are aliases of the new table and
  `normalise_text_type` moved there rather than being copied a sixth time; `cli._class_name` is gone,
  its only caller asking `type_for`/`custom_for` instead -- `Predictor._class_names` keeps the comma
  join for `Prediction.label`, which is not an XML attribute. The table now answers a fourth set of
  names too: `class_1`, `class_2`, the placeholder `n_classes` gives a class id when no `page2label`
  table names one (`is_synthetic_class_name`). The xsd permits it in `@custom`, where any string is
  legal, and Aletheia does not -- the dialect it checks lists the classes a page may name, and a
  numbered stand-in is not one of them -- so it counts as the *absence* of a class and the region is
  named `structure {type:text_zone;}`, the word escriptorium itself writes for a block with no class,
  where this writer had written `structure {type:class_1;}`. `cli._regions_for_mask` now shapes the
  answer by use case (`textline` groups lines into blocks via `gt.geometry.group_contours_into_blocks`
  and `convex_hull` and nests them, `layout` writes one region per class, `printspace` and
  `binarization` write the `PrintSpace` outline alone instead of the bogus regions they used to add,
  and `word`/`glyph` answer `ClickException` naming the use case). The output of the reference page
  the user supplied validates, and a page written twice is the same bytes.
- **A single-class config refused the labels of its own class.** Training on
  `{"use_case": "textline", "case_label": 1}` with `n_classes: 2` -- the user's own dataset, and the
  shape `configs/textline.json` has -- died before the first batch with `no class id configured for
  label values [1] in 2-class setup`; the shipped file, whose layout ids survive beside `case_label`,
  failed one check later with `class id 2 for stored value 2 is outside [0, 2)`. Both came from
  `SegmentationDataset._resolve_mapping` deriving its `{stored_value: class_id}` map from
  `Page2Label.class_ids()`, which answers "which ids do the *regions* use" -- empty for a table that
  names its class in `case_label`, and for the shipped file a six-id table whose every entry past `1`
  exceeds the `n_classes` printed beside it. The pair of use cases that answers the question the
  config actually asks lived in `gt/generator.py`, which `config` cannot import, so
  `SINGLE_CLASS_USE_CASES` and `_CASE_LABEL_USE_CASES` move to `config/models.py` as the public
  `SINGLE_CLASS_USE_CASES` / `CASE_LABEL_USE_CASES` (the generator imports them), `Page2Label` gains
  `legal_class_ids()` -- `{0, case_label}` for those use cases, `{0, *class_ids()}` for the rest, so
  a layout config's mapping is exactly what it was -- and the loader asks it.
- **A textline model with no boundary class fused every line into one.** The fix above left
  `configs/textline.json` training on `{0, 1}`, which is a *two*-class model, and a two-class
  textline model has no separators: `gt/generator.py::_single_class_map` paints each line at full
  size unless `artificial_class_label` names a ring to carve out of it, so the lines in a label
  image touched their neighbours and the trained model answered with one `TextLine` the width of the
  page (the user's own `inference.xml`, 239 regions and no `TextLine` among them, was the same
  defect seen from the writer). The legacy pipeline's third class was exactly that ring -- `9` in
  `custom_config_page2label.json` -- and the file now carries it as `n_classes: 3` with
  `artificial_class_label: 2` beside `case_label: 1`, the layout table that shared those ids
  dropping out with it (`columns_width` among them, so the label generator no longer resizes a
  one-column page to that table's 1000 px). `Page2Label.region_class_ids()` now separates the ids a
  page is written from from the ids a label image may carry -- `legal_class_ids()` is that set plus
  the background and the ring -- so a predicted boundary is trained on and never written as a
  `TextRegion` of fragments beside the lines it rings, and `validate_pair`, which read the region
  table and so saw nothing at all in a cleared single-class config, reads the legal set instead:
  the check that would have flagged `n_classes: 2` sitting beside a class id of `2`.
- **The `sbb train` bar kept only the epoch it was on, and two of its numbers.** `SBBProgressBar` is
  a `RichProgressBar`, and it inherited that class's `leave=False`: rich paints one *live* panel and
  repaints it in place, so epoch 0's line was erased the moment epoch 1 started and a finished run
  left a single line on the terminal -- the last epoch, not the run. The subclass defaults to
  `leave=True` now, which the parent implements by printing the finished frame before rebuilding the
  live display, so every epoch's line survives with the numbers it ended on. The numbers were the
  other half: the bar renders exactly the metrics logged with `prog_bar=True`, and the module flagged
  `train_loss` and `val_loss` alone, so `train_loss_step` -- whose own comment says it exists to keep
  the bar live -- and `val_accuracy` were in the logs and in the checkpoint filenames while the bar
  showed neither. The bar now carries the pair a run is judged on plus the step loss
  (`training.module.SBBLightningModule.bar_keys`), and leaves `val_miou`, the two dice scores and the
  per-class `iou/2`/`recall/2`/`support/2` to the logs -- flagging *everything* instead turned a
  six-class run's line into a wall of `train_support/4: 1408.000` entries that pushed the losses off
  the screen. `early_stopping_fields` printed `early_stopping/val_loss: 0/10` *and*
  `early_stopping/val_accuracy: 0/10` -- the metric names twice more, values left out, beside the same
  numbers shown as values -- and emits one `early_stopping: 0/10`, from the callback closest to
  stopping.

### Removed

- The root modules `train.py`, `training.py`, `initialization.py`, `model_utils.py`,
  `models_pytorch.py`, `dataset.py`, `augmentation.py`, `utils.py`, `metrics.py`, `losses.py`,
  `config_loader.py`, `inference_pytorch.py`, `gt_gen_utils.py`, `generate_gt_for_training.py`,
  `build_model_load_pretrained_weights_and_save.py` (a Keras script in a PyTorch repository) and
  the root `__init__.py`, plus `requirements.txt`, `train.md` and the four root config JSONs. Each
  one is replaced in `src/sbb_pytorch/` before it is deleted; the deletion is the last commit of
  the refactor.
- The `sacred` dependency, whose only user was the Keras build script.

## [0.1.0] - 2025-10-23

The pre-refactor state of the repository: the port of
[qbayee/sbb_pixelwise_segmentation](https://github.com/qbayee/sbb_pixelwise_segmentation) from
TensorFlow 2.12.1 to PyTorch 2 with PyTorch Lightning, as a set of scripts in the repository root.

### Added

- `models_pytorch.py`: `ResNet50UNet`, `ResNet50UNetLight`, `ResNet50Classifier`,
  `MachineBasedReadingOrderModel`, `ViTResNet50UNet`, `ViTResNet50UNetTransformerBeforeCNN`,
  `EnhancementModel` -- torchvision ResNet50 encoders with hand-written decoders, five copies of
  the same decoder.
- `train.py` (Sacred-style config plus `argparse`, a `SegmentationModel` Lightning module used for
  every task), `training.py` (a hand-written trainer and its own `ModelCheckpoint`),
  `initialization.py` (three weight-init schemes, a gradient clipper, an L2 regularizer that
  returned a norm), `model_utils.py` (save/load/checkpoint in a fourth layout).
- `dataset.py` and `augmentation.py` (the dataset plus two `albudataloading`-style augmentation
  pipelines, one of which the dataset never used), `utils.py` (49 KB of image, patch, logging and
  checkpoint helpers, several functions duplicated with divergent behaviour), `metrics.py` and
  `losses.py` (five ways to compute accuracy/IoU/Dice and their losses).
- `config_loader.py` with eight dataclasses and enums, `config_params.json`,
  `config_train_textline.json`, `custom_config_page2label.json`, `scales_enhancement.json`,
  `requirements.txt`, `train.md`, `README.md`, `CHANGELOG.md`, `TECHNICAL_CHANGELOG.md`.
- `inference_pytorch.py` (`SBBPredict`), `gt_gen_utils.py` and `generate_gt_for_training.py`
  (ALTO/PageXML parsing, contour extraction and label rasterising, six of its functions
  near-identical).

[Unreleased]: https://github.com/cneud/sbb_pytorch/compare/main...refactor/packaged-rewrite
[0.2.0]: https://github.com/cneud/sbb_pytorch/tree/refactor/packaged-rewrite
[0.1.0]: https://github.com/cneud/sbb_pytorch/tree/main
