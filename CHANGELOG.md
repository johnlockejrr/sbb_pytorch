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
  `generate_gt_for_training.py`): `train`, `infer`, `gt generate`, `gt visualize`, `export`,
  `validate`, `config-schema`.
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
  architectures, replacing seven classes built from five copied decoders. `pairing="legacy"`
  reproduces the old off-by-one skip pairing so old checkpoints still load.
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
  losses, storage, inference, training (plus the training factory) and ground truth.
  `tests/test_gt.py` is the last to land and the largest -- 2183 lines, 111 test functions, 241
  collected cases over `parser`, `geometry`, `render`, `reading_order` and `generator`, none of
  which had ever been tested outside the `--dry-run` CLI paths of `tests/test_cli.py`. Its expected
  values are derived from the geometry or from the legacy contract each docstring states rather
  than read back off the implementation. Eleven of its assertions disagreed with the code they
  covered; each was first pinned with a `# TODO(file.py:line)` marker and is now an ordinary
  assertion, and the eleven fixes are listed in
  [docs/migration.md](docs/migration.md) section 3.8 and under **Fixed** below.
- `configs/`: six ready-made configs (textline, layout, binarization, enhancement,
  classification, reading_order). The first two carry the values of `config_train_textline.json`
  and `config_params.json`; the other four tasks had no legacy config at all.

### Changed

- Every legacy script is replaced by a module: see [TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md)
  for the file-by-file table, including the four root JSON configs (now `configs/*.json`),
  `requirements.txt` (now `pyproject.toml`), and `train.md` (folded into the README and the
  changelogs).
- `--device` defaults to `auto` and is validated at parse time. `train.py` hard-coded `cuda` and
  called `pl.Trainer(accelerator=device, devices=1)`, which rejected the valid spec `cuda:1` and
  made the second GPU of a machine unreachable.
- Config flags and config fields are the same values. `train.py`'s `get_arg_with_priority` helper
  became `config.loaders.apply_overrides` (flag, then file, then default), and `--resume` --
  accepted and never forwarded to `fit(ckpt_path=...)` -- is a real field.
- Checkpoint file names: `model-{epoch:02d}-{val_loss:.4f}.ckpt` and
  `best-{epoch:02d}-{val_accuracy:.4f}.ckpt`, `top_k=3` each; mid-epoch checkpoints go in a
  `steps/` subdirectory instead of one directory per step.
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
  segmentation model; `pairing="legacy"` keeps the bare conv its checkpoints hold (`head.weight`
  becomes `head.classify.weight`), and `LIGHT_DECODER_CHANNELS` takes the widths the light decoder
  really has now that its 32-channel conv has moved back into the head.
- **Training.** `torch.save({... "val_acc": trainer.callback_metrics["val_acc"]})` raised `KeyError`
  *after* a full run for any task that does not log it, throwing away the result.
  `config.precision`, `config.accumulate_grad_batches` and `config.gradient_clip_val` were read by
  nothing. `ReduceLROnPlateau` was imported and never constructed, so no scheduler ever ran;
  `torch.optim.Adadelta` was built without `weight_decay`; `momentum` was passed to optimizers that
  do not take it; pretrained ImageNet weights were re-initialised with kaiming one line after being
  downloaded. `device.available_devices` called `torch.device_count()` -- an attribute that does not
  exist, on the one branch that only runs when CUDA is present, so it passed every test here and
  would have raised `AttributeError` on the machine `--device auto` is meant for.
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
