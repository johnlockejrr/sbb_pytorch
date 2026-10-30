# Technical changelog

Two refactors are recorded here. Version 0.1.0 was the port of the TensorFlow codebase to PyTorch,
and this file used to be its API notes (Keras layers to `nn.Module`, `load_weights` to
`load_state_dict`): that text is condensed in section 6, and the original is in git. Version 0.2.0
is the packaged rewrite -- sixteen loose scripts in the repository root turned into the
`sbb_pytorch` package -- and sections 1 to 5 record it.

[docs/migration.md](migration.md) is the authoritative semantic log: every change that moves a
number, marked **(breaking)**, **fix** or **api**. This file is the map you read before porting a
script or rerunning a job -- where each old file went, what each old command became, and which
numbers moved. Where the two disagree, `docs/migration.md` wins.

## 1. Legacy file to module map

| Legacy file (repository root) | Now | What it had been |
| --- | --- | --- |
| `train.py` (428 lines) | [training/factory.py](src/sbb_pytorch/training/factory.py), [training/module.py](src/sbb_pytorch/training/module.py), [training/callbacks.py](src/sbb_pytorch/training/callbacks.py), [training/trainer.py](src/sbb_pytorch/training/trainer.py), [cli.py](src/sbb_pytorch/cli.py) | One script holding an argparse front end, a `SegmentationModel` Lightning module used for all six tasks, three copies of `get_optimizer`, a `SaveWeightsAfterSteps` callback that never fired, and the `torch.save({...})` that read `callback_metrics["val_acc"]`. Split into one file each for builders, the module, the callbacks and the trainer, plus the `sbb train` command. |
| `training.py` | [training/trainer.py](src/sbb_pytorch/training/trainer.py), [training/callbacks.py](src/sbb_pytorch/training/callbacks.py) | A hand-written `Trainer` (`train_step`/`validate_step`/`train_epoch`) beside pytorch-lightning, and a `ModelCheckpoint` whose `save_model` read `self.model`/`self.optimizer` -- attributes its own `__init__` never assigned, so the first save raised `AttributeError`. |
| `initialization.py` | [training/factory.py](src/sbb_pytorch/training/factory.py) | Three weight-init schemes, `GradientClipper`, `WeightRegularizer`. `init_weights`/`clip_gradients`/`regularization_loss` reproduce the formulas exactly -- including the L2 one, which returns `weight_decay * \|\|p\|\|₂`, a norm rather than `0.5 * sum(p²)`. |
| `model_utils.py` | [storage/checkpoint.py](src/sbb_pytorch/storage/checkpoint.py) | `save_model`/`load_model`/`save_checkpoint`, each writing or reading a different checkpoint layout. One `Checkpoint` schema now, with `load_checkpoint` accepting every spelling in the wild. |
| `models_pytorch.py` | [models/base.py](src/sbb_pytorch/models/base.py), [models/unet.py](src/sbb_pytorch/models/unet.py), [models/vit.py](src/sbb_pytorch/models/vit.py), [models/heads.py](src/sbb_pytorch/models/heads.py), [models/registry.py](src/sbb_pytorch/models/registry.py) | Seven model classes assembled from five copies of the same decoder, four of which could not complete a forward pass. One `UNetDecoder`, four thin assemblies, a `MODEL_REGISTRY` of seven keys. |
| `dataset.py` | [data/dataset.py](src/sbb_pytorch/data/dataset.py), [data/datamodule.py](src/sbb_pytorch/data/datamodule.py), [data/patches.py](src/sbb_pytorch/data/patches.py), [data/augmentations.py](src/sbb_pytorch/data/augmentations.py) | The `Dataset` class, the `DataModule` shell that ran `preprocess_and_augment_data` in `__init__`, the patch generation, and a hard-coded `A.Compose` that ignored `augmentation: false`. Four modules now, each with one job. |
| `augmentation.py` | [data/augmentations.py](src/sbb_pytorch/data/augmentations.py) | The dead half of a two-pipeline split: this module read the config, `dataset.py`'s inline `A.Compose` was what actually ran. All three of its pipelines also raised on albumentations 2.0.8, so the transform *set* is preserved and the constructor calls could not be. |
| `utils.py` (49 KB) | [storage/images.py](src/sbb_pytorch/storage/images.py), [storage/checkpoint.py](src/sbb_pytorch/storage/checkpoint.py), [data/patches.py](src/sbb_pytorch/data/patches.py), [logging.py](src/sbb_pytorch/logging.py) | Image I/O (three `resize_image` implementations, eight helpers duplicated with divergent behaviour), the `provide_patches` family, and the logger that appended a new `FileHandler` on every call. Nothing was deleted without a home. |
| `metrics.py` | [metrics/functional.py](src/sbb_pytorch/metrics/functional.py), [metrics/segmentation.py](src/sbb_pytorch/metrics/segmentation.py), [metrics/enhance.py](src/sbb_pytorch/metrics/enhance.py) | Five ways to compute two numbers (`seg_metrics`, `IoU`, `DiceScore`, `class_tversky`, `generalized_dice_coeff`). One `Metric` interface; the semantics are listed in `metrics/__init__.py` under *SEMANTIC CHANGES*. |
| `losses.py` | [losses.py](src/sbb_pytorch/losses.py) | Kept as the `nn.Module` layer, now a thin wrapper over `metrics.functional` instead of a second copy of the same maths with different constants. |
| `config_loader.py` | [config/models.py](src/sbb_pytorch/config/models.py), [config/loaders.py](src/sbb_pytorch/config/loaders.py) | Eight dataclasses plus a stateful `ConfigLoader` whose `get_training_config()` depended on `load_config()` having run first. Three pydantic models and five functions; the JSON schema they generate *is* the config documentation now. |
| `inference_pytorch.py` | [inference/predictor.py](src/sbb_pytorch/inference/predictor.py), [inference/visualize.py](src/sbb_pytorch/inference/visualize.py) | `SBBPredict`: a class mixing model loading, normalisation, rendering, file naming and an unused ground-truth comparison. The engine and the renderer are separate now, and `--save` no longer writes nothing for classification. |
| `gt_gen_utils.py` (89 KB) | [gt/parser.py](src/sbb_pytorch/gt/parser.py), [gt/geometry.py](src/sbb_pytorch/gt/geometry.py), [gt/render.py](src/sbb_pytorch/gt/render.py), [gt/generator.py](src/sbb_pytorch/gt/generator.py), [gt/reading_order.py](src/sbb_pytorch/gt/reading_order.py) | Five near-identical XML readers (~1000 duplicated lines) and five rasterising functions, each with its own colour table and blend maths. One parser, one geometry module, one renderer, plus the reading-order pair builder. |
| `generate_gt_for_training.py` | the `sbb gt` group in [cli.py](src/sbb_pytorch/cli.py) | Its own `click.group`, whose `--out` meant "directory of reading-order XML" in one command and "output directory" in another. See section 2 for what each command became. |
| `build_model_load_pretrained_weights_and_save.py` | deleted; the job is `sbb export` | A Keras/TensorFlow script (it imported `tensorflow.keras`) in a PyTorch repository, used to copy weights between architectures. |
| `__init__.py` (root, empty) | deleted | Zero bytes, and it made the source tree importable as a module by accident -- `import sbb_pytorch` worked from the root and failed everywhere else. |
| `requirements.txt` | [pyproject.toml](pyproject.toml) | `sacred` dropped (its only user was the Keras script), `torch`/`torchvision` declared but the active build still chosen by hand, floors raised where the code needs it, `pydantic`/`click`/`rich`/`tensorboard` added. |
| `config_params.json`, `config_train_textline.json`, `custom_config_page2label.json`, `scales_enhancement.json` | [configs/](configs/) + [docs/config-reference.json](config-reference.json) | Four files at the root with no indication of which paired with which. See section 4 for the value deltas. |
| `README.md`, `CHANGELOG.md`, `TECHNICAL_CHANGELOG.md`, `train.md` | rewritten ([README.md](README.md), [CHANGELOG.md](CHANGELOG.md), this file) | `train.md` (27 KB of legacy run notes) is folded into the README and the changelogs rather than deleted blind. The README's license claim is fixed: `LICENSE` has always been Apache-2.0 while the README said MIT. |

## 2. Command-line surface, old to new

Three argument-parsing styles became one group, so the same idea no longer spells itself three ways
(`--batch-size` here, `--batch_size` there) and `--out` means one thing.

| Before | Now | Notes |
| --- | --- | --- |
| `python train.py --config F [--batch-size --num-workers --epochs --patience --device --resume --page2label --output-dir]` | `sbb train F [...]` | Same flags, and each one is a config field (`--batch-size` -> `n_batch`, `--epochs` -> `n_epochs`, `--resume` -> `dir_of_start_model`), resolved by `apply_overrides`. `--resume` used to be accepted and never passed to `fit(ckpt_path=...)`. New: `--model`, `--pretrained/--no-pretrained`, `--every-n-steps`, `--logger/--no-logger`, `--dry-run`. |
| `python inference_pytorch.py -i IMG -s PRED.png -m DIR [--patches --save-layout --ground-truth --xml-file --min-area]` | `sbb infer MODEL [INPUT...] [-i --save --save-layout --patches -c --architecture -d --limit]` | `--model` meant the weights path in one script and the architecture in another; now the path is the argument and the architecture is `--architecture`. `--patches` was a `pass` statement and is a real sliding window. `--ground-truth` was documented as a directory, read as a file, and its IoU computed and discarded. |
| `python generate_gt_for_training.py ...` (six click commands) | `sbb gt generate`, `sbb gt visualize` | `pagexml2label` -> `gt generate`, the `visualize_*` commands -> `gt visualize`. `image_enhancement` and `machine_based_reading_order` have no `sbb gt` equivalent yet -- `docs/migration.md` lists `gt enhance` and `gt reading-order` as their replacements; the modules exist ([gt/generator.py](src/sbb_pytorch/gt/generator.py), [gt/reading_order.py](src/sbb_pytorch/gt/reading_order.py)) but the subcommands are not in [cli.py](src/sbb_pytorch/cli.py) yet. |
| `python build_model_load_pretrained_weights_and_save.py` (edit constants, run) | `sbb export CHECKPOINT [-o OUT]` | |
| reading `config_params.json` and its hand-written table | `sbb validate CONFIG [--strict]`, `sbb config-schema [-o OUT]` | |

## 3. Changes that move a number

Summarised by area; [docs/migration.md](migration.md) section 3 carries the full list with
markers, and section 4 is the symbol-by-symbol import table.

- **Metrics and losses** (migration.md 3.1). Tversky's `alpha`/`beta` were swapped; focal loss used
  `alpha=4.0` in `metrics.py` against `0.25` in `losses.py` and diluted the loss by `1/n_classes`;
  focal-Tversky used an unpublished formula; Jaccard loss was multiplied by its smoothing constant;
  generalized dice merged every image in the batch; `ignore_index` filtered the result vector
  instead of the labels, so classes that scored exactly 0 vanished from the mean and mIoU read
  high; `losses.SSIMLoss` raised `NameError` on construction (`np.exp` with numpy never imported),
  making the only enhancement loss unreachable. Smoothing constants that were `1.0`, `0.001`,
  `1e-8` and `100.0` in four places are `DEFAULT_SMOOTHING`/`DEFAULT_EPSILON`, overridable.
- **Models** (3.2). Four of the five copied decoders could not complete a forward pass (3072
  channels into a `Conv2d(2048, 1024)`; `ResNet50UNet` ran only because it special-cased `i == 0`,
  and its pairing was `[f5, f3, f2, f1, f1]` -- `f4` never used, `f1` twice). Both ViT variants
  raised `RuntimeError`. Every model now returns raw logits, where four previously returned
  `softmax` and `CrossEntropyLoss` normalised twice on top. `weight_decay` is gone from the model
  constructors (accepted, never used -- it is an optimizer setting). `TaskType.BINARIZATION` had no
  enum member and no `create_model` branch.
- **Data, images, augmentation** (3.3). Training loaded images as RGB through PIL and inference as
  BGR through cv2 while applying RGB statistics, so the model saw two different tensors for one
  file. `read_mask` had collapsed 3-channel SBB label PNGs by luminance, turning class ids
  `{1,2,3}` into `{10,90,187}`. `im.split('.')[0]` truncated `page.001.png` to `page`, so every
  variant resolved to one label. `augmentation: false` was ignored. `preprocess_and_augment_data`
  ran in `DataModule.__init__`, so constructing the datamodule copied the dataset and two
  processes wrote the same files. Patch generation used global RNG state and is seeded now
  (`PatchConfig.seed`) -- the one change that makes a rerun of an old config produce different data.
- **Configuration** (3.4). Any key a file omitted raised `KeyError`, even when the value was
  irrelevant, and `config_train_textline.json` omitted ten of them. Nothing was validated: the
  three hand-written checks in `validate_configs` become `SBBConfig.warnings()` +
  `validate_pair()`, which report instead of raising (the shipped pair declares `n_classes=2` while
  its mapping uses ids up to 6). `" "` for `dir_of_start_model` becomes `None`; the
  `"blur_k": ["blur", "guass", "median"]` typo is corrected rather than crashing.
- **Checkpoints and storage** (3.5). Three checkpoint layouts were in simultaneous use and each
  module could read only one, so a model trained by `train.py` could not be loaded by
  `model_utils.load_model` and the reverse. Pruning sorted lexicographically --
  `epoch_10` before `epoch_2` -- and deleted the *newest* files. `cv2.imwrite` was called on
  `int32`/float arrays in some paths, producing 32-bit PNGs; everything now goes through
  `storage.images.imwrite`. Step checkpoints move from `model_step_1200/model.pt` to
  `steps/model_step_1200.pt`.
- **Training loop** (3.6). `config.device` was read by nothing and `pl.Trainer(accelerator=device,
  devices=1)` rejected the valid spec `cuda:1`; `config.precision`,
  `config.accumulate_grad_batches` and `config.gradient_clip_val` were ignored;
  `ReduceLROnPlateau` was imported and never constructed, so no scheduler ran; `Adadelta` was built
  without `weight_decay` and `momentum` was passed to optimizers that do not accept it;
  `create_model(config)` downloaded ImageNet weights and `initialize_weights(model)` then re-drew
  every one of them. The final save read `callback_metrics["val_acc"]` and raised `KeyError` after a
  whole run on any task that does not log it.
- **Metric names** (3.6). `train_acc`/`train_iou`/`train_dice` and their `val_` twins came from
  three objects, each recomputing the same counts on every batch, so the logged epoch value was a
  running average of intermediate states and depended on batch size and order. One confusion matrix
  now supplies `val_loss`, `val_accuracy`, `val_miou`, `val_dice_macro`, `val_dice_micro`;
  **`val_acc` -> `val_accuracy`**, and anything reading old TensorBoard or CSV runs by the old key
  needs the new one.
- **Inference and rendering** (3.7). The saved overlay was a thumbnail: `resize_image(img,
  layout.shape[0], layout.shape[1])` downsampled a 3000x4000 page to the model's 448x672 output
  instead of upscaling the mask. Bilinearly resizing the label map invented class ids.
  Normalisation happened before resizing, and the classification path skipped it entirely.
  `color_images` hard-coded `{0: white, 1: red, 2: blue}` -- every class from 3 up rendered black,
  and the two colours were swapped relative to their own comments because one site treated the
  array as BGR and the rest as RGB.
- **Ground-truth generation** (3.8). The XML namespace prefix was derived as `"PcGts}"` for a
  namespace-less document, leaving `imageHeight`/`imageWidth` unbound; `config_params=None` crashed
  the layout branch although `--config` was documented as ignorable; `case_label` was applied to
  every region whatever the use case; the overlay blend used `cv2.addWeighted(img, 0.5, layout,
  0.1, 0)` -- weights summing to 0.6, dimming the result. Areas are measured with
  `cv2.contourArea` throughout, where the legacy code mixed it with `shapely` and the two halves of
  one filter disagreed. The `3d` palette is the legacy table verbatim -- despite its name it was
  handed to OpenCV, so keeping the order keeps the colours identical.
- **Packaging and license** (3.9). `requirements.txt` -> `pyproject.toml`, `sacred` dropped, the
  package installable, and the README's MIT claim corrected to the Apache-2.0 that `LICENSE` has
  always contained.

## 4. Config file deltas

The six files in [configs/](configs/) versus the four at the root; the same list in
[docs/.configs-changes.txt](.configs-changes.txt), which exists to be folded in and deleted.

| File | Source | Deltas |
| --- | --- | --- |
| `configs/textline.json` | `config_train_textline.json` | Adds the ten keys the legacy file omitted but `ConfigLoader` rebuilt by hand (`num_workers`, `patience`, `device`, `optimizer`, `momentum`, `shuffle_indexes`, `number_of_backgrounds_per_image`, `dir_rgb_backgrounds`, `dir_rgb_foregrounds`, `dir_img_bin`) -- their absence was the `KeyError` that made the file loadable through only one code path. `device` becomes `auto`. `"guass"` -> `"gauss"`. `page2label` inlined from `custom_config_page2label.json`. `pretraining` and `n_classes` unchanged. |
| `configs/layout.json` | `config_params.json` | `transformer_num_patches_xy: [56, 56]` -> `num_patches_xy: [112, 112]`: with patchsize 4 the legacy grid covers 224x224 behind a 448x448 input, and `448 // 4 == 112` (the `[128, 128]` in some copies covers 512x512 and is wrong for 448 too). The eight flat `transformer_*` keys become the nested `transformer` block (the flat spelling still loads). `device: "cuda"` -> `"auto"`. `page2label.use_case`: `"textline"` -> `"layout"`, the region-to-id mapping unchanged. `n_epochs` stays `0`, the legacy value: `sbb train` builds a trainer with `max_epochs=0`, a validation-only run -- kept verbatim rather than silently restyled, set it before training a real model. |
| `configs/binarization.json` | new | No legacy file existed (`TaskType` had no `binarization` member). 672x448, `n_classes: 2`, 50 epochs, batch 4 -- the geometry of the other ResNet50UNet config, since `task: binarization` builds `resnet50_unet`. |
| `configs/classification.json`, `configs/reading_order.json` | new | 224x224, the input size `ResNet50Classifier` and `MachineBasedReadingOrderModel` default to; 30 epochs, batch 16 and 8. |
| `configs/enhancement.json` | new | 448x448, `n_classes: 3` (the schema requires 3 for `task: enhancement`), with `scales_enhancement.json` inlined as `enhancement_scales` -- the values verbatim, the root file untouched, and inline wins over `--scales`. |

Common to the four new files: `pretraining: false` (the rule is that `true` appears only where a
legacy config said so, because `true` means a torchvision download at build time and that fails
offline), `augmentation: false`, `data_is_provided: true`, and placeholder data directories
(`data/<task>/{train,eval}`, `runs/<task>`) to be pointed at real data.

## 5. Import table

For a script being ported: `docs/migration.md` section 4 maps every legacy symbol to its new
location. The renames worth memorising are `ConfigLoader.load_config` -> `config.load_config`,
`model_utils.load_model`/`save_model` -> `storage.load_checkpoint`/`save_checkpoint`,
`train.get_optimizer`/`initialization.initialize_weights` -> `training.factory.build_optimizer` /
`init_weights`, `models_pytorch.ResNet50UNet` -> `models.ResNet50UNet` (via `create_model`), and
`utils.<image helper>` -> `storage.images.<helper>`.

## 6. The 0.1.0 changelog, condensed

Before the package existed, this file documented the TensorFlow 2.12.1 -> PyTorch port: Keras
`Model`/`layers`/`regularizers`/`optimizers` imports replaced by `torch`/`torch.nn`/
`torch.optim`/`pytorch_lightning`; the hand-written Keras ResNet50 (`conv_block`/`identity_block`
functions and a functional-API U-Net decoder) replaced by `torchvision.models.resnet50` plus
`nn.Module` decoders; `model.load_weights()` -> `model.load_state_dict()`; Sacred's
`@ex.automain` config and `model.fit_generator()` loops replaced by a `pl.Trainer` with
`ModelCheckpoint`/`EarlyStopping`/`TensorBoardLogger`; the `data_gen` while-loop generator replaced
by `Dataset`/`DataLoader` with albumentations; the Keras `soft_dice_loss` and
`weighted_categorical_crossentropy` replaced by `nn.Module` losses; `.h5` files replaced by `.pt`
checkpoints; and TF's `ConfigProto` GPU-memory setup replaced by torch's own allocator. The full
0.1.0 text, including its per-file and breaking-change tables, is in git at `5ad9cc7`.
