# Migrating from the legacy repository to the `sbb_pytorch` package

The refactor on the `refactor/packaged-rewrite` branch (from `5ad9cc7`) turns sixteen loose
scripts in the repository root into the installed [`sbb_pytorch`](../src/sbb_pytorch/__init__.py)
package described in [README.md](../README.md). The rule the whole refactor was written under was
**behaviour preservation outranks elegance**: a port that changes a published number is a bug fix
or a regression, and the difference has to be written down before it can be accepted.

This document is that written-down difference. It is the sign-off sheet: read section 3 before you
rerun a training job, and section 4 before you port a script that imported one of the old modules.

## 1. How to read this

Every entry carries one of three markers:

| Marker | Meaning | What you do about it |
| --- | --- | --- |
| **(breaking)** | A reported number, a saved file, or a call signature changes value. | Re-baseline the run. There is no flag that gets the old number back. |
| **fix** | The old behaviour was a defect -- a crash, a silently wrong tensor, a value computed and thrown away. | Nothing; the new value is the correct one. |
| **api** | Names and signatures moved, no value moved. | Section 4 maps every old symbol to its new home. |

Where a legacy value is reproduced *deliberately* even though it is odd -- the redundant
normalisation in `class_weights`, the `micro`-only Dice reduction, the BGR colour table -- that is
said out loud and repeated in section 5. A refactor that silently "improves" a formula is the
failure mode this document exists to prevent, so the list is exhaustive rather than short.

## 2. Where every old file went

| Legacy file (repository root) | New location | Notes |
| --- | --- | --- |
| `train.py` | [training/factory.py](../src/sbb_pytorch/training/factory.py), [training/module.py](../src/sbb_pytorch/training/module.py), [training/callbacks.py](../src/sbb_pytorch/training/callbacks.py), [training/trainer.py](../src/sbb_pytorch/training/trainer.py) | One file each for builders, the Lightning module, checkpointing, and the `pl.Trainer` -- plus the `sbb train` command in [cli.py](../src/sbb_pytorch/cli.py). |
| `training.py` | [training/trainer.py](../src/sbb_pytorch/training/trainer.py), [training/callbacks.py](../src/sbb_pytorch/training/callbacks.py) | The hand-written `Trainer` class (`train_step`/`validate_step`/`train_epoch`) and its `ModelCheckpoint`/`Metric` wiring. |
| `initialization.py` | [training/factory.py](../src/sbb_pytorch/training/factory.py) | `initialize_weights`, the three init schemes, `GradientClipper`, `WeightRegularizer`. |
| `model_utils.py` | [storage/checkpoint.py](../src/sbb_pytorch/storage/checkpoint.py) | `save_model`/`load_model`/`save_checkpoint` collapse into one `Checkpoint` schema. |
| `models_pytorch.py` | [models/base.py](../src/sbb_pytorch/models/base.py), [models/unet.py](../src/sbb_pytorch/models/unet.py), [models/vit.py](../src/sbb_pytorch/models/vit.py), [models/heads.py](../src/sbb_pytorch/models/heads.py), [models/registry.py](../src/sbb_pytorch/models/registry.py) | Seven classes and five decoder copies become one `UNetDecoder` plus four thin assemblies and a registry. |
| `dataset.py` | [data/dataset.py](../src/sbb_pytorch/data/dataset.py), [data/datamodule.py](../src/sbb_pytorch/data/datamodule.py), [data/patches.py](../src/sbb_pytorch/data/patches.py), [data/augmentations.py](../src/sbb_pytorch/data/augmentations.py) | The class, the `DataModule` shell, `preprocess_and_augment_data` and the inline `A.Compose` are four modules now. |
| `augmentation.py` | [data/augmentations.py](../src/sbb_pytorch/data/augmentations.py) | The dead half of a two-pipeline split; see section 3.3. |
| `utils.py` (49 KB) | [storage/images.py](../src/sbb_pytorch/storage/images.py), [storage/checkpoint.py](../src/sbb_pytorch/storage/checkpoint.py), [data/patches.py](../src/sbb_pytorch/data/patches.py), [logging.py](../src/sbb_pytorch/logging.py) | Pixel helpers, image I/O, the `provide_patches` family and the logger. Nothing was deleted without a home. |
| `metrics.py` | [metrics/functional.py](../src/sbb_pytorch/metrics/functional.py), [metrics/segmentation.py](../src/sbb_pytorch/metrics/segmentation.py), [metrics/enhance.py](../src/sbb_pytorch/metrics/enhance.py) | Five ways to compute two numbers become one `Metric` interface. |
| `losses.py` | [losses.py](../src/sbb_pytorch/losses.py) | Kept as the `nn.Module` layer, now a thin wrapper over `metrics.functional` instead of a second copy of the maths. |
| `config_loader.py` | [config/models.py](../src/sbb_pytorch/config/models.py), [config/loaders.py](../src/sbb_pytorch/config/loaders.py) | Eight dataclasses → three pydantic models; the stateful `ConfigLoader` class → five functions. |
| `inference_pytorch.py` | [inference/predictor.py](../src/sbb_pytorch/inference/predictor.py), [inference/visualize.py](../src/sbb_pytorch/inference/visualize.py) | `SBBPredict` splits into the engine and the renderer. |
| `gt_gen_utils.py` (89 KB) | [gt/parser.py](../src/sbb_pytorch/gt/parser.py), [gt/geometry.py](../src/sbb_pytorch/gt/geometry.py), [gt/render.py](../src/sbb_pytorch/gt/render.py), [gt/generator.py](../src/sbb_pytorch/gt/generator.py), [gt/reading_order.py](../src/sbb_pytorch/gt/reading_order.py) | ~1000 duplicated lines of XML reading and `fillPoly` writing collapse to one implementation each. |
| `generate_gt_for_training.py` | its six `click` commands → [cli.py](../src/sbb_pytorch/cli.py): `pagexml2label` → `sbb gt generate`, `visualize_reading_order` / `visualize_textline_segmentation` / `visualize_layout_segmentation` → `sbb gt visualize` | `image_enhancement` and `machine_based_reading_order` have no command yet -- their code is ported ([data/patches.py](../src/sbb_pytorch/data/patches.py), [gt/reading_order.py](../src/sbb_pytorch/gt/reading_order.py)); see section 5.6. |
| `build_model_load_pretrained_weights_and_save.py` | `sbb export CHECKPOINT` in [cli.py](../src/sbb_pytorch/cli.py), over [storage/checkpoint.py](../src/sbb_pytorch/storage/checkpoint.py) | A Keras-era script (it still imported TensorFlow) whose job was "build a model, load weights, save them under a key the next script can guess" -- `sbb export` normalises any legacy `.pth`/`.ckpt` into `model.pt` plus the `*_config.json` sidecar. |
| `__init__.py` (root, empty) | deleted | The root `__init__.py` was 0 bytes and made the source tree an importable module by accident. |
| `requirements.txt` | [pyproject.toml](../pyproject.toml) | See section 3.9 for what was dropped, added and version-bumped. |
| `config_params.json`, `config_train_textline.json`, `custom_config_page2label.json`, `scales_enhancement.json` | `configs/*.json` + [docs/config-reference.json](config-reference.json) | The example configs move under `configs/`; the reference is generated from the schema, not hand-written. |
| `README.md`, `CHANGELOG.md`, `TECHNICAL_CHANGELOG.md`, `train.md` | rewritten ([README.md](../README.md), `CHANGELOG.md`, `TECHNICAL_CHANGELOG.md`) | `train.md` (27 KB of legacy run notes) is folded into the README and the changelogs, not deleted blind. |
| `LICENSE` | kept | The file has always been Apache-2.0; the README claimed MIT. See section 3.9. |

Nothing on this list is deleted before its replacement exists and its tests pass; the root cleanup
happens as the last commit of the refactor.

## 3. Changes that move a number

### 3.1 Metrics and losses

The full list lives in
[metrics/__init__.py](../src/sbb_pytorch/metrics/__init__.py) under *SEMANTIC CHANGES*; it is
summarised here with the loss-side half of
[losses.py](../src/sbb_pytorch/losses.py) merged in, because the two files used to disagree with
each other and the disagreement is the point.

| # | Change | Marker |
| --- | --- | --- |
| 1 | **Tversky `alpha`/`beta` were swapped.** `metrics.class_tversky` computed `TP / (TP + alpha*FN + beta*FP)`; the definition is `TP / (TP + alpha*FP + beta*FN)`. The shipped `alpha=0.2, beta=0.8` pairing penalised false positives lightly and false negatives heavily -- the opposite of what it asked for. | **(breaking)** |
| 2 | **Focal-Tversky used the wrong formula.** Old `sum((1 - tversky) ** 1.3)`; new `mean((1 + DT) ** 3)` with `DT = (FP + TP) / (TP + alpha*FP + beta*FN)` -- the published loss. Loss values shift by a large factor, so a focal-Tversky run must be re-baselined. | **(breaking)** |
| 3 | **Focal-loss `alpha` is now `0.25` everywhere.** `metrics.focal_loss` defaulted to `4.0`, `losses.FocalLoss` to `0.25`; `4.0` is outside `[0, 1]` and scales the loss 16x relative to Lin et al. One constant now: `metrics.functional.DEFAULT_FOCAL_ALPHA`. | **(breaking)** |
| 4 | **`metrics.focal_loss` no longer dilutes the loss by `1 / n_classes`** -- it averaged the per-class loss vector over all `C` classes, so `C-1` zeros were folded into the mean. The true-class term is gathered now, which is the definition. | **(breaking)** |
| 5 | **Focal loss takes logits, not probabilities.** The old function clamped a probability tensor that `losses.FocalLoss` simultaneously fed to `binary_cross_entropy_with_logits` (a sigmoid on top of a softmax). | **fix** |
| 6 | **Jaccard loss is no longer multiplied by its smoothing constant** -- `(1 - jac) * 100` becomes `1 - jac` in `[0, 1]`. | **(breaking)** |
| 7 | **Generalized Dice is computed per image, not per batch.** `generalized_dice_coeff` summarised `axes = range(1, ndim - 1)`, which on `(B, C, H, W)` is `(C, H)`: it folded the class axis into the spatial one and merged every image into a single number. The `1 / count**2` weighting itself is unchanged. | **fix** |
| 8 | **`ignore_index` means exactly one thing:** a target label whose pixels are excluded. `metrics.IoU`/`DiceScore` also filtered the *result vector* with `scores != ignore_index`, comparing a ratio against a class id, so every class scored exactly `0` vanished from the mean and mIoU read high. | **(breaking)** |
| 9 | **Metrics pool counts instead of averaging per-batch ratios.** `MetricTracker` averaged batch means, so one image weighed as much as sixty-four, and `seg_metrics` averaged only over images where a class appeared. Epoch curves shift slightly; they are now independent of batch size and order. | **(breaking)** |
| 10 | **Smoothing constants consolidated:** `1.0` (`losses.DiceLoss`), `0.001` (`metrics.seg_metrics`), `1e-8` (`metrics.IoU`) and `100.0` (`jaccard_distance_loss`) become `DEFAULT_SMOOTHING = 1.0` and `DEFAULT_EPSILON = 1e-8`, both overridable. `DiceLoss(reduction="micro")` reproduces the old value exactly. | **api** |
| 11 | **`losses.SSIMLoss` used to raise `NameError` on construction** (`np.exp` with numpy never imported), which made `get_loss_function("enhancement")` -- the only enhancement loss -- unreachable. | **fix** |
| 12 | **SSIM uses a real Gaussian, reflect padding, and `C1`/`C2` derived from `data_range`.** The old exponent was `-x**2 / sigma**2` (no factor 2, so `sigma=1.5` behaved like `1.06`); `F.conv2d(padding=k//2)` averaged black into every border window; `C1 = 0.01 ** 2` assumed `[0, 1]` images while the pipeline decodes `0..255`. Now matches `skimage.metrics.structural_similarity(use_sample_covariance=False)`. | **fix** |
| 13 | **`weighted_categorical_crossentropy` / `image_categorical_cross_entropy` fold into** `metrics.functional.cross_entropy_loss`. They fed probabilities into `binary_cross_entropy_with_logits` and built the weight mask with `torch.maximum(..., ones)`, which floored every class weight at `1.0` -- a weight below 1 could not be expressed. | **fix** |
| 14 | **Prediction layout is always stated, never guessed.** `class_tversky` and `generalized_dice_coeff` did `permute(3, 1, 2, 0)` -- a channels-last (TensorFlow) operation on channels-first (PyTorch) tensors -- while `metrics.IoU` used `argmax(dim=1)`. Every function now takes ``layout="channels_first"`` or ``layout="channels_last"`` and validates the channel count against `n_classes`. | **fix** |
| 15 | **`binarization` and `reading_order` are handled by `get_metrics`** (the old function raised `ValueError` for anything but `segmentation`/`classification`/`enhancement`). Also fixed during the merge: `str()` of an `Enum` member is `"TaskType.SEGMENTATION"`, not `"segmentation"`, so `get_metrics`/`get_loss_function` rejected every `SBBConfig.task` their own tests passed as strings. | **fix** |
| 16 | **`DiceLoss` only ever computed the *micro* Dice.** `reduction="macro"` and `"none"` are now available and `"micro"` stays the default, so existing runs are unchanged. | **api** |
| 17 | **`SoftDiceLoss` did not exist in `losses.py`** (it was the `is_loss_soft_dice` branch inside `training.py`), and `TverskyLoss`/`FocalTverskyLoss`/`GeneralizedDiceLoss`/`JaccardLoss` existed only as plain functions in `metrics.py`. All are constructible `nn.Module`s now. | **api** |
| 18 | **`Adadelta` now receives `weight_decay`.** `torch.optim.Adadelta` accepts it (default `0.0`), and the three legacy optimizer builders dropped it -- which is why the hard-coded `Adam` never looked wrong. | **fix** |

### 3.2 Models

| Change | Marker |
| --- | --- |
| **Four of the five legacy decoders could not complete a forward pass.** The copied decoder fed 3072 channels into a `Conv2d(2048, 1024)` (`RuntimeError` on the first batch). `ResNet50UNet` ran only because it special-cased `if i == 0`, and its pairing was still off by one: `[f5, f3, f2, f1, f1]` -- `f4` never used as a skip, `f1` used twice. [models/base.py](../src/sbb_pytorch/models/base.py) is the single `UNetDecoder`, pairing each block with the encoder feature map of the same level and sizing every fusion with `F.interpolate` instead of assuming a clean pyramid (`f1` and `f2` are both at `H/2` in a ResNet, so `Upsample(scale_factor=2)` is wrong for one of them). | **fix** |
| **`pairing="legacy"` reproduces the off-by-one pairing** so checkpoints trained against the old `ResNet50UNet` still load; `create_model` and `sbb infer --pairing` expose it. | **api** |
| **Every model now returns raw logits.** `ResNet50UNet` returned logits for segmentation and `sigmoid` otherwise, while `ResNet50UNetLight` and both ViT variants returned `softmax`; `CrossEntropyLoss` on softmaxed logits normalises twice, so those four trained on a flattened loss surface. `models.base.probs` gives probabilities where they are wanted. | **(breaking)** |
| **`weight_decay` is gone from every model constructor** -- it was accepted and never used; it is an optimizer setting and lives in [training/factory.py](../src/sbb_pytorch/training/factory.py). | **api** |
| **Both ViT variants raised `RuntimeError` on their first forward pass.** `ViTResNet50UNet` fed a 32768-dim patch into a `nn.Linear(projection_dim, projection_dim)` (= 64 in, 64 out) and assumed `f5` was evenly divisible by the patch size (true only for inputs that are a multiple of 64, so a 448 page raised from `unfold`). `ViTResNet50UNetTransformerBeforeCNN` reshaped its tokens into a 48-channel tensor and handed it to a ResNet stage expecting 3; the `view(B, H//py, W//px, -1).permute(0, 3, 1, 2)` treated the token axis as the row axis. [models/vit.py](../src/sbb_pytorch/models/vit.py) projects from the real patch dimension and uses a tested `unpatch` inverse. | **fix** |
| **The position-embedding size is derived from the geometry and a supplied value is checked**, instead of building `nn.Embedding(num_patches, ...)` from a config number that disagreed with the tokens and broadcast-raised at forward time. | **fix** |
| **`TaskType.BINARIZATION` had no branch in `create_model`** -- worse, the enum had no such member, so a binarization config failed at *parse* time. It routes to the plain U-Net now. | **fix** |

### 3.3 Data, images and augmentation

| Change | Marker |
| --- | --- |
| **Training and inference fed the model different tensors for the same file.** `dataset.py` loaded photos through `PIL.Image.convert('RGB')`; `inference_pytorch.py` used `cv2.imread` (BGR) and then applied ImageNet's *RGB* statistics. [storage/images.py](../src/sbb_pytorch/storage/images.py) is the single reader: RGB in memory, BGR on write, with `channel_order=` stated at the call. | **fix** |
| **Label loading collapsed 3-channel SBB label PNGs by luminance**: `Image.open(mask).convert('L')` turned class ids `{1, 2, 3}` into `{10, 90, 187}`, so the ids never matched the page-to-label mapping. `read_mask` reads the channel that holds the ids; `remap_mask` turns stored values into class ids before the tensor is built. | **fix** |
| **`img_name = im.split('.')[0]` truncated `page.001.png` to `page`**, so every `page.00N` variant of one document resolved to a single label and overwrote its siblings. `data.dataset.label_path_for` (Path.stem based) is used everywhere, and [data/patches.py](../src/sbb_pytorch/data/patches.py) keeps both historical naming schemes (`img_{n}.png` and `{name}_flip_{code}.png`) because run directories on disk already use them. | **fix** |
| **The two `do_degrading` copies disagreed** -- `utils.py` downscaled and upscaled, `dataset.py` added noise and a blur. `utils.py`'s is the one `provide_patches` actually called, so `storage.images.downscale_upscale` stands in for it and `add_noise` covers the other. | **api** |
| **`return_binary_image_with_given_rgb_background_and_given_foreground_rgb` indexed an `(H, W, 3)` array with `rgb_foreground[0]`** -- one *row* -- and assigned it to a boolean-masked slice, which only works when the mask selects exactly one pixel. | **fix** |
| **Padding built `float64`/`int16` canvases and a bare `except: pass` around the brightening variants swallowed every error**, including "file unreadable". Padding returns `uint8` and failures are logged with the path; a single `A.Resize` replaces the `F.interpolate`-after-`ToTensorV2` fix-ups (a `TypeError` path whenever the transform output was 2-D) and the double `mask.squeeze()`. | **fix** |
| **`random.choice` for background/foreground picks ran on global RNG state**, so a rerun of the same config produced different data. `PatchConfig.seed` makes it reproducible. | **(breaking)** |
| **`augmentation=false` was ignored.** `dataset.DataModule` built a hard-coded `A.Compose` that applied flips and rotations whatever the config said, and `augmentation.py` -- the module that read the config -- was dead code. One factory now, every transform gated on the config field that asks for it. | **fix** |
| **All three of `augmentation.py`'s pipelines raise on albumentations 2.0.8** (`A.Flip` removed, `A.RandomResizedCrop(height=...)` now `size=`, `A.set_seed` gone in favour of `A.Compose(seed=...)`, `GaussNoise(var_limit=)` deprecated in favour of `std_range`). The *set of transforms and their probabilities* is preserved; the constructor calls could not be, because the pipeline they built had never once run in this environment. | **fix** |
| **`preprocess_and_augment_data` ran in `DataModule.__init__`**, so constructing the datamodule copied the dataset into `dir_output` and two distributed processes wrote the same files at once. It is `SBBDataModule.prepare_data` now, which Lightning calls once. | **fix** |
| **A missing image folder raised `FileNotFoundError` from a random worker mid-epoch** instead of at construction, where pairing can report it. Pairing happens in `__init__`; an image whose label is missing is a `ConfigError`, not a worker crash. | **fix** |

### 3.4 Configuration

| Change | Marker |
| --- | --- |
| **Any key a file omitted raised `KeyError`, even when the value was irrelevant.** `config_train_textline.json` omits `adding_rgb_background`, `shuffle_indexes`, `number_of_backgrounds_per_image`, `dir_rgb_backgrounds` and the optimizer keys, so `get_directory_config()` and `get_augmentation_config()` both crashed on it -- the file was only loadable through the one code path that did not call them. Every field now has a type, a default, and where meaningful a constraint. | **fix** |
| **Nothing was validated.** Types, ranges and cross-field rules existed only where a hand-written `if` happened to exist (three of them, in `validate_configs`). | **fix** |
| **`" "` for `dir_of_start_model`** -- which the shipped config uses to mean "unset" -- becomes `None`. | **fix** |
| **The shipped `"blur_k": ["blur", "guass", "median"]` misspelling is corrected** instead of crashing, matching `storage.images.BlurKind`. | **fix** |
| **Advisory problems are reported, not raised.** `SBBConfig.warnings()` and `validate_pair()` keep loading a shipped-but-imperfect config and print the problems (`sbb train` prints them); `validate_configs` raised instead. | **api** |
| **`ConfigLoader`'s hidden state is gone.** `get_training_config()` silently depended on `load_config()` having run first (`AttributeError` on `self.config` otherwise); `page2label_config` and `scales_config` were mutable instance attributes. Five functions, arguments in, values out. | **api** |
| **The documentation cannot drift any more**: `SBBConfig.model_json_schema()` *is* the config documentation ([docs/config-reference.json](config-reference.json), written by `sbb config-schema`); the old hand-written option table had already drifted from the code. | **api** |
| **"Is this config valid?" is a command** (`sbb validate CONFIG [--page2label ...] [--scales ...] [--strict]`) instead of something you learn by starting a run. Schema failures always exit 1; the cross-config advisories (the shipped `config_params.json` declares `n_classes=2` while `custom_config_page2label.json` uses ids up to 6) are printed and only fail with `--strict`, because refusing to start would break runs that exist today. | **api** |
| **Flags are fields and fields are flags.** `train.py`'s `get_arg_with_priority` helper -- which decided whether the flag or the JSON file won -- is `config.loaders.apply_overrides`, and `--resume` (accepted, never forwarded to `fit(ckpt_path=...)`) is a real field. | **fix** |

### 3.5 Checkpoints and storage

| Change | Marker |
| --- | --- |
| **Three checkpoint layouts were in simultaneous use** and each module could read only one: `model_state` (`model_utils.save_model`), `model_state_dict` (`train.py`), and bare state dicts. A model trained with `train.py` could not be loaded by `model_utils.load_model` and vice-versa. `Checkpoint.load` accepts all three spellings plus `state_dict`/`model`; everything new is written in one format with a `<stem>_config.json` sidecar. | **fix** |
| **`training.py`'s `ModelCheckpoint.save_model` read `self.model` and `self.optimizer`, attributes its own `__init__` never assigned** -- the first save raised `AttributeError` and no file was ever written by that class. | **fix** |
| **Pruning sorted lexicographically**: `sorted(glob("..._epoch_*.pth"))` put `epoch_10` before `epoch_2`, so `keep_last_n` deleted the *newest* files. Ordering is numeric now (and pytorch-lightning's own `save_top_k` covers the epoch files). | **fix** |
| **`SaveWeightsAfterSteps` declared `on_train_batch_end(self, model, optimizer, epoch)`** while pytorch-lightning calls `(self, trainer, pl_module, outputs, batch, batch_idx)` -- its first argument was the trainer and its "epoch" was the loss dict, so it never ran. `StepCheckpoint` has the signature pl calls and writes a real `Checkpoint`. | **fix** |
| **Step checkpoints used to be written one directory per step** (`model_step_1200/model.pt`), which no reader could glob; they now go in a `steps/` subdirectory, so `latest_checkpoint(dir_output)` still returns an epoch checkpoint. | **api** |
| **`cv2.imwrite` was called on `int32`/float arrays** in some paths, producing 32-bit PNGs; every save goes through `storage.images.imwrite`, which converts to `uint8` and to the file's BGR order. | **fix** |
| **`resize_image` existed three times and eight more helpers existed twice with divergent behaviour** (`dataset.py` fixed a negative-slice bug in `scale_image_for_no_patch` and nobody deleted the broken original). One module per concern now. | **api** |

### 3.6 Training loop

| Change | Marker |
| --- | --- |
| **`config.device` was read by nothing** and validated by nothing; `pl.Trainer(accelerator=device, devices=1)` accepted only `auto`, `cpu`, `gpu`, `cuda`, `mps` and `tpu`, so a config `device` of `"cuda:1"` -- valid for `torch.device` -- raised `MisconfigurationException`, and the second GPU of a machine was unreachable however it was configured. `resolve_accelerator` splits the spec into `{accelerator, devices}`. | **fix** |
| **`torch.save({... "val_acc": trainer.callback_metrics["val_acc"]})` raised `KeyError` *after* a full training run** for any task that does not log `val_acc` (enhancement, reading order), throwing away the result. The final save goes through `Checkpoint`, whose `metrics` field tolerates whatever the run logged. | **fix** |
| **`config.precision`, `config.accumulate_grad_batches` and `config.gradient_clip_val` were ignored** by the legacy `Trainer(...)` call. They are passed now. | **fix** |
| **No scheduler ever ran:** `from torch.optim.lr_scheduler import ReduceLROnPlateau` at the top of `train.py` was imported and never constructed, and `configure_optimizers` returned an optimizer alone. `SBBLightningModule.configure_optimizers` returns optimizer *and* scheduler (`schedule=False` reproduces the old return value). | **fix** |
| **Pretrained weights were overwritten the moment they were downloaded**: `create_model(config)` fetched ImageNet weights and `initialize_weights(model)` then re-drew every one with kaiming. `init_method="auto"` -- the default -- applies kaiming only to a from-scratch model. | **fix** |
| **Three implementations of `get_optimizer`** (module-level, `configure_optimizers`, `initialization.py`), agreeing only on being incomplete -- `initialization.py`'s docstring records that `"optimizer": "adam"` in the config coexisted with a hard-coded `torch.optim.Adam`. One builder, all four `OptimizerType` members. | **api** |
| **`clip_gradients(model)` used to default `max_norm=1.0`**, so asking for value clipping meant writing `max_norm=None` by hand and forgetting it clipped by norm while reporting the method that was asked for. There is no default now: exactly one of `max_norm`/`clip_value` is required. | **api** |
| **`momentum` was passed to optimizers that do not take it** (the `TypeError` three of the duplicated if/elif chains raised). It is applied to SGD only. | **fix** |
| **`train_acc`/`train_iou`/`train_dice` and their `val_` twins came from three objects** each recomputing the same counts, and `self.log(...)` was called with a fresh `compute()` **on every batch**, so the logged value was a running average of every intermediate state. One confusion matrix supplies `val_accuracy`, `val_miou`, `val_dice_macro`, `val_dice_micro` and the per-class keys, computed once per epoch. | **(breaking)** |
| **The `val_acc` → `val_accuracy` rename.** Anything reading TensorBoard/CSV runs by the old key needs the new one; `logged_val_keys` is the single source of the names a run produces. | **(breaking)** |
| **Metrics were built for the wrong task.** `SegmentationModel` gave every task `Accuracy`/`IoU`/`DiceScore`, including enhancement -- whose target is an image -- so `IoU.update` scattered an image into a one-hot tensor and produced nonsense. Metrics now come from `get_metrics(task, ...)`, and a task with no pixel-wise metric trains with an empty set instead of raising. | **fix** |
| **`save_hyperparameters(ignore=["model", "criterion"])` is dropped.** It worked because `ConfigLoader` handed out a plain dict; a pydantic `SBBConfig` is not YAML-representable and would break every checkpoint write. The config lives on `self.config` and inside every checkpoint. | **api** |
| **Two `ModelCheckpoint`s and two `EarlyStopping`s were registered unconditionally**, so any task that never logs `val_acc` raised at its first validation epoch end. `build_callbacks` derives the monitored keys from what the module really logs. | **fix** |
| **`initialization.py`'s `l2_regularization` returned `weight_decay * ‖p‖₂` -- a norm, not `0.5 * sum(p²)`** -- and `elastic_net_regularization` applied `weight_decay` to both of its terms. Both formulas are reproduced exactly so old runs match, and `regularization_loss` documents what they compute rather than what their names suggest. | **api** |

### 3.7 Inference and rendering

| Change | Marker |
| --- | --- |
| **The input page was shrunk to the prediction, not the other way round:** `visualize_model_output` called `resize_image(img, layout.shape[0], layout.shape[1])`, so a 3000×4000 page was resampled down to the model's 448×672 output before the blend and the saved overlay was a thumbnail of its input. The *mask* is upscaled now (nearest, via `resize_mask`) and the original page is kept at full size. | **fix** |
| **Bilinearly resizing the label map back to page size invented class ids** the label set never contained (interpolating `[0, 1, 2]` yields `[0, 1, 1, 2]` plus everything between). `resize_mask` is nearest-neighbour and refuses to invent. | **fix** |
| **Normalisation was applied before resizing, and the classification path skipped it entirely** -- so it scored a different input distribution than the one it was trained on. Order is now `resize` → `normalize`, and every task normalises. | **fix** |
| **`--save` behaviour is explicit.** The legacy script required `-s` for segmentation (and `sys.exit(1)`-ed without it), wrote one file for binarization and enhancement and *nothing at all* for classification. `sbb infer` writes `<stem>_prediction.png` (and `<stem>_layout.png` with `--save-layout`) into `--output-dir`, defaulting to the run directory's `predictions/`, and asking for a rendering the task did not produce is an error instead of a silently missing file -- which is why a classification run needs the JSON-label output listed in section 5.6 before `--save` can serve it. | **fix** |
| **`--patches` and the entire `reading_order` inference branch were `pass` statements.** `sbb infer --patches` now runs a real sliding window (`Predictor.predict_grid`, 0.5 overlap by default -- open question in section 5.3), and the reading order itself is computed in [gt/reading_order.py](../src/sbb_pytorch/gt/reading_order.py) instead of being a comment inside the inference script. | **fix** |
| **`--ground_truth` was documented as a directory, read as a file** (`cv2.imread(self.ground_truth)`), and its IoU was computed and thrown away -- `self.calculate_iou(gt, pred)`'s return value was never printed. The numbers exist in the port (`metrics.mean_iou` / `accuracy` / `dice_score`, or the CLI's table via `compute_metrics`); an `--ground-truth` flag on `sbb infer` is open question 5.6. | **fix** |
| **`model_dir/model.pt` was hard-coded** with two branches for the two checkpoint layouts in the wild; `latest_checkpoint` plus `Checkpoint` now handle every layout the repo has written. | **api** |
| **`color_images` hard-coded `{0: white, 1: red, 2: blue}`**, so a four-class config rendered every class from 3 upwards black -- and the two non-white colours were swapped relative to their own comments, because the array was treated as BGR here and as RGB everywhere else. Rendering uses the one palette `gt.render` also uses. | **fix** |

### 3.8 Ground-truth generation

| Change | Marker |
| --- | --- |
| **`shapely` and `matplotlib` are no longer dependencies of this package.** The legacy code mixed `cv2.contourArea` and `shapely.geometry.Polygon(...).area`, which disagree on collinear and self-intersecting polygons -- so the two copies of the area filter disagreed about which contours to keep. Areas are measured with `cv2.contourArea` throughout. | **api** |
| **The namespace prefix was derived with `alltags[0].split('}')[0] + '}'`**, which yields the nonsense prefix `"PcGts}"` for a namespace-less document and then leaves `imageHeight`/`imageWidth` unbound. The parser derives it from the `{uri}tag` convention and falls back to no prefix. | **fix** |
| **Five near-identical XML readers (~1000 duplicated lines)** become one parser; five rasterising functions, each with its own colour table and blend maths, become one `label_image` plus two derived views (`2d`, `3d`). | **api** |
| **`get_images_of_ground_truth`'s two near-duplicate halves** (single-class and layout branches), each re-implementing XML traversal, contour extraction, print-space cropping, column resizing and `fillPoly`, share every helper now and differ only in how they assemble the label map. | **api** |
| **Output naming is preserved** (`000123456-1800.xml` → `1800`, whole stem when the name has no hyphen, full stem for `machine_based_reading_order`), because existing datasets and the training loaders depend on it. The legacy code reached it with `gt_list[index].split('-')[1].split('.')[0]` inside a bare `except:`. | **api** |
| **The default `config_params=None` crashed the moment the layout branch ran** (`AttributeError: 'NoneType' object has no attribute 'keys'`) even though `--config` was documented as ignorable. `GroundTruthConfig.parse` makes the config single and typed; a region family the config does not label is *skipped and reported*, not a `KeyError`. | **fix** |
| **`case_label` was applied to every textline/word/glyph/printspace region whatever the use case**, so a bare `layout` config labelled the page's PrintSpace with `1` even with `printspace_as_class_in_layout` unset -- a page it could not label produced an image instead of being skipped. | **fix** |
| **The overlay blend used `cv2.addWeighted(img, 0.5, layout, 0.1, 0)`** -- weights summing to 0.6, which dimmed the result -- and drew onto a single-channel canvas it then mixed with a three-channel photo. `gt.render.overlay` owns the maths and normalises both operands. | **fix** |
| **The `3d` palette keeps the legacy `labels_rgb_color` table verbatim** -- despite the name it was handed to OpenCV and therefore behaved as BGR, so keeping the order keeps the rendered colours identical. | **api** |

### 3.9 Packaging and license

| Change | Marker |
| --- | --- |
| **The license is Apache-2.0, not MIT.** `LICENSE` has been the Apache-2.0 text all along while `README.md` said MIT; the README now matches the file, and `pyproject.toml` declares `license = "Apache-2.0"` with `license-files = ["LICENSE"]`. | **fix** |
| **`requirements.txt` → `pyproject.toml`.** `sacred` is dropped (only `build_model_load_pretrained_weights_and_save.py` used it, and that file is a Keras script); `torch`/`torchvision` are declared but the active build (CPU / cu118 / cu121 / rocm) must still be chosen before install, which pip cannot decide by itself; floors are raised where the code requires it (`numpy>=1.24,<3`, `albumentations>=1.4.5` because earlier versions cannot import against numpy 2.x, `scipy>=1.11` because `scipy.ndimage.filters`/`.interpolation` aliases are gone in 1.16), and `pydantic`, `click`, `rich`, `tensorboard` join the list. Seven declared packages are never imported by `src/` (`shapely`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `tqdm`, `Pillow`) -- section 5.7. | **api** |
| **The package is installable** (`pip install -e .`), so `sbb` is on `PATH` and `import sbb_pytorch` works from anywhere; the root `__init__.py` that made the source tree an accidental module is gone. | **api** |
| **One CLI, three argument-parsing styles fewer.** `train.py` used `argparse`, `inference_pytorch.py` a bare `click.command`, `generate_gt_for_training.py` its own `click.group` -- so the same idea was `--batch-size` in one and `--batch_size` in the next, and `--out` meant "directory of reading-order XML" in one and "output directory" in the other. Everything is `sbb <subcommand>` now. | **api** |
| **Logging is idempotent.** The pre-refactor repo had one logger (in `train.py`) that appended a fresh `FileHandler` on every call, so calling it twice duplicated every later line; everything else used `print()` (`utils.py` had 11 calls, 8 of them commented-out debug prints). | **fix** |

## 4. API-only changes: where every symbol lives

Nothing below moves a number; it is the import table for a script being ported.

| Legacy symbol | New location |
| --- | --- |
| `ConfigLoader` (class) | `config.loaders.load_config` / `load_page2label` / `load_scales` / `apply_overrides` / `dump_json_schema`; `SBBConfig.from_dict` / `.from_json` |
| `ModelConfig`, `TransformerConfig`, `AugmentationConfig`, `AugmentationParams`, `TrainingConfig`, `DirectoryConfig`, `Page2LabelConfig`, `ScalesConfig` | three pydantic models in [config/models.py](../src/sbb_pytorch/config/models.py): `SBBConfig`, `TransformerConfig`, `Page2Label`, `ScalesConfig` (the eight flat dataclasses are fields or nested models now) |
| `SegmentationModel` (Lightning module) | `training.SBBLightningModule` |
| `get_optimizer`, `initialize_weights`, `kaiming_init`, `orthogonal_init`, `xavier_init`, `GradientClipper`, `WeightRegularizer` | `training.factory.build_optimizer`, `init_weights`, `kaiming_init`, `orthogonal_init`, `xavier_init`, `clip_gradients`, `regularization_loss` |
| `calculate_class_weights` | `training.factory.class_weights` / `class_weights_from_directory` (the folder ambiguity between a split root and an images dir is resolved in `_label_directory`) |
| `model_utils.save_model` / `load_model` / `save_checkpoint` | `storage.checkpoint.Checkpoint` (+ `latest_checkpoint`, `prune_checkpoints`) |
| `create_model`, `ResNet50UNet`, `ResNet50UNetLight`, `ViTResNet50UNet`, `ViTResNet50UNetTransformerBeforeCNN`, `EnhancementModel`, `MachineBasedReadingOrderModel`, the classification branch | `models.registry.create_model` / `models.registry.MODEL_REGISTRY` and the classes in `models/unet.py`, `models/vit.py`, `models/heads.py` |
| `metrics.MetricTracker`, `Accuracy`, `IoU`, `DiceScore`, `seg_metrics`, `mean_iou`, `jaccard_distance_loss`, `class_tversky`, `generalized_dice_coeff`, `focal_loss`, `SSIM`, `PSNR` | `metrics.Metric`, `metrics.ConfusionMatrixMetric`, `metrics.mean_iou` / `dice_score` / `accuracy` / `dice_macro` / `dice_micro`, and the formulas in `metrics.functional` / `metrics.enhance` |
| `losses.DiceLoss`, `FocalLoss`, `CombinedLoss`, `SSIMLoss`, `get_loss_function` | `sbb_pytorch.losses` (same names) plus `SoftDiceLoss`, `TverskyLoss`, `FocalTverskyLoss`, `GeneralizedDiceLoss`, `JaccardLoss`, `CrossEntropy` and `WeightedCrossEntropyLoss` |
| `dataset.SegmentationDataset`, `dataset.get_dirs_or_files`, `dataset.DataModule`, `dataset.preprocess_and_augment_data` | `data.SegmentationDataset`, `data.SBBDataModule`, `data.make_loader`, `data.patches` |
| `augmentation.get_training_augmentations`, `get_validation_augmentations`, `get_enhancement_augmentations`, `get_augmentations` | `data.augmentations` (one factory) |
| `utils.provide_patches` (43 positional arguments) and `get_patches`, `get_patches_num_scale`, `get_patches_num_scale_new` | `data.patches.PatchConfig` + `data.patches.variants` + `data.patches.write_patches`; both historical file-naming schemes survive because run directories on disk use them |
| `utils.resize_image` (x3), `otsu_copy`, `do_brightening`, `do_degrading`, `shift_image_and_label`, `return_shuffled_channels`, `rotation_not_90_func`, `scale_image_for_no_patch`, `bluring` | `storage.images.resize`, `resize_mask`, `otsu_binarize`, `brighten`, `downscale_upscale` + `add_noise`, `shift_image_and_label`, `shuffle_channels`, `rotate90` + `rotate_not_90`, `blur`, `imread_rgb`/`imread_bgr`/`imwrite`, `read_mask`, `remap_mask`, `one_hot_mask`, `to_uint8` |
| `inference_pytorch.SBBPredict` | `inference.Predictor` (+ `inference.Prediction`) |
| `inference_pytorch.color_images`, `visualize_model_output`, `make_image_from_bb` | `inference.visualize.render` / `colorize` / `overlay` / `binarization_image` / `enhancement_image` / `save_images` |
| `gt_gen_utils.read_xml`, `get_textline_contours_for_visualization`, `get_layout_contours_for_visualization`, `get_contours_of_image`, `get_contours`, `filter_contours_area_of_image`, `filter_contours_area_of_image_tables`, `return_parent_contours`, `get_contours_of_region`, `update_region_contours`, `bounding_box`, `find_new_features_of_contours`, `rotatedRectWithMaxArea`, `largest_contour`, `get_largest_contour_bbox`, `resize_image` | `gt.parser.parse_pagexml` / `discover_xml_files`, `gt.geometry.contours_of_image` / `filter_contours_by_area` / `return_parent_contours` / `contours_of_region` / `update_region_contours` / `bounding_box` / `ContourFeatures` / `rotated_rect_with_max_area` / `largest_contour_bbox`, and `storage.images.resize` |
| `gt_gen_utils.get_images_of_ground_truth`, `visualize_image_from_contours_layout`, `visualize_image_from_contours`, `visualize_model_output`, `make_image_from_bb` | `gt.generator.generate_ground_truth` and `gt.render.render_page` / `label_image` / `label_map_to_image` / `paint_contours` |
| the `pagexml2label`, `image_enhancement`, `machine_based_reading_order`, `visualize_reading_order`, `visualize_textline_segmentation`, `visualize_layout_segmentation` commands | `sbb gt generate` and `sbb gt visualize`; `image_enhancement` and `machine_based_reading_order` are ported as code (`gt.reading_order.generate_pairs` / `write_pair`, `storage.images.downscale_upscale`) but have no command yet -- section 5.6 |
| `train.py`'s argparse flags (`--config`, `--page2label`, `--output-dir`, `--batch-size`, `--num-workers`, `--epochs`, `--patience`, `--device`, `--resume`) | `sbb train CONFIG` with `--page2label`, `-o/--output-dir`, `-b/--batch-size`, `-w/--num-workers`, `-e/--epochs`, `--patience`, `-d/--device`, `--resume`, plus `-m/--model`, `--pretrained/--no-pretrained`, `--scales`, `--every-n-steps` (the old `--save-interval`), `--logger/--no-logger`, `--progress-bar/--no-progress-bar`, `--dry-run` |
| `inference_pytorch.py`'s click options (`--image`, `--out`, `--patches/--no-patches`, `--save`, `--save_layout`, `--model`, `--ground_truth`, `--xml_file`, `--min_area`) | `sbb infer MODEL [INPUTS...]` with `-i/--input` (repeatable), `-o/--output-dir`, `--save/--no-save`, `--save-layout/--no-save-layout`, `--patches/--no-patches`, `-c/--config`, `--architecture`, `--pairing`, `-d/--device`, `--limit`. `--model` splits into the positional `MODEL` (a run directory) and `--architecture` (a registry key); `--ground_truth`/`--xml_file`/`--min_area` have no flag yet (section 5.6) |
| `build_model_load_pretrained_weights_and_save.py` | `sbb export CHECKPOINT [-o dest]` |
| `config_loader.py`'s `ConfigLoader.validate_configs()`, and the question "does this config file load?" | `sbb validate CONFIG [--page2label ...] [--scales ...] [--strict]` |
| the hand-written config docs (`config.json` conventions, the README's option table) | `sbb config-schema [-o docs/config-reference.json]` |

## 5. Open questions for the maintainer

These are deliberate non-decisions: places where the refactor found a value that is probably wrong
but where changing it would silently rewrite someone's results, so the change is offered here
instead of being made.

1. **`num_patches_xy` in `config_params.json` says `[56, 56]` for a 448×448 input with 4×4
   patches.** That geometry yields `[112, 112]` (12544 tokens); the config's number came from the
   TensorFlow project. The ViT model now *checks* a supplied value against the derived one and
   fails at build time with both numbers in the message. Which is authoritative: the config file
   (keep `[56, 56]` and shrink the derived geometry) or the arithmetic (`configs/*.json` should
   ship `[112, 112]`)?
2. **`class_weights` divides by its own minimum and then renormalises** -- `w = 1/(count + 1e-6)`,
   normalise, divide by `min`, normalise again. The middle step puts the most frequent class at
   `1.0` and the last one undoes it: the result is the same normalised distribution either way, so
   the pair is redundant. It is kept verbatim because it produced the published numbers. Confirm it
   can be simplified, or leave it.
3. **Sliding-window inference (`--patches`) had no implementation to preserve** -- the branch was
   `pass`. The port averages overlapping tiles (`Predictor.predict_grid`, overlap 0.5) at the
   configured input size. Is 0.5 the right default for 448×672 tiles on a 3000×4000 page, should
   `--patches` become the default whenever the input is larger than the model's input size, and
   should the overlap be a `--overlap` flag (on `sbb infer` it is not one yet)?
4. **Which device should the config ask for?** The legacy default is `"cuda"` (which the old code
   read and then ignored), `resolve_device("auto")` walks `cuda → mps → cpu`, and `sbb --device`
   can override it per run. Opting a shipped config into `"auto"` is a behaviour change on a Mac or
   a CPU-only box, so it has not been done silently.
5. **`pairing="legacy"` is available but not the default.** `create_model` and `sbb infer` default
   to the corrected `pairing="aligned"`; a checkpoint trained against the old `ResNet50UNet` needs
   `--pairing legacy`. Are there published checkpoints we should treat as the default target, and
   should `Checkpoint` record the pairing so inference picks it up on its own?
6. **Two legacy commands and three legacy flags have no CLI surface yet.** `image_enhancement`
   (degraded images from `scales_enhancement.json`) and `machine_based_reading_order` (the `.png` +
   `.npy` pair samples) are ported as code -- `storage.images.downscale_upscale`,
   `gt.reading_order.generate_pairs` / `write_pair` -- but not as `sbb gt enhance` /
   `sbb gt reading-order`; and `inference_pytorch.py`'s `--ground_truth` (score a prediction),
   `--xml_file` and `--min_area` have no equivalent flag on `sbb infer`. Add them, or declare the
   current surface complete and say so in the README?
7. **Seven declared dependencies are imported nowhere in `src/`** -- `shapely`, `matplotlib`,
   `seaborn`, `scikit-learn`, `scipy`, `tqdm`, `Pillow`. They carried over from `requirements.txt`
   (`matplotlib` also arrives transitively through `albumentations`); `sacred` was dropped because
   only the deleted Keras script used it. Drop the rest as well, or keep them because an external
   script still imports one of them?
8. **`docs/migration.md` says what changed; nothing yet verifies it automatically.** A useful next
   step is a small "reference run" fixture -- one tiny dataset, one 1-epoch train, one checkpoint,
   one inference -- whose numbers and file names are compared against the legacy script's output on
   a machine with the old code checked out. Worth building before the root scripts are deleted?
9. **`train.md` (27 KB of legacy run notes) and `TECHNICAL_CHANGELOG.md` (17 KB, describing the
   TensorFlow → PyTorch port) both describe files this branch deletes.** They are being rewritten
   rather than dropped; confirm whether the TensorFlow-port history belongs in `CHANGELOG.md`, in
   `TECHNICAL_CHANGELOG.md`, or in a `docs/legacy/` folder.
10. **Sandbox artifacts in the repository root** (`.bashrc`, `.bash_profile`, `.profile`, `.zshrc`,
   `.zprofile`, `.gitconfig`, `.mcp.json`, `.vscode`, `.idea`, `.ripgreprc`, `__pycache__/`,
   `.pytest_cache/`, `.ruff_cache/`) are tracked/untracked leftovers of the environment, and
   `.gitmodules` is an empty file with no submodule. The planned `.gitignore` excludes them; the
   empty `.gitmodules` is deleted. Say so if `.gitmodules` should be kept.

## 6. Verifying a change

For any entry above, the order to check it in:

1. Read the module it names -- every fix above is documented *at its own definition*, not only
   here; the docstrings carry the legacy line references.
2. Run the matching test file: `tests/test_metrics.py`, `test_losses.py`, `test_models.py`,
   `test_dataset.py`, `test_augmentations.py`, `test_datamodule.py`, `test_images.py`,
   `test_checkpoint.py`, `test_config.py`, `test_inference.py`, `test_training.py`,
   `test_training_factory.py`.
3. For a **breaking** number, re-run one of your own historic configs with the legacy code and
   the package side by side and compare the two logged values against the formula in section 3 --
   the change should be explainable arithmetically, not just by "the old code was wrong".
