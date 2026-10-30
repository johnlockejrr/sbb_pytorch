"""Config loading -- replaces the ``ConfigLoader`` class with functions and no hidden state.

``ConfigLoader`` kept ``config``, ``page2label_config`` and ``scales_config`` as mutable
instance attributes, so ``get_training_config()`` silently depended on ``load_config()`` having
run first (calling it first raised ``AttributeError`` on ``self.config``). These functions take
their input as arguments and return a value, which removes that ordering constraint entirely.

Typical use from the CLI::

    config = load_config("configs/textline.json", page2label_path="configs/page2label.json")
    config = apply_overrides(config, epochs=30, device="cpu")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from sbb_pytorch.config.models import (
    Page2Label,
    SBBConfig,
    ScalesConfig,
    validate_pair,
)
from sbb_pytorch.exceptions import ConfigError
from sbb_pytorch.logging import get_logger

__all__ = ["apply_overrides", "dump_json_schema", "load_config", "load_page2label", "load_scales"]

M = TypeVar("M", bound=BaseModel)

#: ``train.py`` flag names mapped onto the fields they override (``--epochs`` -> ``n_epochs``).
_OVERRIDE_ALIASES = {
    "epochs": "n_epochs",
    "batch_size": "n_batch",
    "workers": "num_workers",
    "lr": "learning_rate",
    "resume": "dir_of_start_model",
    "start_model": "dir_of_start_model",
}


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    """Read a JSON file, reporting the path in every failure mode."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{label} config not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{label} config could not be read ({path}): {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{label} config is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{label} config must contain a JSON object, got {type(data).__name__}")
    return data


def _build(model: type[M], data: dict[str, Any], path: Path) -> M:
    """Validate ``data`` into ``model``, turning pydantic's report into a ``ConfigError``."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path.name}: {_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    """One line per bad field -- pydantic's default rendering is correct but hard to scan."""
    lines = [f"{exc.error_count()} validation error(s)"]
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']} (got {error.get('input')!r})")
    return "\n".join(lines)


def load_page2label(path: str | Path) -> Page2Label:
    """Load ``custom_config_page2label.json`` on its own."""
    return _build(Page2Label, _read_json(path, "page2label"), Path(path))


def load_scales(path: str | Path) -> ScalesConfig:
    """Load ``scales_enhancement.json`` on its own."""
    return _build(ScalesConfig, _read_json(path, "scales"), Path(path))


def load_config(
    path: str | Path,
    page2label_path: str | Path | None = None,
    scales_path: str | Path | None = None,
) -> SBBConfig:
    """Load a main config, optionally attaching the page-to-label and scales files beside it.

    Args:
        path: The main config (``config_params.json``, ``config_train_textline.json``, ...).
        page2label_path: Written into :attr:`SBBConfig.page2label` if the main config does not
            already inline one.
        scales_path: Written into :attr:`SBBConfig.enhancement_scales`.

    Raises:
        ConfigError: the file is missing, is not JSON, or fails schema validation. Cross-config
            advisories are *not* raised here -- see :func:`sbb_pytorch.config.models.validate_pair`
            and :meth:`SBBConfig.warnings`, which the CLI prints instead.
    """
    path = Path(path)
    config = _build(SBBConfig, _read_json(path, "main"), path)

    if page2label_path is not None and config.page2label is None:
        config.page2label = load_page2label(page2label_path)
    if scales_path is not None and config.enhancement_scales is None:
        config.enhancement_scales = load_scales(scales_path)

    # Log rather than raise: an inconsistent-but-shipped pair must still be loadable.
    for problem in validate_pair(config):
        get_logger("sbb.config").warning("%s: %s", path, problem)
    return config


def apply_overrides(config: SBBConfig, **overrides: Any) -> SBBConfig:
    """Return ``config`` with CLI overrides applied; ``None`` values are ignored.

    ``train.py`` exposed ``--batch-size``, ``--epochs``, ``--num-workers``, ``--patience``,
    ``--device`` and ``--resume`` as overrides of the file, and several of them had no config
    field at all, so they could not be set persistently. Now they are fields *and* flags: the
    flag wins when given, otherwise the file, otherwise the field default.

    Raises:
        ConfigError: an override name is not a config field (or a known flag alias), or the
            new value is invalid.
    """
    fields = set(type(config).model_fields)
    updates: dict[str, Any] = {}
    for name, value in overrides.items():
        if value is None:
            continue
        key = name if name in fields else _OVERRIDE_ALIASES.get(name, name)
        if key not in fields:
            raise ConfigError(
                f"unknown config option {name!r}; {type(config).__name__} defines "
                f"{', '.join(sorted(fields))}"
            )
        updates[key] = value

    if not updates:
        return config
    try:
        return type(config).model_validate({**config.model_dump(mode="json"), **updates})
    except ValidationError as exc:
        raise ConfigError(f"invalid override: {_format_errors(exc)}") from exc


def dump_json_schema(dest: str | Path = "docs/config-reference.json") -> Path:
    """Write ``docs/config-reference.json`` -- the generated schema, and the only config doc.

    The pre-refactor configs were documented by a hand-written table that had already drifted
    from the code; this file is generated from :class:`SBBConfig`, so it cannot drift, and there
    is no second copy to keep in step (a Markdown reference used to be named here while only the
    JSON one existed).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    schema = SBBConfig.model_json_schema()
    dest.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
