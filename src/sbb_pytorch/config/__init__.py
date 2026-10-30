"""Configuration for the package.

* :mod:`sbb_pytorch.config.models` -- the schema (:class:`~sbb_pytorch.config.models.SBBConfig`).
* :mod:`sbb_pytorch.config.loaders` -- the functions that read a config file.

Every field is documented by ``SBBConfig.model_json_schema()``, which
:func:`~sbb_pytorch.config.loaders.dump_json_schema` writes to
``docs/config-reference.json`` (``sbb config-schema`` runs it).
"""

from __future__ import annotations

from sbb_pytorch.config.loaders import (
    apply_overrides,
    dump_json_schema,
    load_config,
    load_page2label,
    load_scales,
)
from sbb_pytorch.config.models import (
    BackboneType,
    BlurKind,
    OptimizerType,
    OutputType,
    Page2Label,
    SBBConfig,
    ScalesConfig,
    TaskType,
    TransformerConfig,
    TypeOutput,
    UseCase,
    validate_pair,
)

__all__ = [
    "Page2Label",
    "SBBConfig",
    "ScalesConfig",
    "BackboneType",
    "BlurKind",
    "OptimizerType",
    "OutputType",
    "TaskType",
    "TransformerConfig",
    "TypeOutput",
    "UseCase",
    "apply_overrides",
    "dump_json_schema",
    "load_config",
    "load_page2label",
    "load_scales",
    "validate_pair",
]
