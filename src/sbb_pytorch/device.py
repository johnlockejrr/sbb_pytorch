"""Device selection in one place, so ``config.device``, ``sbb --device`` and ``torch.device`` agree.

``train.py`` and ``inference_pytorch.py`` each wrote their own
``torch.device('cuda' if torch.cuda.is_available() else 'cpu')``, while the ``device`` field
``config_loader.py`` stored was validated by nothing and read by nothing else -- so a config
saying ``cuda`` on a machine without CUDA died inside ``Tensor.to`` with *"Torch not compiled
with CUDA enabled"* instead of telling the user what was available.

:func:`resolve_device` is now the single implementation the CLI, the training module and the
predictor all call:

* ``"auto"`` walks ``cuda -> mps -> cpu`` and returns the first that exists -- which is exactly
  what the two inline expressions did.
* An explicit ``"cuda:1"``, ``"mps"`` or ``"cpu"`` is honoured verbatim, and asking for a device
  that is not there raises ``ValueError`` naming what *is* available rather than failing at the
  first tensor. The config keeps its legacy ``"cuda"`` default, so opting into ``auto`` is a
  deliberate change (see ``docs/migration.md``).
"""

from __future__ import annotations

import torch

__all__ = ["available_devices", "resolve_device"]


def _mps_available() -> bool:
    """``True`` on an Apple-Silico build; the attribute itself is absent on older torch builds."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    """Every device torch can use here, best first, always ending in ``cpu``."""
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    if _mps_available():
        devices.append("mps")
    return [*devices, "cpu"]


def resolve_device(spec: str | None = "auto") -> torch.device:
    """Turn a config or CLI device string into a device that actually exists.

    Args:
        spec: ``"auto"`` (pick the best available), or an explicit ``cpu`` / ``mps`` / ``cuda`` /
            ``cuda:N``. ``None`` means ``"auto"``.

    Raises:
        ValueError: the string is not a device torch knows, or names one that is unavailable
            here -- ``ValueError("device 'cuda:1' is not available here; torch reports
            ['cpu']")`` instead of the downstream ``RuntimeError`` from ``Tensor.to``.
    """
    requested = "auto" if spec is None else str(spec).strip().lower()
    if not requested:
        raise ValueError("device must be a non-empty string such as 'auto', 'cpu' or 'cuda:0'")

    if requested == "auto":
        return torch.device(available_devices()[0])

    try:
        device = torch.device(requested)
    except (AssertionError, RuntimeError) as exc:  # torch rejects an unparseable device string
        raise ValueError(f"device {spec!r} is not a device torch knows") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"device {spec!r} was requested but torch reports no CUDA here "
                f"(available: {available_devices()}); pass --device auto to fall back"
            )
        # torch builds ``cuda:7`` happily on a 1-GPU host and only fails at the first tensor.
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"device {spec!r} is not available here; torch reports {available_devices()}"
            )
    if device.type == "mps" and not _mps_available():
        raise ValueError(
            f"device {spec!r} was requested but MPS is unavailable "
            f"(available: {available_devices()}); pass --device auto to fall back"
        )
    return device
