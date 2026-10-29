"""Exception hierarchy.

Every failure the package raises deliberately is one of these, so callers can catch
``SBBError`` rather than guessing whether the code will raise ``TypeError``, ``IndexError``
or -- as the pre-refactor code did -- silently return a blank image.
"""

from __future__ import annotations


class SBBError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(SBBError):
    """A configuration file is missing, unreadable, or fails validation.

    Raised by :mod:`sbb_pytorch.config.loaders` and by :func:`click` parameter callbacks.
    Field-level detail comes from pydantic's ``errors()`` payload.
    """


class ImageReadError(SBBError):
    """``cv2.imread`` returned ``None``.

    The pre-refactor code let ``None`` flow into ``cv2.resize``/``np.copy``, which raised
    ``TypeError`` hundreds of lines away from the unreadable file -- or, inside a bare
    ``except:``, silently substituted an all-white image and an all-zero mask. Callers now
    get the path in the message instead.

    ``cv2.imread`` also returns ``None`` for non-ASCII paths on some platforms, which is
    worth checking before blaming a corrupt file.
    """

    def __init__(self, path: str, reason: str = "could not be decoded") -> None:
        self.path = path
        super().__init__(f"image {path!r} {reason}")


class CheckpointError(SBBError):
    """A checkpoint file is missing, unreadable, or has an unrecognised layout.

    See :mod:`sbb_pytorch.storage.checkpoint` for the three historical key spellings this
    package still accepts on read.
    """


class ModelBuildError(SBBError):
    """No model could be built from a config, or a model's forward pass is inconsistent.

    Covers the pre-refactor class of error where a decoder block was handed a channel count
    its ``Conv2d`` did not declare (``RuntimeError`` at the first forward pass rather than at
    construction), which is now checked when the model is built.
    """


class TrainingError(SBBError):
    """A training step cannot be set up or run.

    Covers what the legacy code raised as bare ``ValueError`` from three separate
    ``get_optimizer`` implementations (``Unknown optimizer: adam``), and what the checkpoint and
    metric paths raised as ``KeyError`` when a logged metric a callback asked for was never
    produced -- ``trainer.callback_metrics['val_acc']`` on an enhancement run, for instance.
    """
