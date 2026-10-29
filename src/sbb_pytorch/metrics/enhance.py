"""PSNR and SSIM for the enhancement task, with the two construction-time bugs fixed.

Bugs fixed here
---------------
* **``losses.SSIMLoss`` could not be constructed at all.** Its ``_gaussian`` helper called
  ``np.exp`` but ``losses.py`` never imported numpy, so ``SSIMLoss()`` raised ``NameError`` --
  and since ``losses.get_loss_function("enhancement")`` is the only way to build an enhancement
  loss, that whole task was unreachable. (``metrics.SSIM`` had the same helper with numpy
  imported, so the two copies only agreed on being broken differently.) Here the window is built
  with ``torch.exp`` and no numpy at all.
* **The "Gaussian" window was not a Gaussian.** The legacy exponent was ``-x**2 / sigma**2``; a
  Gaussian is ``-x**2 / (2 * sigma**2)``, so ``sigma=1.5`` actually produced a window with an
  effective sigma of ``1.5 / sqrt(2) ~= 1.06``. The window is now the real thing and matches
  ``skimage.feature.general_image_gradient``/``structural_similarity`` windowing.
* **Zero padding instead of reflect padding.** ``F.conv2d(..., padding=window_size // 2)`` pads
  with zeros, so every border window averaged in black pixels and systematically depressed SSIM
  on real pages. The windows are now reflect-padded with a valid convolution, which is what
  ``skimage.metrics.structural_similarity`` does.
* **``C1``/``C2`` hard-coded the dynamic range to 1.** They were written as ``0.01 ** 2`` and
  ``0.03 ** 2``, which are only correct for images scaled to ``[0, 1]``; images decoded from
  disk are ``0..255``. Both constants now derive from an explicit ``data_range``.
* **PSNR divided by zero** when the prediction equalled the target exactly
  (``20 * log10(max_val / sqrt(mse))`` with ``mse == 0`` yields ``inf`` silently), and
  ``metrics.PSNR`` averaged per-batch PSNRs instead of pooling the squared error, so the epoch
  value depended on batch sizes. The pooled MSE is now accumulated and converted once.
* **Two ~30-line SSIM copies** (``metrics.SSIM`` and ``losses.SSIMLoss``) existed with different
  bugs. One implementation, exported as a function and wrapped as a metric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from sbb_pytorch.metrics.segmentation import Metric

__all__ = [
    "PSNRMetric",
    "SSIMMetric",
    "gaussian_window",
    "psnr",
    "psnr_from_mse",
    "ssim",
    "ssim_map",
]


def _as_image_batch(tensor: Tensor, name: str) -> Tensor:
    """Validate a ``(B, C, H, W)`` float image batch and return it."""
    tensor = torch.as_tensor(tensor)
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be a (B, C, H, W) image batch, got shape {tuple(tensor.shape)}; "
            "add the batch dimension rather than relying on inference"
        )
    if not torch.is_floating_point(tensor):
        raise TypeError(
            f"{name} must be float (scale uint8 images by 1/255 or pass data_range=255), got "
            f"dtype {tensor.dtype}"
        )
    return tensor


def _check_pair(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Validate that two image batches are comparable."""
    predictions = _as_image_batch(predictions, "predictions")
    targets = _as_image_batch(targets, "targets")
    if predictions.shape[1] != targets.shape[1]:
        raise ValueError(
            f"channel mismatch: predictions have {predictions.shape[1]}, targets have "
            f"{targets.shape[1]}"
        )
    return predictions.to(targets.dtype), targets


# ------------------------------------------------------------------------ PSNR


def psnr_from_mse(mse: float | Tensor, data_range: float = 1.0) -> Tensor:
    """``20 * log10(data_range / sqrt(mse))``, with the identical-image case handled.

    Args:
        mse: Mean squared error. ``0`` means a perfect reconstruction.
        data_range: Dynamic range of the images -- ``1.0`` for normalised floats, ``255.0`` for
            uint8-derived images. Passing the wrong one shifts PSNR by ``20*log10(255) ~= 48 dB``,
            which is why it is an explicit argument and not a constant.

    Returns:
        PSNR in decibels; ``inf`` when ``mse == 0`` (the legacy code produced ``inf`` too, but
        via a division by zero that some backends report as an error).
    """
    mse = torch.as_tensor(mse, dtype=torch.float64)
    if float(data_range) <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    if float(mse) <= 0.0:
        return torch.tensor(float("inf"), dtype=torch.float64)
    return 20 * torch.log10(torch.as_tensor(float(data_range), dtype=torch.float64) / mse.sqrt())


def psnr(predictions: Tensor, targets: Tensor, data_range: float = 1.0) -> Tensor:
    """Peak signal-to-noise ratio between two image batches.

    The MSE is taken over every element of the batch (not per image then averaged), matching
    ``skimage.metrics.peak_signal_noise_ratio``.

    Raises:
        ValueError: the batches have different shapes or channel counts.
    """
    predictions, targets = _check_pair(predictions, targets)
    mse = ((predictions - targets) ** 2).mean()
    return psnr_from_mse(mse, data_range)


class PSNRMetric(Metric):
    """Accumulates squared error across batches and reports one PSNR per epoch.

    ``metrics.PSNR`` stored each batch's PSNR and averaged them, which is not the same number:
    ``mean(log(mse_i)) != log(mean(mse_i))``, and the result depended on how the validation set
    happened to be batched. Accumulating the MSE and converting once is both cheaper and exact.

    Args:
        data_range: Dynamic range the images were normalised to. ``losses``/``metrics`` both
            defaulted to ``1.0`` while reading ``0..255`` imagery; choose deliberately.
    """

    name = "psnr"

    def __init__(self, data_range: float = 1.0) -> None:
        if float(data_range) <= 0:
            raise ValueError(f"data_range must be positive, got {data_range}")
        self.data_range = float(data_range)
        self._mse_sum = torch.zeros((), dtype=torch.float64)
        self._count = 0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Add one batch's squared error."""
        predictions, targets = _check_pair(predictions, targets)
        self._mse_sum = self._mse_sum + ((predictions - targets) ** 2).sum().to(torch.float64)
        self._count += int(predictions.numel())

    def reset(self) -> None:
        self._mse_sum = torch.zeros((), dtype=torch.float64)
        self._count = 0

    def compute(self) -> dict[str, float]:
        """``{"psnr": ..., "mse": ...}``; ``nan`` for both if nothing has been accumulated."""
        if self._count == 0:
            return {"psnr": float("nan"), "mse": float("nan")}
        mse = self._mse_sum / self._count
        return {
            "psnr": float(psnr_from_mse(mse, self.data_range)),
            "mse": float(mse),
        }


# ------------------------------------------------------------------------- SSIM


def gaussian_window(
    window_size: int, sigma: float = 1.5, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Normalised 2-D Gaussian window of shape ``(1, 1, n, n)``, built in torch.

    **Fixed:** the legacy helper was
    ``torch.Tensor([np.exp(-(x - n // 2) ** 2 / float(sigma ** 2)) for x in range(n)])`` --
    ``np`` was never imported in ``losses.py`` (``NameError``), and the missing factor of ``2`` in
    the denominator made the window narrower than the ``sigma`` it claimed.

    Raises:
        ValueError: ``window_size`` is not odd or positive, or ``sigma`` is not positive.
    """
    size = int(window_size)
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if float(sigma) <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    centre = size // 2
    offsets = torch.arange(size, dtype=torch.float64) - centre
    kernel_1d = torch.exp(-(offsets**2) / (2 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.to(dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)


def _ssim_map(
    predictions: Tensor,
    targets: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """The ``(B, C, H, W)`` SSIM map -- the whole computation, shared by every entry point.

    Raises:
        ValueError: a zero/negative dynamic range, or an even window size (via
            :func:`gaussian_window`).
    """
    if float(data_range) <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    channels = predictions.shape[1]
    height, width = predictions.shape[2], predictions.shape[3]
    window = gaussian_window(window_size, sigma, predictions.dtype).to(predictions.device)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    padding = window_size // 2

    def blur(images: Tensor) -> Tensor:
        # Reflect padding matches skimage; it requires padding < dimension, so an image smaller
        # than the window falls back to replicate instead of the zeros the legacy code padded with.
        if padding < min(height, width):
            images = F.pad(images, (padding, padding, padding, padding), mode="reflect")
        else:  # pragma: no cover - only for images smaller than the window
            images = F.pad(images, (padding, padding, padding, padding), mode="replicate")
        return F.conv2d(images, window, stride=1, groups=channels)

    mu_x = blur(predictions)
    mu_y = blur(targets)
    # clamp_min(0): a variance computed as E[x^2] - E[x]^2 can go marginally negative from
    # floating-point cancellation, which used to leak a negative term into the ratio.
    sigma_x_sq = (blur(predictions * predictions) - mu_x * mu_x).clamp_min(0)
    sigma_y_sq = (blur(targets * targets) - mu_y * mu_y).clamp_min(0)
    sigma_xy = blur(predictions * targets) - mu_x * mu_y

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return numerator / denominator


def ssim(
    predictions: Tensor,
    targets: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Structural similarity between two ``(B, C, H, W)`` image batches.

    Mirrors ``skimage.metrics.structural_similarity`` with ``gaussian_weights=True`` and
    ``use_sample_covariance=False``: a normalised Gaussian window per channel, reflect padding
    with a valid convolution, and ``C1 = (0.01 * data_range) ** 2``, ``C2 = (0.03 * data_range) ** 2``.

    Returns:
        A ``(B,)`` tensor of per-image SSIM (the mean of each image's map over channels and
        positions, which is how skimage reduces a multi-channel image).

    Raises:
        ValueError: non-matching batch shapes, an even window size, or a zero dynamic range.
    """
    predictions, targets = _check_pair(predictions, targets)
    return _ssim_map(predictions, targets, data_range, window_size, sigma).mean(dim=(1, 2, 3))


def ssim_map(
    predictions: Tensor,
    targets: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """The raw per-pixel SSIM map ``(B, C, H, W)``, for localising where an enhancement failed."""
    predictions, targets = _check_pair(predictions, targets)
    return _ssim_map(predictions, targets, data_range, window_size, sigma)


class SSIMMetric(Metric):
    """Accumulates SSIM across batches; ``compute()`` returns the mean over everything seen.

    Replaces both ``metrics.SSIM`` and ``losses.SSIMLoss``, which duplicated the same window
    code and disagreed about whether ``ssim_map`` was averaged over channels or per image.
    """

    name = "ssim"

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> None:
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self._sum = 0.0
        self._count = 0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Add one batch's per-image SSIM values."""
        values = ssim(predictions, targets, self.data_range, self.window_size, self.sigma)
        self._sum += float(values.sum())
        self._count += int(values.numel())

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def compute(self) -> dict[str, float]:
        """``{"ssim": ...}``; ``nan`` when nothing has been accumulated."""
        if self._count == 0:
            return {"ssim": float("nan")}
        return {"ssim": self._sum / self._count}
