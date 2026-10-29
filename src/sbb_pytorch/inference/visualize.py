"""Every picture the CLI writes, from one palette and one overlay implementation.

``inference_pytorch.color_images`` hard-coded three colours --
``{0: [255,255,255], 1: [255,0,0], 2: [0,0,255]}`` annotated "white / red / blue" -- which
means (a) a four-class config rendered every class from 3 upwards black, and (b) the two
non-white colours were swapped relative to their own comments, because that array was treated
as BGR here and as RGB everywhere else (``[255,0,0]`` written by ``cv2.imwrite`` is blue). The
canonical table already exists in :mod:`sbb_pytorch.gt.render` -- the legacy
``labels_rgb_color`` list, verbatim -- so this module calls it instead of inventing a second
palette, which is what "one palette" in the refactor plan meant.

Two further rendering fixes are baked in:

* **The photo was shrunk to the prediction, not the other way round.** The legacy
  ``visualize_model_output`` called ``resize_image(img, layout.shape[0], layout.shape[1])``, so
  a 3000x4000 page was resampled down to the model's 448x672 output before the blend and the
  saved overlay was a thumbnail of the input. Here the *mask* is upscaled (nearest, through
  :func:`sbb_pytorch.storage.images.resize_mask`) and the original image is kept at full size.
* **``cv2.imwrite`` was called on ``int32``/float arrays** in some paths, producing 32-bit PNGs;
  every save now goes through :func:`~sbb_pytorch.storage.images.imwrite`, which converts to
  ``uint8`` and to the file's BGR order.
"""

from __future__ import annotations

import cv2
import numpy as np

from sbb_pytorch.config.models import TaskType
from sbb_pytorch.gt.render import DEFAULT_LABEL_PALETTE, label_map_to_image
from sbb_pytorch.storage.images import imwrite, resize, resize_mask, to_uint8

__all__ = [
    "OVERLAY_ALPHA",
    "binarization_image",
    "colorize",
    "enhancement_image",
    "overlay",
    "render",
    "save_images",
]

#: Blend weight of the coloured layout over the page -- the ``cv2.addWeighted`` 0.7/0.3 pair
#: the legacy code wrote inline.
OVERLAY_ALPHA = 0.3


def colorize(
    mask: np.ndarray,
    size: tuple[int, int] | None = None,
    palette: object = DEFAULT_LABEL_PALETTE,
) -> np.ndarray:
    """Class-id mask -> RGB image, using the canonical palette.

    Args:
        mask: ``(H, W)`` class ids, or ``(H, W, C)`` whose first channel holds them (the shape
            ``cv2`` label images and the legacy ``seg[:, :, 0]`` extraction both produce).
        size: Resize the mask to this ``(height, width)`` first, with **nearest** neighbour.
            Resizing the coloured image instead would smear every class boundary.
        palette: Palette override; defaults to :data:`~sbb_pytorch.gt.render.DEFAULT_LABEL_PALETTE`.
    """
    array = np.asarray(mask)
    if size is not None and tuple(array.shape[:2]) != (int(size[0]), int(size[1])):
        array = resize_mask(array, int(size[0]), int(size[1]))
    return label_map_to_image(array, palette)


def overlay(
    image: np.ndarray, top: np.ndarray, alpha: float = OVERLAY_ALPHA, *, top_is_mask: bool = False
) -> np.ndarray:
    """Blend ``top`` over the page ``image`` at ``alpha``, both in RGB, at the page's own size.

    Args:
        image: The page as RGB ``(h, w, 3)`` -- written back out at this resolution.
        top: The thing blended onto it.
        alpha: Weight of ``top``; the page keeps ``1 - alpha``.
        top_is_mask: ``top`` holds class ids, so it is upscaled by nearest-neighbour and
            colourised at the page size instead of being resampled as a photo.
    """
    image_uint8 = to_uint8(image)
    if top_is_mask:
        blended_source = colorize(top, size=(image_uint8.shape[0], image_uint8.shape[1]))
    else:
        blended_source = to_uint8(
            top
            if tuple(np.asarray(top).shape[:2]) == image_uint8.shape[:2]
            else resize(top, image_uint8.shape[0], image_uint8.shape[1])
        )
    return cv2.addWeighted(image_uint8, 1.0 - alpha, blended_source, alpha, 0.0)


def binarization_image(mask: np.ndarray) -> np.ndarray:
    """Render a binarisation prediction as an RGB page: ink black, paper white.

    Kept from the legacy ``prediction * -1 + 1`` -- class 1 *is* the ink layer, so inverting the
    mask is what makes the saved file read as a scanned page rather than its negative.
    """
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    binary = np.where(array > 0, 0, 255).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def enhancement_image(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Model output ``(3, H, W)`` or ``(H, W, 3)`` in ``[0, 1]`` -> ``uint8`` RGB page.

    ``to_uint8`` is what the legacy path skipped: it multiplied by 255 with ``astype`` and let
    values above 1 wrap, so an over-bright enhancement came out speckled.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[2] != 3:
        array = np.transpose(array, (1, 2, 0))  # (C, H, W) model output -> (H, W, C)
    array = to_uint8(array)
    if array.ndim == 2:  # pragma: no cover - a single-channel enhancement is a caller error
        array = np.repeat(array[:, :, None], 3, axis=2)
    if size is not None and tuple(array.shape[:2]) != (int(size[0]), int(size[1])):
        array = resize(array, int(size[0]), int(size[1]))
    return array


def render(
    task: TaskType,
    *,
    mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
    source: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The images one prediction can produce, keyed by the ``sbb infer`` flag that asks for them.

    ``overlay`` comes from ``--save`` and ``layout`` from ``--save-layout``; the binarization and
    enhancement tasks produce a single ``prediction`` image, which both flags write.

    Raises:
        ValueError: the task's required input is missing (a segmentation task with no ``mask``,
            or an overlay asked for without the ``source`` page it is blended onto).
    """
    task = TaskType(task)
    if task in (TaskType.SEGMENTATION, TaskType.BINARIZATION):
        if mask is None:
            raise ValueError(f"task {task.value} needs a mask to render")
        if task is TaskType.BINARIZATION:
            prediction = binarization_image(mask)
            return {"prediction": prediction, "overlay": prediction, "layout": prediction}
        images = {"layout": colorize(mask)}
        images["overlay"] = (
            overlay(source, mask, top_is_mask=True) if source is not None else images["layout"]
        )
        return images

    if task is TaskType.ENHANCEMENT:
        if image is None:
            raise ValueError("task enhancement needs the model's image output to render")
        size = (source.shape[0], source.shape[1]) if source is not None else None
        prediction = enhancement_image(image, size=size)
        return {"prediction": prediction, "overlay": prediction, "layout": prediction}

    if image is None and mask is None:
        raise ValueError(f"task {task.value} produced nothing to render")
    return {"prediction": image} if image is not None else {"layout": colorize(mask)}  # type: ignore[arg-type]


def save_images(images: dict[str, np.ndarray], **destinations: str | None) -> list[str]:
    """Write the named renderings to disk, skipping the ones with no destination.

    Used by ``sbb infer --save out.png --save-layout layout.png``, where either flag may be
    omitted. Returns the names actually written, so the CLI can report them.
    """
    written: list[str] = []
    for name, destination in destinations.items():
        if not destination:
            continue
        if name not in images:
            raise ValueError(f"no {name!r} image to save; this task produced {sorted(images)}")
        imwrite(destination, images[name])
        written.append(name)
    return written
