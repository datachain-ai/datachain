from collections.abc import Callable

import torch
from PIL import Image as PILImage

from datachain.lib.file import File, FileError, Image, ImageFile


def image_info(file: File | ImageFile) -> Image:
    """
    Returns image file information.

    Args:
        file (ImageFile): Image file object.

    Returns:
        Image: Image file information.
    """
    try:
        img = file.as_image_file().read()
    except Exception as exc:
        raise FileError("unable to open image file", file.source, file.path) from exc

    return Image(
        width=img.width,
        height=img.height,
        format=img.format or "",
    )


def convert_image(
    img: PILImage.Image,
    mode: str = "RGB",
    size: tuple[int, int] | None = None,
    transform: Callable | None = None,
    encoder: Callable | None = None,
    device: str | torch.device | None = None,
) -> PILImage.Image | torch.Tensor:
    """
    Resize, transform, and otherwise convert an image.

    Args:
        img (Image): PIL.Image object.
        mode (str): PIL.Image mode.
        size (tuple[int, int]): Size in (width, height) pixels for resizing.
        transform (Callable): Torchvision transform or huggingface processor to apply.
        encoder (Callable): Encode image using model.
        device (str or torch.device): Device to use.
    """
    if mode:
        img = img.convert(mode)
    if size:
        img = img.resize(size)
    if transform:
        img = transform(img)

        try:
            from transformers.image_processing_utils import BaseImageProcessor

            if isinstance(transform, BaseImageProcessor):
                img = torch.as_tensor(img.pixel_values[0]).clone().detach()  # type: ignore[assignment,attr-defined]
        except ImportError:
            pass
        if device:
            img = img.to(device)  # type: ignore[attr-defined]
        if encoder:
            img = img.unsqueeze(0)  # type: ignore[attr-defined]
    if encoder:
        img = encoder(img)
    return img


def convert_images(
    images: PILImage.Image | list[PILImage.Image],
    mode: str = "RGB",
    size: tuple[int, int] | None = None,
    transform: Callable | None = None,
    encoder: Callable | None = None,
    device: str | torch.device | None = None,
) -> list[PILImage.Image] | torch.Tensor:
    """
    Resize, transform, and otherwise convert one or more images.

    Args:
        images (Image, list[Image]): PIL.Image object or list of objects.
        mode (str): PIL.Image mode.
        size (tuple[int, int]): Size in (width, height) pixels for resizing.
        transform (Callable): Torchvision transform or huggingface processor to apply.
        encoder (Callable): Encode image using model.
        device (str or torch.device): Device to use.
    """
    if isinstance(images, PILImage.Image):
        images = [images]

    converted = [
        convert_image(img, mode, size, transform, device=device) for img in images
    ]

    if isinstance(converted[0], torch.Tensor):
        converted = torch.stack(converted)  # type: ignore[assignment,arg-type]

    if encoder:
        converted = encoder(converted)

    return converted  # type: ignore[return-value]
