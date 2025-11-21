import torch
from PIL import Image
import numpy as np


def to_pil(tensor):
    """
    Convert a ComfyUI tensor to a PIL Image.

    Args:
        tensor: A torch.Tensor representing an image in ComfyUI format

    Returns:
        PIL.Image: A PIL Image object

    Raises:
        TypeError: If tensor is not a torch.Tensor
        ValueError: If tensor has an unexpected channel count
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(tensor)}")

    if tensor.dim() == 4:
        tensor = tensor[0]

    if tensor.shape[0] <= 4:
        tensor = tensor.permute(1, 2, 0)

    image_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    modes = {1: "L", 3: "RGB", 4: "RGBA"}
    channels = image_np.shape[2]

    if channels not in modes:
        raise ValueError(f"Unexpected channel count: {channels}")

    return Image.fromarray(
        image_np[..., 0] if channels == 1 else image_np, mode=modes[channels]
    )
