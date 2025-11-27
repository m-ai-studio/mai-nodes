import torch
import torchvision.transforms.functional as F


class MaiImageContrast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call_image_contrast"
    CATEGORY = "mAI"

    def call_image_contrast(
        self,
        image: torch.Tensor,
        factor: float,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(factor, float)

        image = image.permute(0, 3, 1, 2)
        image = F.adjust_contrast(image, factor)
        image = image.permute(0, 2, 3, 1)

        return (image,)
