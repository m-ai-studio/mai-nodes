import requests
import io
import torch
from PIL import Image
import numpy as np
import base64
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiGoogleImageGenerate(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "regular",
                        "ultra",
                    ],
                    {"default": "regular"},
                ),
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "16:9",
                        "4:3",
                        "3:4",
                        "9:16",
                    ],
                    {"default": "auto"},
                ),
                "width": ("INT", {"default": 0, "min": 0}),
                "height": ("INT", {"default": 0, "min": 0}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 100, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "mAI"

    def _get_aspect_ratio(self, aspect_ratio, width, height):
        if aspect_ratio != "auto":
            return aspect_ratio

        if width <= 0 or height <= 0:
            return "1:1"

        ratio = width / height
        if abs(ratio - 1.0) < 0.1:  # Close to 1:1
            return "1:1"
        elif abs(ratio - 16 / 9) < 0.1:  # Close to 16:9
            return "16:9"
        elif abs(ratio - 4 / 3) < 0.1:  # Close to 4:3
            return "4:3"
        elif abs(ratio - 3 / 4) < 0.1:  # Close to 3:4
            return "3:4"
        elif abs(ratio - 9 / 16) < 0.1:  # Close to 9:16
            return "9:16"
        else:
            # Default to 1:1 if no close match
            return "1:1"

    def generate_image(
        self,
        url,
        api_key,
        prompt,
        model,
        aspect_ratio,
        width,
        height,
        enhance_prompt,
        seed,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        payload = {
            "prompt": prompt,
            "model": model,
            "aspectRatio": self._get_aspect_ratio(aspect_ratio, width, height),
            "enhancePrompt": enhance_prompt,
            "seed": seed,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()

            pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)

            return (image_tensor,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[REQUEST ERROR] {e}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] {str(e)}")
