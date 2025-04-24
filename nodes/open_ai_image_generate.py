import requests
import io
import torch
from PIL import Image
import numpy as np
import base64
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiOpenAiImageGenerate(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "quality": (
                    [
                        "auto",
                        "low",
                        "medium",
                        "high",
                    ],
                    {"default": "low"},
                ),
                "size": (
                    [
                        "auto",
                        "1024x1024",
                        "1536x1024",
                        "1024x1536",
                    ],
                    {"default": "auto"},
                ),
                "width": ("INT", {"default": 0, "min": 0}),
                "height": ("INT", {"default": 0, "min": 0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "mAI"

    def _get_size(self, size, width, height):
        if size != "auto":
            return size

        if width <= 0 or height <= 0:
            return "auto"

        if width == height:
            return "1024x1024"

        if width > height:
            return "1536x1024"

        return "1024x1536"

    def generate_image(
        self,
        url,
        api_key,
        prompt,
        quality,
        size,
        width,
        height,
        seed,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        payload = {
            "prompt": prompt,
            "quality": quality,
            "size": self._get_size(size, width, height),
            "seed": seed,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result_json = response.json()

            if "data" not in result_json:
                raise ValueError("[ERROR] The API returned an invalid response format.")

            # Decode base64 image data
            base64_data = result_json["data"]
            image_data = base64.b64decode(base64_data)

            # Convert to PIL Image and ensure RGB
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Convert to tensor in ComfyUI format (B, H, W, C)
            image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)

            return (image_tensor,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[REQUEST ERROR] {e}")
        except base64.binascii.Error as e:
            raise RuntimeError(f"[BASE64 DECODE ERROR] {e}")
