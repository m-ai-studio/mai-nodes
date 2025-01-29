import requests
import io
import torch
from PIL import Image
import numpy as np
import base64
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiImageEdit(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call_image_edit"
    CATEGORY = "mAI"

    def call_image_edit(
        self,
        image,
        url,
        api_key,
        user_prompt,
        seed,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        # Convert the incoming ComfyUI tensor to a valid PIL image
        pil_image = self._to_pil(image)

        # Save as JPEG in memory
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        # Prepare request
        headers = {"x-api-key": api_key.strip()}
        data = {
            "user_prompt": user_prompt,
            "seed": seed,
        }
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

        try:
            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=30
            )
            response.raise_for_status()
            result_json = response.json()
            response_data = result_json.get("data", {})

            if not response_data or "data" not in response_data:
                raise ValueError("[ERROR] The API returned an invalid response format.")

            # Decode base64 image data
            base64_data = response_data["data"]
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

    def _to_pil(self, tensor):
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
