import requests
import io
import torch
from PIL import Image
import numpy as np
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiLLMVision(PromptSaverMixin):
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
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "step": 1, "display": "number"},
                ),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "call_llm_vision"
    CATEGORY = "mAI"

    def call_llm_vision(
        self,
        image,
        url,
        api_key,
        user_prompt,
        temperature,
        top_p,
        max_tokens,
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
            "temperature": str(temperature),
            "top_p": str(top_p),
            "max_tokens": str(max_tokens),
            "seed": str(seed),
        }
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

        try:
            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=180
            )
            response.raise_for_status()
            result_json = response.json()
            llm_text = result_json.get("data", "")

            if not llm_text.strip():
                raise ValueError("[ERROR] The LLM returned an empty response.")

            self.save_content(llm_text, "MaiLLMVision")
            return (llm_text,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[REQUEST ERROR] {e}")

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
