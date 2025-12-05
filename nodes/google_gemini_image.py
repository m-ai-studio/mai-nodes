import requests
import io
import torch
from PIL import Image
import numpy as np
import base64
import json
from ..helpers.prompt_helpers import PromptSaverMixin
from ..helpers.image_helpers import to_pil


class MaiGoogleGeminiImage(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {"default": "auto"},
                ),
                "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "display": "number",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "number",
                    },
                ),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call_gemini"
    CATEGORY = "mAI"

    def call_gemini(
        self,
        url,
        api_key,
        system_prompt,
        user_prompt,
        aspect_ratio,
        image_size,
        temperature,
        top_p,
        seed,
        image=None,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        user_parts = [{"text": user_prompt}]
        if image is not None:
            if image.dim() == 4:
                # Batch of images
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i]
                    pil_image = to_pil(single_image)
                    image_bytes = io.BytesIO()
                    pil_image.save(image_bytes, format="JPEG")
                    image_bytes.seek(0)
                    image_base64 = base64.b64encode(image_bytes.getvalue()).decode(
                        "utf-8"
                    )
                    user_parts.append(
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}
                    )
            else:
                # Single image
                pil_image = to_pil(image)
                image_bytes = io.BytesIO()
                pil_image.save(image_bytes, format="JPEG")
                image_bytes.seek(0)
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
                user_parts.append(
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}
                )

        image_config = {
            "imageSize": image_size,
        }

        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        payload = {
            "model": "gemini-3-pro-image-preview",
            "config": {
                "temperature": temperature,
                "topP": top_p,
                "responseModalities": ["IMAGE"],
                "imageConfig": image_config,
                "tools": [{"googleSearch": {}}],
                "systemInstruction": [{"text": system_prompt}],
            },
            "contents": [{"role": "user", "parts": user_parts}],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()

            pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)

            return (image_tensor,)
        except requests.exceptions.RequestException as e:
            error_message = f"[REQUEST ERROR] {e}"

            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    if "error" in error_data and "message" in error_data["error"]:
                        error_message = f"[API ERROR] {error_data['error']['message']}"
                    elif "message" in error_data:
                        error_message = f"[API ERROR] {error_data['message']}"
                except:
                    pass

            raise RuntimeError(error_message)
