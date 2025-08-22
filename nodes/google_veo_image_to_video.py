import requests
import io
import torch
from PIL import Image
import numpy as np
import json
from ..helpers.prompt_helpers import PromptSaverMixin
from comfy_api.input_impl.video_types import VideoFromFile

# List of supported params: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation


class MaiGoogleVeoImageToVideo(PromptSaverMixin):
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
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": ("STRING", {"default": "720p", "multiline": False}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("STRING", "VIDEO", "IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("video_url", "video", "frames", "audio", "fps")
    FUNCTION = "call_veo"
    CATEGORY = "mAI"

    def call_veo(
        self,
        image,
        url,
        api_key,
        user_prompt,
        negative_prompt,
        resolution,
        enhance_prompt,
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
            "prompt": user_prompt,
            "params": json.dumps(
                {
                    "negative_prompt": negative_prompt,
                    "resolution": resolution,
                    "enhance_prompt": enhance_prompt,
                }
            ),
        }
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

        try:
            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=400
            )
            response.raise_for_status()
            result_json = response.json()
            video_url = result_json.get("url", "")

            if not video_url.strip():
                raise ValueError("[ERROR] Empty response.")

            # Download the video and create a proper VideoFromFile object
            video_response = requests.get(video_url, timeout=400)
            video_response.raise_for_status()

            video_bytes = io.BytesIO(video_response.content)
            video_bytes.seek(0)

            # Create a proper video object that ComfyUI can handle
            video_obj = VideoFromFile(video_bytes)

            # Extract video components for VideoHelperSuite compatibility
            try:
                components = video_obj.get_components()
                frames = components.images
                audio = components.audio
                fps = float(components.frame_rate)
            except Exception as e:
                # Fallback if component extraction fails
                print(f"Warning: Could not extract video components: {e}")
                frames = torch.zeros((1, 3, 512, 512))  # Single black frame
                audio = None
                fps = 30.0

            self.save_content(video_url, "MaiGoogleVeoImageToVideo")
            return (video_url, video_obj, frames, audio, fps)

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
