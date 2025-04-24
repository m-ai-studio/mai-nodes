import requests
import io
import torch
from PIL import Image
import numpy as np
import base64
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiOpenAiImageEdit(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
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
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call_image_edit"
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

    def _mask_to_pil(self, mask_tensor):
        if not isinstance(mask_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(mask_tensor)}")

        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor[0]

        # Convert to single channel if needed
        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1]

        # Convert to numpy and scale to 0-255
        mask_np = (mask_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Remove channel dimension if present
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        # Create white image with inverted mask as alpha channel
        white = np.full((*mask_np.shape, 4), 255, dtype=np.uint8)
        white[..., 3] = 255 - mask_np  # Invert mask values for alpha channel

        return Image.fromarray(white, mode="RGBA")

    def call_image_edit(
        self,
        image1,
        url,
        api_key,
        prompt,
        quality,
        size,
        width,
        height,
        seed,
        image2=None,
        image3=None,
        image4=None,
        mask=None,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        # Convert the incoming ComfyUI tensors to valid PIL images
        pil_images = []
        for img in [image1, image2, image3, image4]:
            if img is not None:
                pil_images.append(self._to_pil(img))

        if not pil_images:
            raise ValueError("[ERROR] At least one image must be provided.")

        # Prepare image files
        files = {}
        for i, pil_img in enumerate(pil_images, 1):
            image_bytes = io.BytesIO()
            pil_img.save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            files[f"image{i}"] = (f"image{i}.jpg", image_bytes, "image/jpeg")

        # Add mask if provided
        if mask is not None:
            mask_pil = self._mask_to_pil(mask)
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format="PNG")
            mask_bytes.seek(0)
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        # Prepare request
        headers = {"x-api-key": api_key.strip()}
        data = {
            "prompt": prompt,
            "quality": quality,
            "size": self._get_size(size, width, height),
            "seed": seed,
        }

        try:
            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=180
            )
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
