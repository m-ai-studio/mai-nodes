import re
import io
import base64
import requests
import torch
import numpy as np
from PIL import Image
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiOpenAiImageEdit(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gpt-image-2", "multiline": False}),
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
                "refs": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "call_image_edit"
    CATEGORY = "mAI"

    def _resolve_size(self, size, width, height):
        if size != "auto":
            return size

        if width <= 0 or height <= 0:
            return "auto"

        if width == height:
            return "1024x1024"

        if width > height:
            return "1536x1024"

        return "1024x1536"

    def _split_prompt_items(self, prompt):
        items = [p.strip() for p in re.split(r"\n\s*\n+", prompt.strip())]
        return [p for p in items if p]

    def _build_edit_prompt_with_references(self, prompt_item, ref_count):
        if ref_count <= 0:
            return prompt_item
        noun = "reference image" if ref_count == 1 else "reference images"
        return f"{ref_count} {noun} are provided. {prompt_item}"

    def _tensor_frame_to_pil(self, frame):
        if frame.dim() == 3 and frame.shape[0] <= 4 and frame.shape[-1] > 4:
            frame = frame.permute(1, 2, 0)

        image_np = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if image_np.ndim == 2:
            return Image.fromarray(image_np, mode="L")

        modes = {1: "L", 3: "RGB", 4: "RGBA"}
        channels = image_np.shape[-1]

        if channels not in modes:
            raise ValueError(f"Unexpected channel count: {channels}")

        return Image.fromarray(
            image_np[..., 0] if channels == 1 else image_np, mode=modes[channels]
        )

    def _first_image_to_pil(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(tensor)}")
        if tensor.dim() == 4:
            tensor = tensor[0]
        return self._tensor_frame_to_pil(tensor)

    def _iter_batch_to_pil(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(tensor)}")
        if tensor.dim() == 3:
            yield self._tensor_frame_to_pil(tensor)
            return
        for i in range(tensor.shape[0]):
            yield self._tensor_frame_to_pil(tensor[i])

    def _pil_to_b64_png(self, img):
        buf = io.BytesIO()
        mode = "RGBA" if img.mode == "RGBA" else "RGB"
        img.convert(mode).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _base64_to_pil(self, b64_str):
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

    def _pil_to_comfy_image(self, pil_img):
        arr = np.array(pil_img.convert("RGB"))
        tensor = torch.from_numpy(arr).float().div(255.0)
        return tensor.unsqueeze(0)

    def _mask_to_openai_alpha_pil(self, mask_tensor, target_size):
        if not isinstance(mask_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(mask_tensor)}")

        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor[0]

        if mask_tensor.dim() == 3 and mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1]

        mask_np = (mask_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        rgba = np.full((*mask_np.shape, 4), 255, dtype=np.uint8)
        rgba[..., 3] = 255 - mask_np

        mask_pil = Image.fromarray(rgba, mode="RGBA")
        if mask_pil.size != target_size:
            mask_pil = mask_pil.resize(target_size, Image.NEAREST)
        return mask_pil

    def _stack_image_tensors(self, tensors):
        return torch.cat(tensors, dim=0)

    def _join_info_lines(self, lines):
        return "\n".join(lines)

    def call_image_edit(
        self,
        image,
        url,
        api_key,
        model,
        prompt,
        quality,
        size,
        width,
        height,
        seed,
        refs=None,
        mask=None,
    ):
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("[ERROR] No API key provided.")

        target_url = url.strip()
        if not target_url:
            raise ValueError("[ERROR] No URL provided.")

        pil_img = self._first_image_to_pil(image)
        refs_pil = list(self._iter_batch_to_pil(refs)) if refs is not None else []

        native_size = self._resolve_size(size, width, height)

        prompt_items = self._split_prompt_items(prompt)
        if not prompt_items:
            raise ValueError("[ERROR] No prompt provided.")

        mask_b64 = None
        if mask is not None:
            mask_pil = self._mask_to_openai_alpha_pil(mask, pil_img.size)
            mask_b64 = self._pil_to_b64_png(mask_pil)

        base_b64 = self._pil_to_b64_png(pil_img)
        ref_b64s = [self._pil_to_b64_png(r) for r in refs_pil]

        headers = {"x-api-key": api_key, "Content-Type": "application/json"}

        output_images = []
        info_lines = []

        for prompt_idx, prompt_item in enumerate(prompt_items, 1):
            final_prompt = self._build_edit_prompt_with_references(
                prompt_item, len(refs_pil)
            )

            images_b64 = [base_b64] + ref_b64s

            payload = {
                "model": model.strip(),
                "prompt": final_prompt,
                "size": native_size,
                "seed": seed,
                "image": images_b64 if len(images_b64) > 1 else images_b64[0],
            }
            if quality != "auto":
                payload["quality"] = quality
            if mask_b64 is not None:
                payload["mask"] = mask_b64

            try:
                response = requests.post(
                    target_url, headers=headers, json=payload, timeout=300
                )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"[REQUEST ERROR] {e}")

            if response.status_code in (401, 403):
                raise RuntimeError("[UNAUTHORIZED] api_key rejected by proxy.")
            response.raise_for_status()

            try:
                result = response.json()
            except ValueError as e:
                raise RuntimeError(f"[ERROR] Invalid JSON from proxy: {e}")

            data = result.get("data") if isinstance(result, dict) else None
            if not data:
                raise RuntimeError("[ERROR] Proxy returned no data.")

            entry = data[0]
            b64 = entry.get("b64_json")
            if not b64:
                raise RuntimeError("[ERROR] Proxy response missing b64_json.")
            revised = entry.get("revised_prompt")

            output_pil = self._base64_to_pil(b64)
            output_images.append(self._pil_to_comfy_image(output_pil))
            info_lines.append(
                f"[{prompt_idx}/{len(prompt_items)}] "
                + (revised or "Image edited.")
                + f" | Reference images used: {len(refs_pil)}"
                + f" | Output size requested: {native_size}"
            )

        images_out = self._stack_image_tensors(output_images)
        info_out = self._join_info_lines(info_lines)

        try:
            self.save_content(info_out, "MaiOpenAiImageEdit")
        except Exception as e:
            print(f"[MaiOpenAiImageEdit] save_content failed: {e}")

        return (images_out, info_out)
