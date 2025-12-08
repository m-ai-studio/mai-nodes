import requests
import io
import base64
import json
from ..helpers.prompt_helpers import PromptSaverMixin
from ..helpers.image_helpers import to_pil


class MaiGoogleGeminiText(PromptSaverMixin):
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
                "thinking_level": (["LOW", "HIGH"], {"default": "LOW"}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "model")
    FUNCTION = "call_gemini"
    CATEGORY = "mAI"

    def call_gemini(
        self,
        url,
        api_key,
        system_prompt,
        user_prompt,
        temperature,
        top_p,
        thinking_level,
        seed,
        image=None,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        user_parts = [{"text": user_prompt}]

        if image is not None:
            pil_image = to_pil(image)
            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            user_parts.append(
                {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}
            )

        payload = {
            "model": "gemini-3-pro-preview",
            "config": {
                "temperature": temperature,
                "topP": top_p,
                "thinkingConfig": {"thinkingLevel": thinking_level},
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                ],
                "tools": [{"googleSearch": {}}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
            },
            "contents": [{"role": "user", "parts": user_parts}],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            llm_text = data.get("data", "")
            model_name = data.get("model", "")

            if not llm_text.strip():
                raise ValueError("[ERROR] The LLM returned an empty response.")

            self.save_content(llm_text, "MaiGoogleGeminiText-text")
            return (llm_text, model_name)
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
