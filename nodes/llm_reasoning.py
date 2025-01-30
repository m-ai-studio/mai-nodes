import requests
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiLLMReasoning(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
    FUNCTION = "call_llm"
    CATEGORY = "mAI"

    def call_llm(
        self,
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

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        payload = {
            "user_prompt": user_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            llm_text = data.get("data", "")

            if not llm_text.strip():
                raise ValueError("[ERROR] The LLM returned an empty response.")

            self.save_content(llm_text, "MaiLLMReasoning")
            return (llm_text,)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[REQUEST ERROR] {e}")
