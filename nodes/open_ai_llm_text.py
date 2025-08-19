import requests
import json
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiOpenAiLLMText(PromptSaverMixin):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    "STRING",
                    {"default": "gpt-5", "multiline": False},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
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
                "text_verbosity": ("STRING", {"default": "low", "multiline": False}),
                "reasoning_effort": ("STRING", {"default": "low", "multiline": False}),
                "reasoning_summary": (
                    "STRING",
                    {"default": "auto", "multiline": False},
                ),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    FUNCTION = "call_llm"
    CATEGORY = "mAI"

    def call_llm(
        self,
        url,
        api_key,
        model,
        system_prompt,
        user_prompt,
        temperature,
        top_p,
        text_verbosity,
        reasoning_effort,
        reasoning_summary,
        seed,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        payload = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
            "top_p": top_p,
        }

        if text_verbosity.strip():
            payload["text"] = {"verbosity": text_verbosity}

        if reasoning_effort.strip() or reasoning_summary.strip():
            payload["reasoning"] = {}
            if reasoning_effort.strip():
                payload["reasoning"]["effort"] = reasoning_effort
            if reasoning_summary.strip():
                payload["reasoning"]["summary"] = reasoning_summary

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            llm_text = data.get("data", "")
            reasoning = data.get("reasoning", "")

            if not llm_text.strip():
                raise ValueError("[ERROR] The LLM returned an empty response.")

            self.save_content(llm_text, "MaiOpenAiLLMText-text")
            self.save_content(reasoning, "MaiOpenAiLLMText-reasoning")
            return (llm_text, reasoning)
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
