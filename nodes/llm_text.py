import requests
from ..helpers.prompt_helpers import PromptSaverMixin


class MaiLLMText(PromptSaverMixin):
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
                "provider": (["samba_nova", "groq"], {"default": "groq"}),
                "model": (
                    "STRING",
                    {"default": "openai/gpt-oss-120b", "multiline": False},
                ),
                "timeout_ms": ("INT", {"default": 20000, "min": 1, "max": 999999}),
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
                    {
                        "default": 1024,
                        "step": 1,
                        "display": "number",
                        "min": 1,
                        "max": 999999,
                    },
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
        system_prompt,
        user_prompt,
        provider,
        model,
        timeout_ms,
        temperature,
        top_p,
        max_tokens,
        seed,
    ):
        if not url.strip():
            raise ValueError("[ERROR] No URL provided.")

        headers = {"Content-Type": "application/json", "x-api-key": api_key.strip()}

        payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "provider": provider,
            "model": model,
            "timeout_ms": timeout_ms,
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
            timed_out = data.get("timedOut", "")

            if not llm_text.strip():
                raise ValueError("[ERROR] The LLM returned an empty response.")

            if timed_out:
                print(
                    "\033[33m⚠️ [WARNING] mAI LLM Text: "
                    + provider
                    + " | "
                    + model
                    + " - request timed out -> OpenAI fallback was used.\033[0m"
                )

            self.save_content(llm_text, "MaiLLMText")
            return (llm_text,)
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
