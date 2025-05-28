from .nodes.llm_text import MaiLLMText
from .nodes.llm_reasoning import MaiLLMReasoning
from .nodes.llm_vision import MaiLLMVision
from .nodes.open_ai_image_edit import MaiOpenAiImageEdit
from .nodes.open_ai_image_generate import MaiOpenAiImageGenerate
from .nodes.google_image_generate import MaiGoogleImageGenerate

NODE_CLASS_MAPPINGS = {
    "MaiLLMText": MaiLLMText,
    "MaiLLMReasoning": MaiLLMReasoning,
    "MaiLLMVision": MaiLLMVision,
    "MaiOpenAiImageEdit": MaiOpenAiImageEdit,
    "MaiOpenAiImageGenerate": MaiOpenAiImageGenerate,
    "MaiGoogleImageGenerate": MaiGoogleImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaiLLMText": "mAI - LLM Text",
    "MaiLLMReasoning": "mAI - LLM Reasoning",
    "MaiLLMVision": "mAI - LLM Vision",
    "MaiOpenAiImageEdit": "mAI - OpenAI - Image Edit",
    "MaiOpenAiImageGenerate": "mAI - OpenAI - Image Generate",
    "MaiGoogleImageGenerate": "mAI - Google - Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
