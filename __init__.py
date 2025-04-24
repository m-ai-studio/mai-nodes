from .nodes.llm_text import MaiLLMText
from .nodes.llm_reasoning import MaiLLMReasoning
from .nodes.llm_vision import MaiLLMVision
from .nodes.image_edit import MaiImageEdit
from .nodes.open_ai_image_generate import MaiOpenAiImageGenerate

NODE_CLASS_MAPPINGS = {
    "MaiLLMText": MaiLLMText,
    "MaiLLMReasoning": MaiLLMReasoning,
    "MaiLLMVision": MaiLLMVision,
    "MaiImageEdit": MaiImageEdit,
    "MaiOpenAiImageGenerate": MaiOpenAiImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaiLLMText": "mAI - LLM Text",
    "MaiLLMReasoning": "mAI - LLM Reasoning",
    "MaiLLMVision": "mAI - LLM Vision",
    "MaiImageEdit": "mAI - Image Edit",
    "MaiOpenAiImageGenerate": "mAI - OpenAI - Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
