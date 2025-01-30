from .nodes.llm_text import MaiLLMText
from .nodes.llm_reasoning import MaiLLMReasoning
from .nodes.llm_vision import MaiLLMVision

NODE_CLASS_MAPPINGS = {
    "MaiLLMText": MaiLLMText,
    "MaiLLMReasoning": MaiLLMReasoning,
    "MaiLLMVision": MaiLLMVision,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaiLLMText": "mAI - LLM Text",
    "MaiLLMReasoning": "mAI - LLM Reasoning",
    "MaiLLMVision": "mAI - LLM Vision",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
