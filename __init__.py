from .nodes.llm_text import MaiLLMText
from .nodes.llm_reasoning import MaiLLMReasoning

NODE_CLASS_MAPPINGS = {"MaiLLMText": MaiLLMText, "MaiLLMReasoning": MaiLLMReasoning}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaiLLMText": "mAI - LLM Text",
    "MaiLLMReasoning": "mAI - LLM Reasoning",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
