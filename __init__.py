from .nodes.llm_text import MaiLLMText

NODE_CLASS_MAPPINGS = {"MaiLLMText": MaiLLMText}

NODE_DISPLAY_NAME_MAPPINGS = {"MaiLLMText": "mAI - LLM Text"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
