from server import PromptServer


class PromptSaverMixin:
    def save_content(self, content, node_class_name):
        """
        Save generated content to the prompt queue's extra_data.

        Args:
            content: The content to save
            node_class_name: The class name of the node to match in the queue
        """
        try:
            prompt_queue = PromptServer.instance.prompt_queue
            for prompt_data in prompt_queue.currently_running.values():
                nodes = prompt_data[2]
                extra_data = prompt_data[3]

                # Find this node in the nodes dictionary
                for _, node_info in nodes.items():
                    if node_info.get("class_type") == node_class_name:
                        # Add the generated text to extra_data
                        if "generated_texts" not in extra_data:
                            extra_data["generated_texts"] = []
                        extra_data["generated_texts"].append(content)
                        return
        except Exception as e:
            print(f"Failed to save content: {e}")
