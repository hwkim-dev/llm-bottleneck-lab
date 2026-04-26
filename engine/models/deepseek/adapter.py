from engine.models.qwen.adapter import QwenAdapter
from engine.core.registry import ModelRegistry

@ModelRegistry.register("deepseek-distill")
class DeepSeekDistillAdapter(QwenAdapter):
    """
    DeepSeek-R1-Distill uses Qwen architecture.
    """
    def load_weights(self, model_path: str) -> None:
        print(f"Loading DeepSeek Distill weights from {model_path} via Qwen path")
        pass
