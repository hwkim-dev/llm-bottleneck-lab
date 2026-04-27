from engine.models.qwen.adapter import QwenAdapter
from engine.core.registry import ModelRegistry
from typing import List

@ModelRegistry.register("deepseek-distill")
class DeepSeekDistillAdapter(QwenAdapter):
    """
    DeepSeek-R1-Distill uses Qwen architecture.
    """
    def architecture_name(self) -> str:
        return "deepseek-distill"

    def supported_precisions(self) -> List[str]:
        return ["fp16", "int8", "int4"]

    def supported_backends(self) -> List[str]:
        return ["cpu", "vulkan", "npu_uca"]

    def load_weights(self, model_path: str) -> None:
        if not self.runtime.dry_run:
            print(f"WIP: Loading DeepSeek Distill weights from {model_path} via Qwen path is not yet implemented.")
