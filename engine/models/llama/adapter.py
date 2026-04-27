from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry
from typing import List

@ModelRegistry.register("llama")
class LlamaAdapter(BaseDecoderModel):
    def architecture_name(self) -> str:
        return "llama"

    def supported_precisions(self) -> List[str]:
        return ["fp16", "int8", "int4"]

    def supported_backends(self) -> List[str]:
        return ["cpu", "vulkan", "npu_uca"]

    def load_weights(self, model_path: str) -> None:
        if not self.runtime.dry_run:
            print(f"WIP: Loading Llama weights from {model_path} is not yet implemented.")

    def generate(self, prompt_tokens: List[int], max_new_tokens: int) -> List[int]:
        if not self.runtime.dry_run:
            print("WIP: Inference for Llama is not yet implemented.")
        return super().generate(prompt_tokens, max_new_tokens)
