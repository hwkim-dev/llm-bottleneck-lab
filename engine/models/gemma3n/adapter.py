from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry
from typing import List

@ModelRegistry.register("gemma3n")
class Gemma3NAdapter(BaseDecoderModel):
    def architecture_name(self) -> str:
        return "gemma3n"

    def supported_precisions(self) -> List[str]:
        return ["fp16", "int8", "int4"]

    def supported_backends(self) -> List[str]:
        return ["cpu", "vulkan"]

    def load_weights(self, model_path: str) -> None:
        if not self.runtime.dry_run:
            print(f"WIP: Loading Gemma3N weights from {model_path} is handled by the legacy path currently.")

    def generate(self, prompt_tokens: List[int], max_new_tokens: int) -> List[int]:
        if not self.runtime.dry_run:
            print("WIP: Inference for Gemma3N is handled by the legacy path currently.")
        return super().generate(prompt_tokens, max_new_tokens)
