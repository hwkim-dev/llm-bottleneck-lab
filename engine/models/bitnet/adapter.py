from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry
from typing import List

@ModelRegistry.register("bitnet")
class BitNetAdapter(BaseDecoderModel):
    def architecture_name(self) -> str:
        return "bitnet"

    def supported_precisions(self) -> List[str]:
        return ["ternary"]

    def supported_backends(self) -> List[str]:
        return ["cpu"]

    def load_weights(self, model_path: str) -> None:
        if not self.runtime.dry_run:
            print(f"WIP: Loading BitNet ternary weights from {model_path} is experimental and not yet implemented.")

    def generate(self, prompt_tokens: List[int], max_new_tokens: int) -> List[int]:
        if not self.runtime.dry_run:
            print("WIP: Inference for BitNet is not yet implemented.")
        return super().generate(prompt_tokens, max_new_tokens)
