from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry

@ModelRegistry.register("gemma3n")
class Gemma3NAdapter(BaseDecoderModel):
    def load_weights(self, model_path: str) -> None:
        print(f"Loading Gemma3N weights from {model_path} (legacy/reference path)")
        pass
