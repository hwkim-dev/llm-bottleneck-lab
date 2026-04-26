from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry

@ModelRegistry.register("llama")
class LlamaAdapter(BaseDecoderModel):
    def load_weights(self, model_path: str) -> None:
        print(f"Loading Llama weights from {model_path} (adapter skeleton)")
        pass
