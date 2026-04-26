from engine.models.base_decoder import BaseDecoderModel
from engine.core.registry import ModelRegistry

@ModelRegistry.register("bitnet")
class BitNetAdapter(BaseDecoderModel):
    def load_weights(self, model_path: str) -> None:
        print(f"Loading BitNet ternary weights from {model_path} (experimental)")
        pass
