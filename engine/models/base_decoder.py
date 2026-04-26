from engine.core.model_config import ModelConfig
from engine.core.runtime import RuntimeContext
from typing import Any, List, Optional

class BaseDecoderModel:
    """
    Base class for all decoder-only transformer models in the lab.
    """
    def __init__(self, config: ModelConfig, runtime: RuntimeContext):
        self.config = config
        self.runtime = runtime

    def load_weights(self, model_path: str) -> None:
        """Load weights from path. To be implemented by subclasses."""
        pass

    def generate(self, prompt_tokens: List[int], max_new_tokens: int) -> List[int]:
        """Generate tokens given a prompt."""
        # Simple stub returning the prompt and dummy generated tokens
        return prompt_tokens + [0] * max_new_tokens
