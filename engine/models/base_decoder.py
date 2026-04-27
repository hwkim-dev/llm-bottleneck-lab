from engine.core.model_config import ModelConfig
from engine.core.runtime import RuntimeContext
from typing import Any, List, Optional, Dict

class BaseDecoderModel:
    """
    Base class for all decoder-only transformer models in the lab.
    """
    def __init__(self, config: ModelConfig, runtime: RuntimeContext):
        self.config = config
        self.runtime = runtime

    def architecture_name(self) -> str:
        """Return the architecture name."""
        return "base_decoder"

    def supported_precisions(self) -> List[str]:
        """Return a list of supported precisions."""
        return ["fp16", "int8", "int4"]

    def supported_backends(self) -> List[str]:
        """Return a list of supported backends."""
        return ["cpu", "vulkan", "npu_uca"]

    def load_weights(self, model_path: str) -> None:
        """Load weights from path. To be implemented by subclasses."""
        pass

    def generate(self, prompt_tokens: List[int], max_new_tokens: int) -> List[int]:
        """Generate tokens given a prompt."""
        return prompt_tokens + [0] * max_new_tokens

    def dry_run_summary(self) -> Dict[str, Any]:
        """Return a dictionary of the dry-run state."""
        return {
            "name": self.architecture_name(),
            "status": "adapter resolved successfully",
            "supported_backends": self.supported_backends(),
            "supported_precisions": self.supported_precisions()
        }
