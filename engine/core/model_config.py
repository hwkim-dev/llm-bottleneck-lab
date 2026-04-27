import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class ModelConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    torch_dtype: str
    sliding_window: Optional[int] = None
    architectures: List[str] = field(default_factory=list)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, filepath: str) -> 'ModelConfig':
        """Load ModelConfig from a config.json file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Load ModelConfig from a dictionary."""
        return cls(
            model_type=cls.detect_architecture(data),
            hidden_size=data.get('hidden_size', 0),
            num_hidden_layers=data.get('num_hidden_layers', 0),
            num_attention_heads=data.get('num_attention_heads', 0),
            num_key_value_heads=data.get('num_key_value_heads', 0),
            intermediate_size=data.get('intermediate_size', 0),
            vocab_size=data.get('vocab_size', 0),
            rope_theta=data.get('rope_theta', 10000.0),
            rms_norm_eps=data.get('rms_norm_eps', 1e-6),
            torch_dtype=data.get('torch_dtype', 'float16'),
            sliding_window=data.get('sliding_window'),
            architectures=data.get('architectures', []),
            raw_config=data
        )

    @staticmethod
    def detect_architecture(data: Dict[str, Any]) -> str:
        """Normalize model_type based on hints in the configuration."""
        model_type = data.get('model_type', 'unknown').lower()
        architectures = [a.lower() for a in data.get('architectures', [])]

        if 'llama' in model_type or any('llama' in a for a in architectures):
            return 'llama'
        if 'qwen' in model_type or any('qwen' in a for a in architectures):
            # We treat DeepSeek R1 Distill Qwen as 'qwen' compatible
            return 'qwen'
        if 'deepseek' in model_type or any('deepseek' in a for a in architectures):
            return 'deepseek'
        if 'gemma' in model_type or any('gemma' in a for a in architectures):
            return 'gemma3n' # Targeting gemma3n in our lab context
        if 'bitnet' in model_type or any('bitnet' in a for a in architectures):
            return 'bitnet'

        return model_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to simple dict summary."""
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "torch_dtype": self.torch_dtype,
            "sliding_window": self.sliding_window,
            "architectures": self.architectures
        }

    def summary(self) -> Dict[str, Any]:
        return self.to_dict()
