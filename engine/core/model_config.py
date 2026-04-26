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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            model_type=data.get('model_type', 'unknown'),
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

    def to_dict(self) -> Dict[str, Any]:
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
