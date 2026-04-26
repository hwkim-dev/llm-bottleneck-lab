import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class BenchmarkMetrics:
    model_name: str
    architecture: str
    backend: str
    precision: str
    prompt_tokens: int
    generated_tokens: int
    load_time_sec: float
    prefill_time_sec: float
    decode_time_sec: float
    tokens_per_sec: float
    peak_memory_mb: float
    notes: Optional[str] = None

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    def save_markdown(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(f"# Benchmark Report: {self.model_name}\n\n")
            f.write(f"- Architecture: {self.architecture}\n")
            f.write(f"- Backend: {self.backend}\n")
            f.write(f"- Precision: {self.precision}\n")
            f.write(f"- Prompt Tokens: {self.prompt_tokens}\n")
            f.write(f"- Generated Tokens: {self.generated_tokens}\n")
            f.write(f"- Load Time (s): {self.load_time_sec:.4f}\n")
            f.write(f"- Prefill Time (s): {self.prefill_time_sec:.4f}\n")
            f.write(f"- Decode Time (s): {self.decode_time_sec:.4f}\n")
            f.write(f"- Tokens per Second: {self.tokens_per_sec:.2f}\n")
            f.write(f"- Peak Memory (MB): {self.peak_memory_mb:.2f}\n")
            if self.notes:
                f.write(f"- Notes: {self.notes}\n")
