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
    load_time_sec: Optional[float]
    prefill_time_sec: Optional[float]
    decode_time_sec: Optional[float]
    tokens_per_sec: Optional[float]
    peak_memory_mb: Optional[float]
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

            load_time = f"{self.load_time_sec:.4f}" if self.load_time_sec is not None else "not measured"
            prefill_time = f"{self.prefill_time_sec:.4f}" if self.prefill_time_sec is not None else "not measured"
            decode_time = f"{self.decode_time_sec:.4f}" if self.decode_time_sec is not None else "not measured"
            tokens_per_sec = f"{self.tokens_per_sec:.2f}" if self.tokens_per_sec is not None else "not measured"
            peak_memory_mb = f"{self.peak_memory_mb:.2f}" if self.peak_memory_mb is not None else "not measured"

            f.write(f"- Load Time (s): {load_time}\n")
            f.write(f"- Prefill Time (s): {prefill_time}\n")
            f.write(f"- Decode Time (s): {decode_time}\n")
            f.write(f"- Tokens per Second: {tokens_per_sec}\n")
            f.write(f"- Peak Memory (MB): {peak_memory_mb}\n")
            if self.notes:
                f.write(f"- Notes: {self.notes}\n")
