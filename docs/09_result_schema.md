# Benchmark Result Schema

The `benchmark.py` script generates JSON summaries for each test run.

## Schema
```json
{
  "model_name": "tiny_model_stub",
  "architecture": "llama",
  "backend": "cpu",
  "precision": "fp16",
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "load_time_sec": null,
  "prefill_time_sec": null,
  "decode_time_sec": null,
  "tokens_per_sec": null,
  "peak_memory_mb": null,
  "notes": "dry-run; not measured"
}
```

## Description
*   `load_time_sec`, `prefill_time_sec`, `decode_time_sec`: Time elapsed in seconds.
*   `tokens_per_sec`: Model output speed.
*   `peak_memory_mb`: Maximum memory required.
*   **Dry-run vs Measured:** When using `--dry-run`, measured fields will be `null` or 0, and notes will mention that it was an unmeasured dry run.
