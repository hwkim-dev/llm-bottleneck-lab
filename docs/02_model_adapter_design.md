# Model Adapter Design

To support comparative research, `llm-lite` uses an adapter pattern. We do not implement a universal engine; instead, we provide specific adapters for targeted model families.

## Base Architecture

* `BaseDecoderModel`: Provides the common interface for decoder-only transformers.

## Supported Adapters

1. **Gemma 3N Adapter:** The legacy/reference implementation, optimized for the original E4B goals.
2. **Llama Adapter:** Skeleton for Llama 3.2 1B/3B models.
3. **Qwen Adapter:** Skeleton for Qwen 2.5 0.5B/1.5B/3B.
4. **DeepSeek-Distill Adapter:** Extends the Qwen adapter, specifically targeting `DeepSeek-R1-Distill-Qwen-1.5B`.
5. **BitNet Adapter:** A distinct experimental path for handling ternary weights.

This design allows us to isolate architectural differences and measure their impact on specific backends independently.