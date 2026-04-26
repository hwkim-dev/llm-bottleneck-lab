# Quantization Matrix

| Precision | Target Models | Status | Notes |
| :--- | :--- | :--- | :--- |
| **FP16** | All Baselines | Skeleton | The baseline reference point. Memory intensive. |
| **INT8** | Llama, Qwen | Skeleton | Balances memory and accuracy. Requires fast int8 compute. |
| **INT4** | Gemma3N, Llama, Qwen | Skeleton | Primary target for low-spec. High dequantization cost. |
| **Ternary** | BitNet b1.58 | Skeleton | Experimental (-1, 0, 1). Requires distinct packing & compute. |