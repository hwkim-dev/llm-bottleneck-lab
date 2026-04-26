# CPU vs Vulkan iGPU Bottleneck

A core research focus of `llm-lite`.

## CPU Execution

* **Pros:** Direct access to system memory, no API dispatch overhead, easier to optimize with SIMD (AVX2/NEON).
* **Cons:** Limited cores, lower theoretical compute throughput.

## Vulkan iGPU Execution

* **Pros:** Higher theoretical compute throughput, parallel shader execution.
* **Cons:** Unified memory means it shares the same bandwidth limits as the CPU. High overhead to dispatch shader commands.

## Hypothesis

For small models (1B-3B) with batch size 1 (decoding), the CPU will often outperform the iGPU because the memory bandwidth is saturated before the iGPU's compute capability can be fully utilized, and Vulkan overhead dominates the execution time. iGPU offload may only show benefits during the prefill phase.