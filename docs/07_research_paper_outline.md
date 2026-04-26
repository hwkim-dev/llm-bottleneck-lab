# Research Paper Outline

**Title:** Small LLMs on Low-Spec Hardware: A Bottleneck Study of Quantization, iGPU Offload, and Ternary Weights

## Abstract
Briefly summarize the problem (small LLMs are still slow on edge hardware), the methodology (`llm-lite` testbed), and key findings regarding memory bandwidth, iGPU overhead, and ternary weights.

## 1. Introduction
* Motivation: Pushing LLMs to edge/low-spec devices.
* The reality of small model sizes (1B-3B) versus available memory bandwidth.
* Contribution: A systematic bottleneck analysis framework (`llm-lite`).

## 2. Background
* Decoder-only transformer architecture and the decoding phase memory wall.
* Quantization techniques (FP16, INT8, INT4) and their dequantization costs.
* Hardware hierarchy: CPU memory vs. unified iGPU memory.
* BitNet and ternary weights (-1, 0, 1) as a compute optimization.

## 3. System Design (`llm-lite` architecture)
* Model registry and adapter pattern for isolation (Gemma, Llama, Qwen).
* Backend abstraction (CPU, Vulkan iGPU).
* Quantization paths and memory layout.

## 4. Methodology
* Target Hardware: Specifications of the low-end test devices (e.g., Ryzen 4500U, older Intel HD).
* Target Models: 1B to 3B class models.
* Precision Modes tested.
* Benchmark Metrics: prefill vs. decode speeds, peak memory.

## 5. Experiments
* CPU Baseline: Establishing the memory bandwidth limit.
* Vulkan iGPU Offload: Measuring API overhead vs. compute gain.
* Quantization Comparison: INT4 vs INT8 vs FP16.
* BitNet Ternary POC: Evaluating the addition-only compute advantage.

## 6. Results
* Data tables, performance graphs, and bottleneck identification.

## 7. Discussion
* Implications for future edge LLM design.

## 8. Limitations
* Specific hardware configurations tested, lack of massive scale deployment data.

## 9. Conclusion
* Summary of findings.