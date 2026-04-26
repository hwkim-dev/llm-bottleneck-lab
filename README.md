<div align="center">

# llm-lite — Low-Spec LLM Systems Lab

**Why small LLMs are still hard on low-spec hardware.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Lab-purple.svg)](#)

*llm-lite is a low-spec LLM systems lab for studying quantization, CPU/Vulkan/iGPU bottlenecks, BitNet ternary weights, and small-model inference across Gemma, Llama, Qwen, and DeepSeek-Distill.*

**Note: This is not a `llama.cpp` replacement. It is a research scaffold.**

</div>

---

## 1. Why this exists

Even small LLMs (1B-3B parameters) demand significant memory bandwidth. On high-end GPUs, this is easily met. However, on low-spec hardware (older laptops, embedded devices, low-power APUs), several bottlenecks emerge:

*   **CPU Memory Bandwidth:** The primary bottleneck. Loading weights for decoding saturates DDR4/LPDDR4 memory bandwidth.
*   **iGPU Offload Overhead:** Offloading compute to integrated GPUs (via Vulkan) introduces dispatch overhead that can outweigh the compute gains for small workloads.
*   **Quantization Trade-offs:** INT4 saves bandwidth, but the CPU/iGPU must decompress weights. If the hardware lacks fast integer pipelines, decompression becomes the new bottleneck.
*   **Ternary Weights (BitNet):** A theoretical advantage (addition-only math), but requires specific packing and handling to realize gains on consumer edge devices.

## 2. What makes this different

*   **`llama.cpp`**: Aimed at broad, production-ready inference across all hardware.
*   **`bitnet.cpp`**: The official engine for 1-bit/1.58-bit inference.
*   **`vLLM`**: Designed for high-throughput cloud serving.
*   **`MLC LLM`**: Focused on universal deployment across platforms.
*   **`llm-lite`**: We are a **low-spec bottleneck research lab**. We do not aim to replace these engines; rather, we provide a scaffold to analyze memory constraints, quantization effects, and hardware limits via comparative benchmarking.

## 3. Supported Model Matrix

| Model Family | Example Model | Config Parsing | CPU Ref | Quantization | Vulkan | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 3N** | Gemma 3N E4B | Yes | Yes | INT4 | Yes | Tier 0 (Legacy/Ref) |
| **Llama** | Llama 3.2 1B/3B | Skeleton | Skeleton | Skeleton | Skeleton | Tier 1 |
| **Qwen** | Qwen 2.5 1.5B | Skeleton | Skeleton | Skeleton | Skeleton | Tier 1 |
| **DeepSeek** | R1-Distill-Qwen-1.5B | Skeleton | Skeleton | Skeleton | Skeleton | Tier 1 (via Qwen) |
| **BitNet** | b1.58 2B | Skeleton | Skeleton | Ternary | Skeleton | Tier 2 (Experimental) |

## 4. Backend Matrix

| Backend | Target | Status | Notes |
| :--- | :--- | :--- | :--- |
| **CPU** | x64 AVX2/NEON | Skeleton | Baseline memory bandwidth evaluation. |
| **Vulkan** | Integrated GPUs | Skeleton | iGPU offload overhead and compute study. |
| **NPU/uCA** | FPGA/Edge Accelerators | Skeleton | Proprietary bare-metal API path. |

## 5. Quantization Matrix

| Precision | Target Models | Status | Notes |
| :--- | :--- | :--- | :--- |
| **FP16** | All Baselines | Skeleton | Baseline reference point. Memory intensive. |
| **INT8** | Llama, Qwen | Skeleton | Balances memory and accuracy. |
| **INT4** | Gemma3N, Llama, Qwen | Skeleton | Primary target for low-spec. High dequantization cost. |
| **Ternary** | BitNet b1.58 | Skeleton | Experimental (-1, 0, 1). Requires distinct packing & compute. |

## 6. One-Command Smoke Test

Run a dry-run inference on the model adapter skeletons:

```bash
python run.py --model examples/tiny_model_stub --backend cpu --precision fp16 --dry-run
python run.py --model examples/tiny_model_stub --backend vulkan --precision int4 --dry-run
```

## 7. Benchmark Example

Run the benchmark skeleton to evaluate different backends and precisions:

```bash
python benchmark.py --model examples/tiny_model_stub --backends cpu,vulkan --precisions fp16,int4 --dry-run
```

Results are generated in `results/benchmarks/`. Use the generator script to create a summary table:

```bash
python scripts/generate_report.py
```

*Note: Data generated with `--dry-run` is synthetic ("to be measured") and serves as a structural placeholder.*

## 8. Research Direction

This repository is designed to facilitate research that can lead to technical reports or papers.

**Potential Paper Title:**
*"Small LLMs on Low-Spec Hardware: A Bottleneck Study of Quantization, iGPU Offload, and Ternary Weights"*

See [`docs/00_project_thesis.md`](docs/00_project_thesis.md) and [`docs/07_research_paper_outline.md`](docs/07_research_paper_outline.md) for more details.

## 9. Roadmap

*   **Phase 1:** Architecture Registry (Adapter skeletons) - *Completed*
*   **Phase 2:** Quantization Matrix (INT4, Ternary stubs) - *Completed*
*   **Phase 3:** CPU Benchmark capabilities
*   **Phase 4:** Vulkan Offload Study
*   **Phase 5:** BitNet Ternary Study
*   **Phase 6:** Paper-style Report generation

See [`docs/roadmap.md`](docs/roadmap.md) for full progress.

---
*The legacy Gemma3N E4B implementation remains in `x64/gemma3N_E4B/` as a reference.*