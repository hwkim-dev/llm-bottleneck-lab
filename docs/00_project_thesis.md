# Project Thesis: Small LLMs on Low-Spec Hardware

## Core Research Questions

1. **CPU vs iGPU Bottleneck:** In small LLM inference (e.g. 1B-3B parameters), what is the actual bottleneck between low-spec CPUs and integrated GPUs? Is it memory bandwidth, compute capability, or API overhead?
2. **Quantization Trade-offs:** Is INT4 always faster than FP16 or INT8? How does the dequantization overhead impact performance on hardware lacking native INT4 pipelines?
3. **Vulkan Offloading:** Under what specific conditions does offloading to a Vulkan iGPU provide a measurable advantage over pure CPU execution?
4. **Ternary Weights (BitNet):** What are the real-world advantages and limitations of BitNet-style ternary weights (-1, 0, +1) on low-spec hardware?
5. **Architectural Impact:** How do architectural differences across models (Gemma, Llama, Qwen, DeepSeek-Distill) influence backend bottlenecks?

## Why This Project Exists

Running small language models on high-end hardware is a solved problem. However, deploying these models on constrained edge devices, older laptops, and APUs remains challenging. This project serves as an experimental inference lab to study *why* these deployments struggle and *how* different optimizations behave under strict limitations.

## How We Differ from Existing Engines

* **llama.cpp:** Aimed at broad, production-ready inference across all hardware.
* **bitnet.cpp:** The official engine for 1-bit/1.58-bit inference.
* **vLLM:** Designed for high-throughput cloud serving.
* **MLC LLM:** Focused on universal deployment across platforms.
* **llm-lite:** We are a **research-oriented bottleneck lab**. We do not aim to replace these engines; rather, we provide a scaffold to analyze memory constraints, quantization effects, and hardware limits.

## Core Hypothesis

We hypothesize that for small LLMs on low-spec hardware, memory bandwidth is the primary bottleneck, and aggressive quantization (like INT4 or ternary) is necessary. However, the overhead of unpacking these quantized formats can negate the bandwidth savings if the hardware lacks efficient integer pipelines.

## Expected Contributions

* A comparative analysis of CPU vs Vulkan bottlenecks on low-spec systems.
* An evaluation of different quantization strategies across modern small LLM architectures.
* A proof-of-concept adapter for BitNet ternary weights to study their efficacy.

## Path to Publication

The findings from this lab are intended to form the basis of a technical report or research paper, exploring the limits of local LLM inference on consumer edge devices.