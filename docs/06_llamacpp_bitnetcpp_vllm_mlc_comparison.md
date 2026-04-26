# Comparison with Other LLM Engines

This document highlights the differences between `llm-lite` and other prominent open-source inference projects. *Note: `llm-lite` is designed as a research complement, not a production competitor.*

| Feature | `llama.cpp` | `bitnet.cpp` | `vLLM` | `MLC LLM` | `llm-lite` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Goal** | Broad, production-ready inference | Official 1-bit/1.58-bit engine | High-throughput server | Universal deployment | **Low-spec bottleneck lab** |
| **Target Users** | General users, developers | Researchers, BitNet early adopters | Cloud providers, large deployments | App developers, multi-platform | **Systems researchers** |
| **Hardware Scope** | Everything (CPU, GPU, Mac, etc.) | CPU | High-end Datacenter GPUs | Mobile, Web, PC | **Older laptops, APUs, NPU/FPGA** |
| **Model Scope** | Massive ecosystem | BitNet specifically | Large foundation models | Broad | **Small models (Gemma, Llama, Qwen)** |
| **Backend Scope** | Highly optimized universal | Custom ternary kernels | CUDA, ROCm | TVM/compiler based | **Experimental Vulkan / C++ / uCA** |
| **Research Value** | Standard baseline | Ternary optimization | PagedAttention, Serving | Compilation | **Bottleneck comparative analysis** |
| **Production Ready** | Yes | Evolving | Yes | Yes | **No (Research Prototype)** |
| **Differentiator** | Robustness | 1-bit focus | Throughput | Portability | **Controlled experimental matrix** |