# llm-lite

**llm-lite** is a lightweight, multi-backend LLM inference engine designed to run Large Language Models (like Gemma) efficiently on low-spec environments. 
Instead of pruning the model architecture, it leverages quantization and low-level hardware acceleration to maximize inference performance.

## Core Features
* **Multi-Backend Architecture**: 
  * `x64` (amd64): Local PC inference accelerated by C++, SIMD instructions, and Vulkan API.
  * `NPU(uXC)`: Custom FPGA backend utilizing proprietary bare-metal APIs(uXC u(micro) eXcelerator Core).
* **Native & Zero-Bloat**: Built entirely without heavy framework dependencies to minimize memory and CPU overhead.
* **Aggressive Quantization**: Maintains full model architecture while reducing memory footprints to fit strictly constrained environments.
