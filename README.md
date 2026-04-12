# LLM-Lite (Gemma 3N E4B Inference Engine)

**llm-lite** is a lightweight, multi-backend LLM inference engine designed to run Large Language Models (specifically optimized for Gemma 3N E4B) efficiently in strictly constrained and low-spec environments.

Instead of pruning the model architecture, it leverages aggressive quantization and low-level hardware acceleration to maximize inference performance. For local environments, it combines C++ and Vulkan to accelerate the model on integrated graphics (iGPU) and CPUs. It minimizes RAM usage by loading INT4 pre-compressed weights via MMAP, while executing custom matrix operations in FP16/FP32.

## Core Features

* **Multi-Backend Architecture:**
    * **x64 (amd64):** Local PC inference accelerated by C++, SIMD instructions, and the Vulkan API.
    * **NPU (uCA):** Custom FPGA backend utilizing proprietary bare-metal APIs (uCA: u(micro) Compute Architecture).
* **Native & Zero-Bloat:** Built entirely without heavy framework dependencies to minimize memory and CPU overhead.
* **Aggressive Quantization:** Maintains the full model architecture while drastically reducing memory footprints to fit strictly constrained environments.

## Project Architecture

* `main.py`: Main module for model loading, memory profiling, and inference pipeline management.
* `CPU_CORE.py`: C++ fused kernel operations (including KV Cache optimization, RoPE, and GQA) and Python bindings.
* `IGPU_CORE.py`: Vulkan shader-based VRAM data transfer and acceleration management.
* `safeTensor.py`: Virtual RAM mapping for model weights (Zero-copy loading).
* `Gemma3N_Dev_Docs.md`: Comprehensive technical documentation covering development history, architectural specifications, and tensor profiling.

## System Requirements (x64 Backend)

* **OS:** Linux (Ubuntu, etc.)
* **Compiler:** GCC/G++ (Requires `__fp16` and OpenMP support)
* **Graphics:** Vulkan compatible driver

## 1. Prerequisites

First, install the essential system packages required for compiling C++ kernels and Vulkan shaders.

```bash
sudo apt update
sudo apt install build-essential libvulkan-dev glslang-tools


Next, install the required Python packages. Using a virtual environment (e.g., pynq_env) is highly recommended.

```bash
pip install -r requirements.txt
```

## 2. Building Engines
You need to build the optimized C++ shared libraries (.so) that Python will call. The compilation is tailored for the supported hardware (Configured Target: Ryzen 5 4500U znver2).

```bash
bash build.sh
```

Note: Upon successful compilation, my_accelerator.so and vulkan_core.so will be generated in the C_DLL/ directory.

## 3. Running the Inference
Once the build is complete, execute main.py to launch the core engine and enter chatbot mode.

```bash
python3 main.py
```

## Configuration Modes
When launching, the following menu will be displayed for Feature Map Mode (activation precision):
1. FP32 — Full precision (baseline, recommended)
2. BF16 — BFloat16 (half bandwidth)
3. INT8 — 8-bit quantized
4. INT4 — 4-bit quantized (aggressive)
To ensure outlier calibration, output quality, and optimal execution speed, Option 1 (FP32) is currently recommended as the default for activations. The model weights themselves are always executed in an INT4 (MMAP) environment, which drastically reduces RAM usage regardless of the chosen activation precision.
