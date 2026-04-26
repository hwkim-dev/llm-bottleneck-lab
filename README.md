<div align="center">

# llm-lite — Gemma 3N E4B Lightweight Inference Engine

**Low-spec local LLM inference — no cloud, no bloat.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Target](https://img.shields.io/badge/Target-Ryzen_4500U_·_KV260-red)](https://www.amd.com/)
[![Backends](https://img.shields.io/badge/Backends-Vulkan_·_SIMD_·_uCA-purple)](#core-features)
[![Quantization](https://img.shields.io/badge/Quantization-INT4_·_INT8_·_FP16_·_FP32-green)](#quantization-modes)
[![Blog](https://img.shields.io/badge/Blog_Tag-llm--lite-4f46e5?style=for-the-badge)](https://hwkim-dev.github.io/hwkim-dev/blog/tags/llm-lite)

### [📖 Full Documentation & Blog Posts →](https://hwkim-dev.github.io/hwkim-dev/blog/tags/llm-lite)

<a href="https://hwkim-dev.github.io/hwkim-dev/blog/tags/llm-lite">
  <img src="https://img.shields.io/badge/Read_the_Docs-hwkim--dev.github.io-0f172a?style=for-the-badge&logo=readthedocs&logoColor=white&labelColor=4f46e5" alt="Read the full docs on hwkim-dev.github.io" height="48" />
</a>

</div>

---

## Overview

**llm-lite** is a lightweight, multi-backend LLM inference engine designed to run Large Language Models (specifically optimized for **Gemma 3N E4B**) efficiently in strictly constrained and low-spec environments.

Instead of pruning the model architecture, it leverages aggressive quantization and low-level hardware acceleration to maximize inference performance. For local environments, it combines C++ and Vulkan to accelerate the model on integrated graphics (iGPU) and CPUs. It minimizes RAM usage by loading pre-compressed weights via MMAP while executing custom matrix operations in the selected activation precision.

---

## Core Features

* **Multi-Backend Architecture:**
    * **x64 (amd64):** Local PC inference accelerated by C++, SIMD instructions, and the Vulkan API.
    * **NPU (uCA):** Custom FPGA backend utilizing proprietary bare-metal APIs (uCA: u(micro) Compute Architecture).
* **Native & Zero-Bloat:** Built entirely without heavy framework dependencies to minimize memory and CPU overhead.
* **Aggressive Quantization:** Maintains the full model architecture while drastically reducing memory footprints to fit strictly constrained environments.
* **Two-track frontend:** **Web GUI** (Flask + browser, full feature set) and **CLI** (`python3 main.py`, for headless / scripting / KV260-class edge devices). The Dear ImGui native GUI is deprecated — see [`native/DEPRECATED.md`](native/DEPRECATED.md).

## Quantization Modes

| Mode | Description | Recommended on |
|---|---|---|
| `INT4` | Packed 4-bit weights + per-row FP32 scale. **Vulkan-accelerated**. Default. | Modern iGPU / dGPU |
| `INT8` | 8-bit quantized weights + per-row scale (no packing). CPU matmul. | Where INT4 accuracy insufficient |
| `FP16` | Raw half-precision weights. CPU matmul. | Older iGPUs (e.g., Vega 6) where FP16 beats INT |
| `FP32` | Full precision baseline — debugging / accuracy reference. | Large-RAM systems |

> **⚠ Note:** On older integrated GPUs (Renoir Vega 6/7, Intel HD/UHD ≤ Gen 9), floating-point paths can outperform integer quantization due to missing fast INT pipelines. The GUI auto-detects this and surfaces a warning.

## Project Architecture

* `main.py`: Main module for model loading, memory profiling, and inference pipeline management.
* `CPU_CORE.py`: C++ fused kernel operations (including KV Cache optimization, RoPE, and GQA) and Python bindings.
* `IGPU_CORE.py`: Vulkan shader-based VRAM data transfer and acceleration management.
* `safeTensor.py`: Virtual RAM mapping for model weights (Zero-copy loading).
* `quantize.py`: CLI converter — outputs INT4/INT8/FP16/FP32 weight directories.
* `spec_decode.py`: Speculative Decoding scaffold (MatFormer-based E4B→E2B draft; WIP).
* `gui_app.py`: Flask HTTP/SSE server backing the web GUI.
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
```

Next, install the required Python packages. Using a virtual environment (e.g., `pynq_env`) is highly recommended.

```bash
pip install -r requirements.txt
```

## 2. Building Engines
You need to build the optimized C++ shared libraries (`.so`) that Python will call. The compilation is tailored for the supported hardware (Configured Target: Ryzen 5 4500U znver2).

```bash
cd x64/gemma3N_E4B
bash build.sh
```

Note: Upon successful compilation, `my_accelerator.so` and `vulkan_core.so` will be generated in the `C_DLL/` directory.

## 3. Testing (Offline Smoke Test)
If you want to verify that the environment setup and C++ compilation was successful without needing gated Hugging Face weights, you can run the minimal offline smoke test:

```bash
cd x64/gemma3N_E4B
source pynq_env/bin/activate
python3 smoke_test.py
```

## 4. Running the Inference

Two front-ends share the same inference engine. Pick whichever fits the host.

### Web GUI — full feature set (recommended)

```bash
bash run.sh
# → http://127.0.0.1:5000
#   Model manager, stop-generation, KV slider, GPU warning, dark theme.
```

### CLI — headless / scripting / edge

```bash
cd x64/gemma3N_E4B
source pynq_env/bin/activate
python3 main.py
#   Interactive menu for weight / feature modes, then a chat prompt.
#   Ctrl-C interrupts generation mid-stream.
```

The legacy Dear ImGui native GUI (`native/`) is no longer maintained; see
`native/DEPRECATED.md`.

## Configuration Modes
When launching `main.py`, the following menus are displayed:

**Weight Mode (model weights precision):**
1. INT4 — 4-bit packed, Vulkan-accelerated (default, lowest RAM)
2. INT8 — 8-bit symmetric, CPU matmul
3. FP16 — half precision, CPU matmul
4. FP32 — full precision baseline

**Feature Map Mode (activation precision):**
1. FP32 — Full precision (baseline, recommended)
2. BF16 — BFloat16 (half bandwidth)
3. INT8 — 8-bit quantized
4. INT4 — 4-bit quantized (aggressive)

Activations in FP32 generally deliver the best accuracy/speed tradeoff for outlier sensitivity.

## Preparing Model Weights

The GUI exposes a model manager where you can download from HuggingFace and pick the quantization mode. Alternatively, from the CLI:

```bash
# Convert an already-downloaded HF snapshot into the chosen format
python quantize.py --mode int4 --src /path/to/hf/gemma-3n-E4B-it --dst models/gemma-3n-e4b-int4
python quantize.py --mode fp16 --src /path/to/hf/gemma-3n-E4B-it --dst models/gemma-3n-e4b-fp16
```

Each output directory contains a `manifest.json` describing the variant, which the loader uses to pick the right code path at runtime.

## Documentation

| Document | Link |
|---|---|
| Gemma 3N Reference Manual | [`docs/Gemma3N_Reference_Manual.md`](docs/Gemma3N_Reference_Manual.md) |
| Development notes | [`x64/gemma3N_E4B/Gemma3N_Dev_Docs.md`](x64/gemma3N_E4B/Gemma3N_Dev_Docs.md) |
| Speculative Decoding research | [`docs/Speculative_Decoding_Research.md`](docs/Speculative_Decoding_Research.md) |
| Blog (hwkim-dev.github.io) | [**llm-lite tag →**](https://hwkim-dev.github.io/hwkim-dev/blog/tags/llm-lite) |

<div align="center">

### Want deeper context?

<a href="https://hwkim-dev.github.io/hwkim-dev/blog/tags/llm-lite">
  <img src="https://img.shields.io/badge/📖_Visit_the_Blog_Tag_Page-4f46e5?style=for-the-badge&labelColor=0f172a" alt="Visit the llm-lite blog tag page" height="56" />
</a>

</div>
