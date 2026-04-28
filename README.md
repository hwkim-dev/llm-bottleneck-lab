# LLM Bottleneck Lab

[![Smoke Test](https://github.com/hwkim-dev/llm-bottleneck-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hwkim-dev/llm-bottleneck-lab/actions)
![License](https://img.shields.io/github/license/hwkim-dev/llm-bottleneck-lab)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-native%20kernels-00599C)
![WebGPU](https://img.shields.io/badge/WebGPU-visual%20lab-purple)

> **Visualize why LLM inference is slow.**  
> An interactive systems lab for analyzing LLM inference bottlenecks:
> **Prefill vs Decode**, **GEMM/GEMV**, **KV-cache growth**, **quantization**, **memory bandwidth**, and **low-spec hardware behavior**.

`llm-bottleneck-lab` is **not** a production inference server and **not** a `llama.cpp` replacement.  
It is a runnable research and visualization lab for understanding where LLM inference performance actually goes.

---

## Why this repo exists

Most Transformer explainers show **how attention works**.

This repo focuses on a different question:

> **Why does LLM inference become slow in practice?**

In real inference workloads, the bottleneck is often not just "the model is big."  
The slowdown can come from:

- decode-time KV-cache reads
- GEMV-like low-reuse operations
- memory bandwidth limits
- quantization trade-offs
- CPU/iGPU offload overhead
- thread scaling limits
- backend-specific kernel behavior

The goal is to make those bottlenecks **visible**, **measurable**, and **runnable**.

---

## Core idea

```text
Prompt tokens
   │
   ▼
Prefill ── high parallelism / GEMM-like work
   │
   ▼
KV-cache grows
   │
   ▼
Decode ── one token at a time / GEMV-like work
   │
   ▼
Memory bandwidth becomes the wall
```

---

## What this project will visualize

| Topic | What you should understand |
|---|---|
| **Prefill vs Decode** | Why prompt processing and token generation behave differently |
| **GEMM vs GEMV** | Why matrix-matrix work uses hardware better than matrix-vector work |
| **KV-cache growth** | Why longer context increases memory pressure |
| **Quantization** | How FP16, INT8, INT4, and ternary weights trade quality for memory |
| **Memory-bound decoding** | Why decode can be limited by bandwidth rather than compute |
| **Thread scaling** | Why adding more threads does not always increase tokens/sec |
| **CPU / Vulkan / FPGA paths** | How backend choices change the bottleneck profile |

---

## Current status

This repository currently contains a runnable research scaffold and legacy low-level experiments.

| Area | Status |
|---|---|
| CLI scaffold | Working dry-run path |
| Benchmark report generation | Working dry-run path |
| Tiny stub model | Available for platform checks |
| CPU backend | Reference/skeleton path |
| Vulkan backend | Skeleton / planned offload path |
| FPGA-style NPU path | Experimental |
| Gemma3N legacy path | Preserved under `x64/gemma3N_E4B/` |
| WebGPU visual lab | Planned |
| Local model GUI | Planned |
| Full production inference | Not the goal |

---

## Quick start

Use the tiny stub model to verify that the platform scaffolding works.

```bash
# Test adapter resolution
python run.py \
  --model examples/tiny_model_stub \
  --backend cpu \
  --precision fp16 \
  --dry-run
```

```bash
# Generate benchmark JSON/HTML output in dry-run mode
python benchmark.py \
  --model examples/tiny_model_stub \
  --backends cpu,vulkan \
  --precisions fp16,int4 \
  --dry-run
```

```bash
# Generate Markdown summary from benchmark results
python scripts/generate_report.py
```

> Dry-run mode does **not** load real model weights and does **not** represent real performance.  
> It verifies that adapters, backend selection, precision routing, and report generation work.

---

## Repository layout

```text
llm-bottleneck-lab/
├── engine/                     # Modular research engine scaffold
├── native/                     # Native / low-level implementation area
├── examples/
│   └── tiny_model_stub/         # Lightweight test model stub
├── scripts/                    # Report and utility scripts
├── results/                    # Generated benchmark/report artifacts
├── x64/
│   └── gemma3N_E4B/             # Preserved legacy Gemma3N path
├── benchmark.py                 # Benchmark/report entrypoint
├── run.py                       # Runtime entrypoint
├── launch.sh                    # CLI launcher
├── launch_gui.sh                # GUI launcher placeholder/legacy path
└── docs/                        # Design notes and research documentation
```

---

## Planned WebGPU visual lab

The next major direction is a browser-based interactive explainer.

The web demo should be lightweight:

- no huge GIFs
- no bundled model weights
- no server-side rendering requirement
- real-time rendering in the visitor's browser
- WebGPU first, WebGL2 fallback
- Canvas/SVG fallback for unsupported devices

Planned scenes:

```text
1. Prefill vs Decode split-screen animation
2. GEMM vs GEMV tensor-block animation
3. KV-cache memory growth simulator
4. Quantization error and memory-reduction visualizer
5. Memory-bound vs compute-bound roofline-style view
6. Thread scaling and cache contention visualizer
```

The README should eventually link to:

```text
https://hwkim-dev.github.io/llm-bottleneck-lab/
```

---

## Planned local model GUI

The GUI should eventually support local model experiments without turning this repo into a production inference server.

Target modes:

| Mode | Purpose |
|---|---|
| **Simulated mode** | Fake token stream for visualization and UI testing |
| **Local runtime mode** | Connect to a local inference runtime such as Ollama or llama.cpp server |
| **Browser experimental mode** | Future WebGPU/WebLLM-style in-browser experiments |

Target metrics:

- prompt length
- generated tokens
- time to first token
- tokens/sec
- estimated KV-cache memory
- selected backend
- selected precision
- decode timeline
- memory-pressure visualization

The GUI should make the bottleneck visible while generation is happening.

---

## Backends and precision targets

| Backend | Target | Status | Notes |
|---|---|---|---|
| `cpu` | x86 / ARM CPU | Runnable scaffold | Reference path and dry-run adapter |
| `vulkan` | iGPU / dGPU via Vulkan | Skeleton | Planned offload experiments |
| `npu_uca` | FPGA-style NPU | Experimental | Bare-metal research path |

| Precision | Status | Notes |
|---|---|---|
| `fp16` | Skeleton | Baseline 16-bit path |
| `int8` | Skeleton | 8-bit quantization experiments |
| `int4` | Skeleton | 4-bit quantization experiments |
| `ternary` | Experimental | BitNet-style `-1, 0, +1` weights |

---

## Model family targets

This project is model-family aware, but it is not restricted to one model.

| Family | Example | Status |
|---|---|---|
| Gemma / Gemma3N | `gemma-3n-e4b` | Legacy path preserved |
| Llama | `llama-3.2-1b` | Adapter scaffold |
| Qwen | `qwen2.5-1.5b` | Adapter scaffold |
| DeepSeek-Distill | `deepseek-r1-distill-qwen-1.5b` | Adapter scaffold |
| BitNet | `bitnet-b1.58-2b` | Experimental ternary path |

---

## What makes this different

| Project | Primary focus |
|---|---|
| `llama.cpp` | Production local inference |
| `vLLM` | High-throughput serving |
| `MLC LLM` | Universal deployment |
| `bitnet.cpp` | Official 1-bit / BitNet inference |
| **`llm-bottleneck-lab`** | Visual and runnable bottleneck analysis |

This repo is for people who ask:

> "Why is generation slow on my machine?"  
> "Where is the bottleneck: compute, memory, backend, or quantization?"  
> "Why is prefill different from decode?"  
> "Why does more hardware not always mean more tokens/sec?"

---

## Roadmap

### Phase 1 — Rebrand and cleanup

- [ ] Replace old `llm-lite` naming with `llm-bottleneck-lab`
- [ ] Update GitHub repo description
- [ ] Fix badge URLs
- [ ] Keep legacy paths clearly labeled
- [ ] Make README star-friendly and visually clear

### Phase 2 — Bottleneck experiments

- [ ] KV-cache memory estimator
- [ ] GEMM vs GEMV toy benchmark
- [ ] quantization error demo
- [ ] thread scaling experiment
- [ ] benchmark result viewer

### Phase 3 — Web visual lab

- [ ] Vite + React + TypeScript app
- [ ] WebGPU detection
- [ ] WebGL2 fallback
- [ ] low-power rendering mode
- [ ] interactive Prefill vs Decode scene
- [ ] KV-cache growth scene
- [ ] GEMM/GEMV scene
- [ ] quantization scene

### Phase 4 — Local model GUI

- [ ] simulated token streaming mode
- [ ] local runtime connection mode
- [ ] prompt input and streaming output
- [ ] live TTFT / tokens/sec / KV-cache estimate
- [ ] visual bottleneck timeline

### Phase 5 — Advanced systems path

- [ ] Vulkan experiments
- [ ] native SIMD kernels
- [ ] FPGA/NPU notes
- [ ] WebGPU compute toy kernels
- [ ] optional WASM microbenchmarks

---

## Design principles

1. **Explain first, optimize second.**
2. **Visualize bottlenecks, not just architecture.**
3. **Avoid huge assets. Use real-time rendering instead of heavy GIFs.**
4. **Keep every demo runnable on low-spec machines.**
5. **Clearly separate simulation, dry-run, and real inference.**
6. **Do not pretend this is a production inference engine.**
7. **Make the README understandable in under 60 seconds.**

---

## License

Apache-2.0

---

## Contributing

Contributions are welcome, especially in:

- bottleneck explanations
- small reproducible experiments
- visualization scenes
- benchmark scripts
- low-spec hardware results
- CPU/Vulkan/FPGA notes

If you add a benchmark, please include:

1. hardware information
2. backend
3. precision
4. sequence length
5. prompt length
6. generated token count
7. tokens/sec
8. notes about thermal throttling or power mode
