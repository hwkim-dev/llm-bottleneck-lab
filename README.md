# LLM Bottleneck Lab

> **Measure and visualize why LLM inference is slow.**  
> A research-oriented lab for **bottleneck analysis**, **model dissection**, **KV-cache growth**, **GEMM/GEMV behavior**, **quantization**, and **memory-bound decoding**.

<p align="center">
  <a href="https://github.com/hwkim-dev/llm-bottleneck-lab/actions">
    <img alt="CI" src="https://github.com/hwkim-dev/llm-bottleneck-lab/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="./LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  </a>
  <img alt="Status" src="https://img.shields.io/badge/status-research--lab-orange">
  <img alt="Focus" src="https://img.shields.io/badge/focus-LLM%20bottlenecks-purple">
</p>

<p align="center">
  <b>Not another chat UI. Not another llama.cpp clone.</b><br/>
  This repo is a lab for finding where LLM inference time and memory actually go.
</p>

---

## Why this repo exists

Most Transformer explainers show **how attention works**.

This repo asks a different question:

> **When an LLM runs locally, what actually becomes the bottleneck?**

For small and local LLMs, raw model size is only one part of the story. Real inference behavior is shaped by:

- **prefill vs decode** latency
- **GEMM vs GEMV** execution patterns
- **KV-cache** memory growth
- **memory bandwidth** limits
- **thread scaling** overhead
- **quantization** trade-offs
- backend behavior across **CPU, iGPU/Vulkan, WebGPU, and experimental NPU paths**

LLM Bottleneck Lab turns those into measurable experiments, visual reports, and reproducible artifacts.

---

## Core idea

```text
LLM Bottleneck Lab
= automatic bottleneck measurement
+ surgical model-specific dissection
+ quantized runnable inference experiments
+ WebGPU/WebGL visual reports
+ release-ready local profiling tools
```

### Two analysis modes

| Mode | Question it answers | Scope |
|---|---|---|
| **Auto Measurement** | “Where does this model seem slow?” | Broad, model-agnostic profiling |
| **Surgical Dissection** | “Which internal operation is causing it?” | Deep, model-specific analysis |

### 1. Auto Bottleneck Measurement

Use this when you want to profile a model without manually dissecting its internals.

Planned/target metrics:

- TTFT, total latency, decode tokens/sec
- prefill vs decode timing
- memory usage and KV-cache estimate
- thread scaling behavior
- precision impact: FP16 / INT8 / INT4 / ternary
- backend impact: CPU / Vulkan / WebGPU / NPU-style paths

### 2. Surgical Bottleneck Dissection

Use this when a model family has a dedicated dissector.

This mode can expose:

- layer-by-layer latency
- attention vs MLP cost
- GEMM/GEMV behavior
- KV-cache read/write cost
- quantization error
- operator-level bottlenecks

The first deep-dive target is the existing **Gemma 3N E4B** path, but the lab is designed to expand to other model families.

---

## What makes this different

| Project type | Main goal | This repo |
|---|---|---|
| `llama.cpp`-style engines | Run models fast | Explain and measure bottlenecks |
| Chat UIs | Talk to a model | Visualize inference behavior |
| Benchmark tables | Report numbers | Connect numbers to internal causes |
| Transformer explainers | Explain architecture | Explain runtime bottlenecks |
| Model repos | Provide weights | Provide reproducible analysis tools |

This project is intentionally positioned as a **systems lab**, not a production serving engine.

---

## Visual report direction

The GitHub Pages site will be a report-style visual lab, not a static documentation dump.

Planned visual reports:

- **Prefill vs Decode** — why generating one token at a time changes the bottleneck
- **GEMM vs GEMV** — why decode can underuse compute
- **KV-cache Growth** — how context length becomes memory pressure
- **Quantization Trade-offs** — why smaller weights do not automatically mean faster inference
- **Memory Wall View** — when bandwidth dominates arithmetic
- **Thread Scaling** — why more CPU threads can stop helping

The goal is to make bottlenecks visible before looking at the code.

---

## Current status

This repository is being refactored from the original `llm-lite` scaffold into **LLM Bottleneck Lab**.

Current working pieces include:

- a modular research scaffold
- CLI entry points for dry-run execution and benchmarking
- benchmark report generation
- model adapter structure
- quantization/backends skeletons
- preserved Gemma 3N E4B legacy research path

> **Note**  
> Some paths are intentionally experimental. This repo prioritizes measurement, reproducibility, and analysis over production-grade inference.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/hwkim-dev/llm-bottleneck-lab.git
cd llm-bottleneck-lab
```

### 2. Run the smoke test

The tiny stub model verifies that the platform wiring works without downloading real model weights.

```bash
python run.py \
  --model examples/tiny_model_stub \
  --backend cpu \
  --precision fp16 \
  --dry-run
```

### 3. Generate a dry-run benchmark report

```bash
python benchmark.py \
  --model examples/tiny_model_stub \
  --backends cpu,vulkan \
  --precisions fp16,int4 \
  --dry-run
```

### 4. Generate a Markdown summary

```bash
python scripts/generate_report.py
```

Dry-run mode is **not** a real performance benchmark. It is a platform-health check.

---

## Supported model families

| Model family | Current role | Status |
|---|---|---|
| Gemma 3N | First surgical dissection target | Legacy path preserved |
| Llama | General adapter target | Skeleton |
| Qwen | General adapter target | Skeleton |
| DeepSeek-Distill | General adapter target | Skeleton |
| BitNet-style models | Ternary quantization target | Experimental |

The long-term goal is not to support every model as a chat engine.  
The goal is to make each supported model useful for **bottleneck analysis**.

---

## Backends and precision paths

### Backends

| Backend | Target | Status |
|---|---|---|
| `cpu` | x86 / ARM reference path | Runnable scaffold |
| `vulkan` | iGPU / dGPU offload | Skeleton |
| `webgpu` | browser visual lab and future compute experiments | Planned |
| `npu_uca` | FPGA/NPU-style research path | Experimental |

### Precision

| Precision | Purpose | Status |
|---|---|---|
| `fp16` | baseline comparison | Skeleton |
| `int8` | practical quantized path | Skeleton |
| `int4` | memory-pressure experiments | Skeleton |
| `ternary` | BitNet-style research | Experimental |

Quantized inference is treated as a first-class experiment path because bottlenecks change when weights become smaller.

---

## Repository map

```text
llm-bottleneck-lab/
├── engine/                  # modular runtime, backend, quantization scaffold
├── examples/                # tiny stub model and test fixtures
├── scripts/                 # report generation utilities
├── results/                 # generated reports and benchmark artifacts
├── docs/                    # concepts, measurement modes, report notes
├── x64/gemma3N_E4B/         # preserved Gemma 3N E4B deep-dive path
├── run.py                   # runnable entry point
├── benchmark.py             # benchmark/report entry point
└── README.md
```

Planned additions:

```text
web/                         # WebGPU/WebGL visual report site
profiler/                    # automatic and surgical profilers
models/adapters/             # model-family adapters
reports/                     # reproducible report templates
release/                     # packaged local measurement tools
```

---

## Target CLI shape

The release program is planned around a small local profiler CLI:

```bash
llm-bottleneck run       # run quantized inference experiment
llm-bottleneck analyze   # automatic bottleneck measurement
llm-bottleneck dissect   # model-specific internal analysis
llm-bottleneck report    # generate GitHub Pages / Markdown report
```

Example target workflow:

```bash
llm-bottleneck analyze \
  --model ./models/qwen2.5-1.5b \
  --backend cpu \
  --precision int4 \
  --prompt "Explain KV-cache briefly."
```

Example target output:

```text
Likely bottleneck: memory bandwidth
TTFT: 812 ms
Decode speed: 14.2 tok/s
KV-cache estimate: 420 MB
Precision: INT4
Backend: CPU
```

---

## Design principles

1. **Measure first.**  
   Every visual claim should eventually connect to a metric, script, or report.

2. **Do not hide the bottleneck.**  
   The goal is to expose memory movement, KV-cache growth, and operator cost.

3. **Stay reproducible.**  
   Reports should include commands, configs, hardware notes, and raw outputs.

4. **Avoid huge assets.**  
   No large model weights, large GIFs, or heavy videos in git.

5. **Keep it static-hosting friendly.**  
   The report site should work on GitHub Pages or Cloudflare Pages.

6. **Be honest about scope.**  
   This is a research lab, not a production inference engine.

---

## Roadmap

### Phase 0 — Rebrand and clarify

- [ ] Rename all visible `llm-lite` references
- [ ] Update badges and repo description
- [ ] Position repo as a bottleneck analysis lab
- [ ] Separate current scaffold from future targets

### Phase 1 — Measurement baseline

- [ ] Stabilize dry-run benchmark schema
- [ ] Add JSON result format
- [ ] Add TTFT / decode / memory estimate fields
- [ ] Add repeatable hardware metadata capture
- [ ] Add report generator improvements

### Phase 2 — Quantized runnable experiments

- [ ] FP16 baseline path
- [ ] INT8 experiment path
- [ ] INT4 experiment path
- [ ] Quantization error report
- [ ] Compare precision vs memory pressure

### Phase 3 — Surgical dissection

- [ ] Formalize model-family dissector interface
- [ ] Promote Gemma 3N E4B as the first deep-dive specimen
- [ ] Add layer/operator timing schema
- [ ] Add attention/MLP/KV-cache breakdown reports

### Phase 4 — Visual report site

- [ ] Add WebGPU/WebGL report site
- [ ] Add Prefill vs Decode visualization
- [ ] Add KV-cache growth visualization
- [ ] Add GEMM vs GEMV visualization
- [ ] Add quantization visualization
- [ ] Publish GitHub Pages report

### Phase 5 — Release profiler

- [ ] Package local CLI
- [ ] Add Linux x64 release artifact
- [ ] Add reproducible benchmark templates
- [ ] Add example reports for multiple model families

---

## Contributing

Contributions are welcome if they help answer one of these questions:

- Where does inference time go?
- Where does memory go?
- Which operation is the bottleneck?
- How does quantization change the bottleneck?
- How does backend choice change the bottleneck?
- How can the result be visualized clearly?

Good first contribution areas:

- benchmark result schemas
- model adapter skeletons
- KV-cache formulas
- quantization experiments
- WebGPU/WebGL visual scenes
- report templates
- documentation diagrams

---

## Citation / research use

If you use this project for a report, paper, or class project, please cite the repository and include:

- commit hash
- model family
- backend
- precision
- hardware
- benchmark command
- generated report

Reproducibility matters more than a single token/sec number.

---

## License

MIT License. See [`LICENSE`](./LICENSE).
