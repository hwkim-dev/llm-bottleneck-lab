# LLM Bottleneck Lab

> **Measure and visualize why LLM inference is slow.**
>
> A research-oriented lab for **automatic bottleneck measurement**, **model-specific dissection**, **quantized runnable inference**, and **WebGPU-powered visual reports**.

<p align="center">
  <a href="https://github.com/hwkim-dev/llm-bottleneck-lab/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/hwkim-dev/llm-bottleneck-lab/ci.yml?branch=main&label=ci"></a>
  <a href="https://github.com/hwkim-dev/llm-bottleneck-lab/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/hwkim-dev/llm-bottleneck-lab"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="WebGPU" src="https://img.shields.io/badge/WebGPU-visual%20lab-purple">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20lab-orange">
</p>

<p align="center">
  <b>Auto profile any model.</b> Dissect supported architectures. Run quantized experiments. Publish reproducible bottleneck reports.
</p>

---

## What is this?

`llm-bottleneck-lab` is **not another production inference engine** and it is **not a `llama.cpp` replacement**.

It is a lab for answering one practical systems question:

> **Where do the time and memory go during LLM inference?**

Most Transformer explainers show how attention works.
This project focuses on what happens when inference meets real hardware limits:

- prefill vs decode behavior
- GEMM vs GEMV bottlenecks
- KV-cache growth
- memory-bound decoding
- quantization trade-offs
- thread scaling limits
- backend-specific overhead
- low-spec hardware behavior

The goal is to make these bottlenecks **measurable**, **visible**, and **reproducible**.

---

## The core idea

```text
Prompt tokens
   │
   ▼
Prefill
   ├─ many tokens at once
   ├─ GEMM-like work
   └─ compute-friendly parallelism
   │
   ▼
KV-cache grows
   │
   ▼
Decode
   ├─ one token at a time
   ├─ GEMV-like work
   ├─ repeated KV-cache reads
   └─ memory bandwidth becomes the wall
```

This lab is built around that transition: **from compute-friendly prefill to memory-bound decode**.

---

## Two measurement modes

### 1. Auto Bottleneck Measurement

Use this mode when you want to profile a model without manually opening its internals.

Auto mode treats the model/runtime as a mostly black-box system and measures:

- TTFT: time to first token
- prefill latency
- decode latency
- tokens/sec
- memory usage
- KV-cache estimate
- thread scaling
- backend comparison
- precision comparison

```bash
llm-bottleneck analyze \
  --model ./models/my-model \
  --backend cpu \
  --precision int4 \
  --prompt "Explain KV-cache in one paragraph."
```

Auto mode answers:

> **Where does this model seem slow?**

---

### 2. Surgical Bottleneck Dissection

Use this mode when the model architecture has a dedicated dissector.

Surgical mode exposes internal bottlenecks such as:

- layer-by-layer latency
- attention vs MLP cost
- GEMM/GEMV behavior
- KV-cache read/write cost
- weight loading pressure
- activation memory
- quantization error
- operator-level bottlenecks

```bash
llm-bottleneck dissect \
  --model-family gemma3n \
  --model ./models/gemma-3n-e4b \
  --backend cpu \
  --precision int4 \
  --profile attention,mlp,kv-cache,matmul
```

Surgical mode answers:

> **Exactly which internal operation is making it slow?**

The first surgical target is the existing Gemma 3N E4B research path, but the long-term goal is a model-family adapter system for Llama, Qwen, DeepSeek, Phi, BitNet-style models, and other local LLM formats.

---

## Quantized runnable inference

Inference is included for measurement, not for chatbot competition.

The lab should support runnable inference paths across precision modes:

- FP16 / BF16 baseline
- INT8
- INT4
- ternary or BitNet-style experiments later

The key question is not only:

> Can the model run?

The real question is:

> What bottleneck appears when this model runs under this precision, backend, and hardware setup?

---

## Planned report site

The GitHub Pages site will be a **report-style visual lab**, not just a landing page.

Planned sections:

| Section | Purpose |
|---|---|
| Overview | What was measured and why it matters |
| Auto Reports | Model/backend/precision benchmark summaries |
| Surgical Reports | Deep dives for supported architectures |
| Visual Lab | WebGPU/WebGL scenes for inference bottlenecks |
| Reproducibility | Hardware, commands, configs, raw result files |
| Release CLI | Downloadable bottleneck measurement program |

Planned visual scenes:

- Prefill vs Decode
- GEMM vs GEMV
- KV-cache growth
- Quantization trade-offs
- Memory wall / roofline-style view
- Thread scaling
- Backend comparison

Large GIFs are intentionally avoided. The visual lab should render real-time animations in the browser using WebGL/WebGPU where possible, with Canvas/SVG fallbacks.

---

## Release program vision

The long-term release artifact is a CLI/binary named:

```text
llm-bottleneck
```

Planned commands:

```bash
llm-bottleneck run       # run quantized inference for measurement
llm-bottleneck analyze   # automatic bottleneck measurement
llm-bottleneck dissect   # model-specific internal dissection
llm-bottleneck report    # generate Markdown/HTML/JSON reports
llm-bottleneck visualize # export data for WebGPU visual reports
```

Planned release assets:

```text
llm-bottleneck-linux-x64
llm-bottleneck-linux-arm64
llm-bottleneck-windows-x64.exe
llm-bottleneck-macos-arm64
```

---

## Current status

This repository currently contains a runnable research scaffold and low-level experimental paths.

| Area | Status |
|---|---|
| CLI scaffold | Working dry-run path |
| Benchmark/report generation | Working dry-run path |
| Tiny stub model | Available for platform checks |
| CPU backend | Reference/skeleton path |
| Vulkan backend | Skeleton / planned offload path |
| FPGA-style NPU path | Experimental research path |
| Gemma 3N E4B path | Preserved as the first surgical target |
| WebGPU visual lab | Planned |
| Auto bottleneck measurement | Planned |
| Multi-model dissector system | Planned |
| Production inference server | Not the goal |

---

## Quick start

Use the tiny stub model to verify that the scaffold works without downloading model weights.

```bash
# Test adapter resolution
python run.py \
  --model examples/tiny_model_stub \
  --backend cpu \
  --precision fp16 \
  --dry-run

# Generate benchmark JSON/HTML output in dry-run mode
python benchmark.py \
  --model examples/tiny_model_stub \
  --backends cpu,vulkan \
  --precisions fp16,int4 \
  --dry-run

# Generate Markdown summary from benchmark results
python scripts/generate_report.py
```

> Dry-run mode does not load real model weights and does not represent real performance.
> It verifies adapter routing, backend selection, precision routing, and report generation.

---

## Planned architecture

```text
llm-bottleneck-lab/
├── engine/                     # Modular research engine scaffold
├── profiler/
│   ├── auto/                    # Black-box bottleneck measurement
│   └── surgical/                # Model-specific internal dissection
├── models/
│   └── adapters/                # Generic + model-family adapters
├── quantization/                # FP16/INT8/INT4/ternary experiments
├── reports/                     # Report templates and result schemas
├── web/                         # GitHub Pages visual report site
├── experiments/                 # Python experiments and generated data
├── native/                      # Low-level CPU/Vulkan/native work
├── results/                     # Generated benchmark/report artifacts
├── x64/
│   └── gemma3N_E4B/             # Preserved first surgical target
├── benchmark.py
├── run.py
└── scripts/
```

---

## Model support strategy

The project should not be locked to one model family.

### Generic mode

Generic mode should work with many local model paths or runtimes by measuring observable behavior:

- latency
- tokens/sec
- memory usage
- precision impact
- backend impact
- thread scaling

### Surgical mode

Surgical mode should be added one family at a time through dedicated dissectors:

```text
models/adapters/
├── generic/
├── gemma3n/
├── llama/
├── qwen/
├── deepseek/
├── phi/
└── bitnet/
```

The Gemma 3N E4B path is the first deep-dive target because it already has focused experimental work. Future models should be added only when the dissector can expose meaningful internals.

---

## Design principles

1. **Measure first.** Do not guess bottlenecks when they can be measured.
2. **Visualize clearly.** Every chart or animation must explain a system behavior.
3. **Separate generic and surgical modes.** Black-box profiling and architecture dissection are different tasks.
4. **Quantization is core.** Precision changes are part of bottleneck analysis, not an optional add-on.
5. **No huge assets.** Avoid large GIFs, videos, and model weights in the repository.
6. **Static-site friendly.** GitHub Pages should serve the report; computation runs locally or in the browser.
7. **Research honesty.** Mark approximations, dry-runs, and planned features clearly.

---

## Roadmap

### Phase 0 — Rebrand and clarify

- [ ] Replace old `llm-lite` wording
- [ ] Clarify that this is a bottleneck lab, not a production inference engine
- [ ] Add About description and GitHub topics
- [ ] Preserve existing Gemma path as first surgical target

### Phase 1 — Auto bottleneck measurement

- [ ] Define result schema
- [ ] Measure TTFT, prefill, decode, tokens/sec
- [ ] Add memory and KV-cache estimates
- [ ] Add backend and precision comparison
- [ ] Generate JSON/Markdown reports

### Phase 2 — Quantized runnable inference

- [ ] Stabilize `run` path for measurement
- [ ] Add precision routing for FP16/INT8/INT4
- [ ] Add clear measurement output
- [ ] Keep inference scoped to reproducible experiments

### Phase 3 — Surgical dissection

- [ ] Formalize model-family dissector interface
- [ ] Convert Gemma 3N E4B work into the first official surgical report
- [ ] Add layer/operator timing where possible
- [ ] Add attention/MLP/KV-cache breakdown

### Phase 4 — GitHub Pages report site

- [ ] Create report-style site
- [ ] Add benchmark dashboards
- [ ] Add WebGL/WebGPU visual scenes
- [ ] Add static fallback images
- [ ] Link reports to raw JSON/CSV data

### Phase 5 — Release CLI

- [ ] Package `llm-bottleneck` binary
- [ ] Add `run`, `analyze`, `dissect`, `report` commands
- [ ] Publish release artifacts
- [ ] Include reproducibility examples

### Phase 6 — Multi-model expansion

- [ ] Add generic runtime adapters
- [ ] Add Llama-family dissector
- [ ] Add Qwen-family dissector
- [ ] Add DeepSeek/Phi/BitNet-style experiments
- [ ] Compare models under the same report schema

---

## What this project is not

- Not a hosted chatbot
- Not a production inference server
- Not a model zoo
- Not a benchmark leaderboard without explanations
- Not a replacement for `llama.cpp`, vLLM, TensorRT-LLM, or Ollama

This project exists to explain and measure bottlenecks.

---

## One-line summary

> **LLM Bottleneck Lab measures, visualizes, and dissects why LLM inference becomes slow across models, backends, and quantization modes.**

---

## License

Apache-2.0. See [`LICENSE`](./LICENSE).
