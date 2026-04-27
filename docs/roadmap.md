# llm-lite Roadmap

The development of `llm-lite` follows a phased approach to build the ultimate low-spec LLM systems lab.

## Phase 1: Architecture Registry
* [x] Establish the `engine/` directory structure.
* [x] Implement the ModelRegistry and RuntimeContext.
* [x] Create adapter skeletons for Gemma3N, Llama, Qwen, DeepSeek-Distill, and BitNet.

## Phase 1.5: Reliability Pass Completed
* [x] CI pipeline verification
* [x] Runnable dry-run CLI
* [x] Verified model skeletons and configurations

## Phase 2: Quantization Matrix
* [x] Define skeleton interfaces for FP16, INT8, INT4, and Ternary quantization.
* [ ] Implement actual quantization and packing logic for INT4.
* [ ] Implement ternary packing POC for BitNet.

## Phase 3: CPU Benchmark
* [x] Create `benchmark.py` with dummy metrics capability.
* [ ] Implement true CPU timing and memory tracking.
* [ ] Gather baseline data for FP16 and INT4 on target low-spec hardware.

## Phase 4: Vulkan Offload Study
* [ ] Port necessary kernels from the legacy Gemma implementation to the new Vulkan backend architecture.
* [ ] Measure dispatch overhead vs compute time.

## Phase 5: BitNet Ternary Study
* [ ] Complete the ternary adapter and evaluate addition-only matmul on the CPU.

## Phase 6: Paper-style Report
* [ ] Generate visualizations via `generate_report.py`.
* [ ] Synthesize findings into `docs/07_research_paper_outline.md`.
