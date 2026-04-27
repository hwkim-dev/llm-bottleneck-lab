# Contributing to llm-lite

We welcome contributions to expand the experimental reach of this lab!

## Scope
We accept contributions that align with the goal of studying small LLM performance on low-spec hardware (CPU, iGPU, FPGA/NPU paths).
We **do not** accept PRs aiming to make this a universal production deployment engine (use llama.cpp or MLC LLM instead).

## Running Tests
Before submitting a PR, make sure the smoke tests and basic dry runs pass:
```bash
python -m compileall engine scripts
python run.py --model examples/tiny_model_stub --backend cpu --precision fp16 --dry-run
python benchmark.py --model examples/tiny_model_stub --backends cpu,vulkan --precisions fp16,int4 --dry-run
```
