# Reproducibility in llm-lite

To ensure that results and claims in `llm-lite` are trustworthy and verifiable, we provide simple ways to check the platform functionality:

## Environment
Ensure you are using Python 3.10 or newer. There are no heavy external dependencies required for the basic adapter resolutions, smoke tests, and dry runs.

## Smoke Test
You can use the dummy model in `examples/tiny_model_stub` to verify that `run.py` parses configs and resolves adapters correctly without loading real weights:
```bash
python run.py --model examples/tiny_model_stub --backend cpu --precision fp16 --dry-run
```

## Dry-run Benchmark
To ensure that the benchmark scaffolding works:
```bash
python benchmark.py --model examples/tiny_model_stub --backends cpu,vulkan --precisions fp16,int4 --dry-run
```
This generates placeholder JSON and HTML reports inside `results/benchmarks/`.

## Real Benchmark Plan
Currently, real measurements for most models are WIP. We prioritize making the platform solid before implementing unverified real benchmarks.
