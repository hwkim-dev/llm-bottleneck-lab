import argparse
import sys
import os
import json

from engine.benchmark.runner import BenchmarkRunner
from engine.benchmark.html_report import generate_html_report

def main():
    parser = argparse.ArgumentParser(description="llm-lite benchmark runner")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--backends", type=str, required=True, help="Comma-separated list of backends (cpu,vulkan,npu_uca)")
    parser.add_argument("--precisions", type=str, required=True, help="Comma-separated list of precisions (fp16,int8,int4,ternary)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (returns stub data)")

    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    precisions = [p.strip() for p in args.precisions.split(",")]

    runner = BenchmarkRunner(model_path=args.model, backends=backends, precisions=precisions)
    results = runner.run(dry_run=args.dry_run)

    os.makedirs("results/benchmarks", exist_ok=True)

    model_name = os.path.basename(args.model)
    if not model_name:
        model_name = "unknown_model"

    for i, res in enumerate(results):
        base_name = f"results/benchmarks/{model_name}_{res.backend}_{res.precision}"
        res.save_json(f"{base_name}.json")
        res.save_markdown(f"{base_name}.md")

    html_path = f"results/benchmarks/summary.html"
    generate_html_report(results, html_path)

    # Try generating standard summary MD report
    try:
        import scripts.generate_report
        scripts.generate_report.generate_markdown_summary("results/benchmarks", "results/benchmarks/summary.md")
    except Exception as e:
        pass

    print(f"Benchmarking complete. Reports saved to results/benchmarks/")

if __name__ == "__main__":
    main()
