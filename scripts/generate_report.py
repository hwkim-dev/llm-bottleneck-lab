import os
import json
import argparse

def generate_markdown_summary(benchmark_dir: str, output_file: str):
    """Generates a summary markdown table from benchmark JSON files."""
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    files = [f for f in os.listdir(benchmark_dir) if f.endswith('.json')]

    if not files:
        print("No benchmark JSON files found.")
        return

    md = ["# Benchmark Summary\n"]
    md.append("| Model | Backend | Precision | Tokens/s | Peak Mem (MB) | Notes |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for file in sorted(files):
        with open(os.path.join(benchmark_dir, file), 'r') as f:
            data = json.load(f)
            tps = f"{data['tokens_per_sec']:.2f}" if data.get('tokens_per_sec') is not None else "not measured"
            mem = f"{data['peak_memory_mb']:.2f}" if data.get('peak_memory_mb') is not None else "not measured"
            md.append(f"| {data['model_name']} | {data['backend']} | {data['precision']} | {tps} | {mem} | {data['notes']} |")

    with open(output_file, 'w') as f:
        f.write("\n".join(md))
    print(f"Summary generated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/benchmarks", help="Directory containing JSON reports")
    parser.add_argument("--out", default="results/benchmarks/summary.md", help="Output markdown file")
    args = parser.parse_args()

    generate_markdown_summary(args.dir, args.out)
