from engine.core.report import BenchmarkMetrics
from typing import List
import os

def generate_html_report(results: List[BenchmarkMetrics], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    html = ["<!DOCTYPE html>", "<html>", "<head>", "<title>llm-lite Dry-Run Benchmark Summary</title>", "</head>", "<body>"]
    html.append("<h1>llm-lite Dry-Run Benchmark Summary</h1>")

    # Check if dry-run
    if results and "dry-run" in (results[0].notes or ""):
        html.append("<p><strong>Note:</strong> This is a dry-run report. No real model weights were loaded and no performance was measured.</p>")

    html.append("<table border='1'>")
    html.append("<tr><th>Model</th><th>Architecture</th><th>Backend</th><th>Precision</th><th>Prompt Tokens</th><th>Generated Tokens</th><th>Load Time (s)</th><th>Prefill Time (s)</th><th>Decode Time (s)</th><th>Tokens/s</th><th>Peak Memory (MB)</th><th>Notes</th></tr>")

    for r in results:
        load_time = f"{r.load_time_sec:.4f}" if r.load_time_sec is not None else "not measured"
        prefill_time = f"{r.prefill_time_sec:.4f}" if r.prefill_time_sec is not None else "not measured"
        decode_time = f"{r.decode_time_sec:.4f}" if r.decode_time_sec is not None else "not measured"
        tokens_per_sec = f"{r.tokens_per_sec:.2f}" if r.tokens_per_sec is not None else "not measured"
        peak_memory_mb = f"{r.peak_memory_mb:.2f}" if r.peak_memory_mb is not None else "not measured"

        html.append(f"<tr><td>{r.model_name}</td><td>{r.architecture}</td><td>{r.backend}</td><td>{r.precision}</td><td>{r.prompt_tokens}</td><td>{r.generated_tokens}</td><td>{load_time}</td><td>{prefill_time}</td><td>{decode_time}</td><td>{tokens_per_sec}</td><td>{peak_memory_mb}</td><td>{r.notes}</td></tr>")

    html.append("</table></body></html>")

    with open(filepath, 'w') as f:
        f.write("\n".join(html))
