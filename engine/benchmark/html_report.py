from engine.core.report import BenchmarkMetrics
from typing import List
import os

def generate_html_report(results: List[BenchmarkMetrics], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    html = ["<html><body><h1>Benchmark Report</h1><table border='1'>"]
    html.append("<tr><th>Model</th><th>Backend</th><th>Precision</th><th>Tokens/s</th><th>Memory (MB)</th></tr>")

    for r in results:
        html.append(f"<tr><td>{r.model_name}</td><td>{r.backend}</td><td>{r.precision}</td><td>{r.tokens_per_sec:.2f}</td><td>{r.peak_memory_mb:.2f}</td></tr>")

    html.append("</table></body></html>")

    with open(filepath, 'w') as f:
        f.write("\n".join(html))
