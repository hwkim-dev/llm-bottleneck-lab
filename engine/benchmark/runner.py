from engine.core.report import BenchmarkMetrics
import time

class BenchmarkRunner:
    def __init__(self, model_path: str, backends: list[str], precisions: list[str]):
        self.model_path = model_path
        self.backends = backends
        self.precisions = precisions

    def run(self, dry_run: bool = False) -> list[BenchmarkMetrics]:
        results = []
        for backend in self.backends:
            for precision in self.precisions:
                # In dry-run mode, we just generate dummy metrics
                if dry_run:
                    metrics = BenchmarkMetrics(
                        model_name=self.model_path.split('/')[-1] or self.model_path,
                        architecture="unknown",
                        backend=backend,
                        precision=precision,
                        prompt_tokens=10,
                        generated_tokens=20,
                        load_time_sec=0.1,
                        prefill_time_sec=0.05,
                        decode_time_sec=0.5,
                        tokens_per_sec=40.0,
                        peak_memory_mb=1024.0,
                        notes="dry run"
                    )
                    results.append(metrics)
                else:
                    # Stub for actual benchmark run
                    pass
        return results
