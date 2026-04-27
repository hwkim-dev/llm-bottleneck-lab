from engine.core.report import BenchmarkMetrics


class BenchmarkRunner:
    def __init__(self, model_path: str, backends: list[str], precisions: list[str]):
        self.model_path = model_path
        self.backends = backends
        self.precisions = precisions

    def run(self, dry_run: bool = False) -> list[BenchmarkMetrics]:
        results = []
        model_name = self.model_path.split('/')[-1] or self.model_path

        for backend in self.backends:
            for precision in self.precisions:
                # In dry-run mode, we just generate dummy metrics marked as null or notes
                if dry_run:
                    metrics = BenchmarkMetrics(
                        model_name=model_name,
                        architecture="unknown",
                        backend=backend,
                        precision=precision,
                        prompt_tokens=0,
                        generated_tokens=0,
                        load_time_sec=None,
                        prefill_time_sec=None,
                        decode_time_sec=None,
                        tokens_per_sec=None,
                        peak_memory_mb=None,
                        notes="dry-run; not measured"
                    )
                    results.append(metrics)
                else:
                    # Stub for actual benchmark run
                    pass
        return results
