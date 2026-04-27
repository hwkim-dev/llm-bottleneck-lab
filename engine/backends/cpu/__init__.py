class CPUBackend:
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads

    def info(self):
        return {
            "name": "cpu",
            "status": "skeleton",
            "target_hardware": "x86/ARM CPU",
            "notes": "Reference CPU implementation."
        }

    def supports_precision(self, precision: str) -> bool:
        return precision in ["fp16", "int8", "int4", "ternary"]

    def execute(self, op: str, *args, **kwargs):
        pass
