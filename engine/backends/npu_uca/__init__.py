class NPUBackend:
    def __init__(self):
        pass

    def info(self):
        return {
            "name": "npu_uca",
            "status": "experimental",
            "target_hardware": "FPGA-style/proprietary/bare-metal NPU",
            "notes": "FPGA-style/proprietary/bare-metal research path."
        }

    def supports_precision(self, precision: str) -> bool:
        return precision in ["int8", "int4"]

    def execute(self, op: str, *args, **kwargs):
        pass
