class VulkanBackend:
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id

    def info(self):
        return {
            "name": "vulkan",
            "status": "skeleton/offload study target",
            "target_hardware": "iGPU / dGPU via Vulkan",
            "notes": "Targeted for offloading compute-intensive matmul layers."
        }

    def supports_precision(self, precision: str) -> bool:
        return precision in ["fp16", "int8", "int4"]

    def execute(self, op: str, *args, **kwargs):
        pass
