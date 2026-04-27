def quantize_to_fp16(tensor):
    pass

def dequantize_from_fp16(tensor):
    pass

def info():
    return {
        "name": "fp16",
        "bit_width": 16,
        "target_models": ["llama", "qwen", "gemma3n", "deepseek-distill"],
        "notes": "Standard 16-bit floating point.",
        "is_experimental": False
    }
