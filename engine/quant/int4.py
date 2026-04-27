def quantize_to_int4(tensor):
    pass

def dequantize_from_int4(tensor):
    pass

def info():
    return {
        "name": "int4",
        "bit_width": 4,
        "target_models": ["llama", "qwen", "gemma3n", "deepseek-distill"],
        "notes": "4-bit integer quantization.",
        "is_experimental": False
    }
