def quantize_to_int8(tensor):
    pass

def dequantize_from_int8(tensor):
    pass

def info():
    return {
        "name": "int8",
        "bit_width": 8,
        "target_models": ["llama", "qwen", "gemma3n", "deepseek-distill"],
        "notes": "8-bit integer quantization.",
        "is_experimental": False
    }
