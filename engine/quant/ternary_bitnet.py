def quantize_to_ternary(tensor):
    """
    Quantizes weights to -1, 0, 1 for BitNet.
    """
    pass

def dequantize_from_ternary(tensor):
    pass

def info():
    return {
        "name": "ternary_bitnet",
        "bit_width": 2, # Logically
        "target_models": ["bitnet"],
        "notes": "Ternary quantization (-1, 0, +1) targeting BitNet.",
        "is_experimental": True
    }
