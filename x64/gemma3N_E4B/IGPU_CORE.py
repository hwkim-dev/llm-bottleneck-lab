import numpy as np
import os
import ctypes

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "vulkan_core.so")
vk_lib = ctypes.CDLL(dll_path)


# -----------------------------------------------------------
# init Vulkan engine

vk_lib.init_vulkan_engine.argtypes = []
vk_lib.init_vulkan_engine.restype = None

os.chdir(base_dir)

# Load the GPU just once when the program is turned on.
vk_lib.init_vulkan_engine()

# <><><><><><><><><><><><><><>Parameters<><><><><><><><><><><><><><>

vk_lib.run_vulkan_gemv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),   # mat_p
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # scale
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out
    ctypes.c_int, # M_out
    ctypes.c_int  # K_in
]
vk_lib.run_vulkan_gemv.restype = None

# prefetch weight
vk_lib.prefetch_weight_async.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'), # mat_p
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int  
]
vk_lib.prefetch_weight_async.restype = None

# ping pong
vk_lib.run_vulkan_gemv_pingpong.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int  
]
vk_lib.run_vulkan_gemv_pingpong.restype = None

# <><><><><><><><><><><><><><>Parameters<><><><><><><><><><><><><><>

# ================================================================
# Pre-allocated buffer pools (avoid per-call allocation)
# ================================================================
_OUTPUT_BUF_POOL = {}
def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]

# Reusable x buffer (avoids ascontiguousarray + astype every call)
_X_BUF = None
def _get_x_buf(size: int) -> np.ndarray:
    global _X_BUF
    if _X_BUF is None or _X_BUF.size < size:
        _X_BUF = np.empty(size, dtype=np.float32)
    return _X_BUF[:size]

def _prepare_x(x_vec: np.ndarray) -> np.ndarray:
    """Convert x to float32 contiguous, reusing buffer when possible."""
    if x_vec.dtype == np.float32 and x_vec.flags['C_CONTIGUOUS']:
        return x_vec
    buf = _get_x_buf(x_vec.size)
    np.copyto(buf, x_vec.ravel().astype(np.float32))
    return buf

# -----------------------------------------------------------

# legacy
# # ================================================================
def preload_and_free(W: dict, keys: list): pass
def _get_or_upload_weight(weight_data): pass
def warmup(): print("[Vulkan_GEMV] shader engine load complete ")

# ================================================================
# 4. Matrix product interface (optimized: caller-provided out buffer)
# ----------------------------------------------------------------
# Only INT4-packed (uint8) tuples hit the Vulkan GEMV kernel.  INT8 / FP16 /
# FP32 run on CPU (np.dot) — new shaders for those formats are TODO.
# ================================================================
def igpu_matmul(x_vec: np.ndarray, weight_data, out: np.ndarray = None) -> np.ndarray:
    x_f32 = _prepare_x(x_vec)

    if isinstance(weight_data, tuple) and weight_data[0].dtype == np.uint8:
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2

        if out is None:
            out = _get_output_buf(M_out)

        vk_lib.run_vulkan_gemv(x_f32, packed, scale, out, M_out, K_in)
        return out

    # CPU fallback for INT8 / FP16 / FP32 (caller in main.hw_matmul normally
    # routes these directly, this is the safety net).
    if isinstance(weight_data, tuple):
        packed, scale = weight_data
        w_f32 = packed.astype(np.float32) * scale[:, np.newaxis]
        return np.dot(x_f32, w_f32.T)
    w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
    return np.dot(x_f32, w_f32)

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    out = igpu_matmul(x_vec, weight_data)
    import CPU_CORE
    return CPU_CORE.gelu(out)


def prefetch_weight(weight_data, buf_idx: int):
    # Prefetch only makes sense for the Vulkan path (INT4 packed).
    if isinstance(weight_data, tuple) and weight_data[0].dtype == np.uint8:
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        vk_lib.prefetch_weight_async(packed, M_out, K_in, buf_idx)

def compute_pingpong(x_vec: np.ndarray, weight_data, buf_idx: int, out: np.ndarray = None) -> np.ndarray:
    x_f32 = _prepare_x(x_vec)

    if isinstance(weight_data, tuple) and weight_data[0].dtype == np.uint8:
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2

        if out is None:
            out = _get_output_buf(M_out)

        vk_lib.run_vulkan_gemv_pingpong(x_f32, scale, out, M_out, K_in, buf_idx)
        return out

    # CPU fallback for non-INT4 formats.
    if isinstance(weight_data, tuple):
        packed, scale = weight_data
        w_f32 = packed.astype(np.float32) * scale[:, np.newaxis]
        return np.dot(x_f32, w_f32.T)
    w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
    return np.dot(x_f32, w_f32.T)
