import numpy as np
from transformers import AutoTokenizer
import os
import ctypes 

base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: config is in local_gemma_3n_int4 now
model_id = os.path.join(base_dir, "local_gemma_3n_int4")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

# -----------------------------------------------------------
# C - DLL porting (load and init set)

dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)


# ><><><><><><><><Parameters><><><><><><><><

# gelu function 
c_lib.run_gelu_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
c_lib.run_gelu_inplace.restype = None

# int4 unpacking
c_lib.run_unpack_int4_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),  
    ctypes.c_float,                                                        
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int                                                           
]
c_lib.run_unpack_int4_inplace.restype = None

# ROPE function
c_lib.run_rope_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x array
    ctypes.c_int,   # pos
    ctypes.c_float, # theta_base
    ctypes.c_int,   # num_heads
    ctypes.c_int    # dim
]
c_lib.run_rope_inplace.restype = None

# Fused QK Norm + RoPE
c_lib.run_qk_norm_rope_fused.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # q
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # k
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # gamma_q
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # gamma_k
    ctypes.c_int,   # pos
    ctypes.c_float, # theta_base
    ctypes.c_int,   # num_q_heads
    ctypes.c_int,   # num_k_heads
    ctypes.c_int    # head_dim
]
c_lib.run_qk_norm_rope_fused.restype = None

# Fused GQA
c_lib.run_gqa_fused.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # Q
    ctypes.c_void_p,  # K_cache (float16 as raw pointer)
    ctypes.c_void_p,  # V_cache (float16 as raw pointer)
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out
    ctypes.c_int,   # seq_len
    ctypes.c_int,   # num_kv_groups
    ctypes.c_int,   # heads_per_group
    ctypes.c_int    # head_dim
]
c_lib.run_gqa_fused.restype = None

# Small float32 GEMV
c_lib.run_small_gemv_f32.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # mat
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out
    ctypes.c_int,  # M_out
    ctypes.c_int   # K_in
]
c_lib.run_small_gemv_f32.restype = None

# ><><><><><><><><><><><><><><><><><><><><><

# -----------------------------------------------------------

# ================================================================
# Pre-allocated buffers for hot-path operations
# ================================================================
_EMBED_BUF = None
def _get_embed_buf(size: int) -> np.ndarray:
    global _EMBED_BUF
    if _EMBED_BUF is None or _EMBED_BUF.size != size:
        _EMBED_BUF = np.empty(size, dtype=np.float32)
    return _EMBED_BUF

# Pre-allocated buffers for GQA (fixed sizes for Gemma 3N)
_GQA_Q = np.empty((2, 4, 256), dtype=np.float32)
_GQA_SCORES = None  # Will be allocated on first use based on seq_len

# Pre-allocated buffers for QK norm (fixed sizes)
_QK_Q_RESHAPED = np.empty((8, 256), dtype=np.float32)  # 2048 / 256 = 8 heads
_QK_K_RESHAPED = np.empty((2, 256), dtype=np.float32)  # 512 / 256 = 2 heads
_QK_Q_OUT = np.empty(2048, dtype=np.float32)
_QK_K_OUT = np.empty(512, dtype=np.float32)


def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

# cpp ver
def embedding(token_id, W_packed, W_scale):
    row_packed = np.ascontiguousarray(W_packed[token_id])
    row_scale = float(W_scale[token_id])
    packed_length = row_packed.size
    
    out_f32 = _get_embed_buf(packed_length * 2)
    c_lib.run_unpack_int4_inplace(row_packed, ctypes.c_float(row_scale), out_f32, packed_length)
    
    return out_f32

def gelu(x):
    # Fast path: if already float32 1D contiguous, no copy needed
    if x.ndim == 1 and x.dtype == np.float32 and x.flags['C_CONTIGUOUS']:
        c_lib.run_gelu_inplace(x, x.size)
        return x
    
    # General path
    x_flat = np.ascontiguousarray(x.flatten().astype(np.float32))
    c_lib.run_gelu_inplace(x_flat, x_flat.size)
    return x_flat.reshape(x.shape)


# ================================================================
# Fused QK Norm + RoPE (single C++ call replaces 3 Python calls)
# ================================================================
_GAMMA_Q_CACHE = {}
_GAMMA_K_CACHE = {}

def cpu_qk_norm_rope_fused(Q, K, gamma_q, gamma_k, pos, theta_base):
    """Fused: QK norm + RoPE in one C++ call. Modifies Q/K in-place."""
    # Ensure contiguous float32
    if Q.dtype != np.float32 or not Q.flags['C_CONTIGUOUS']:
        Q = np.ascontiguousarray(Q.astype(np.float32))
    if K.dtype != np.float32 or not K.flags['C_CONTIGUOUS']:
        K = np.ascontiguousarray(K.astype(np.float32))
    
    # Cache gamma conversions
    gq_id = id(gamma_q)
    if gq_id not in _GAMMA_Q_CACHE:
        _GAMMA_Q_CACHE[gq_id] = np.ascontiguousarray(gamma_q.astype(np.float32))
    gk_id = id(gamma_k)
    if gk_id not in _GAMMA_K_CACHE:
        _GAMMA_K_CACHE[gk_id] = np.ascontiguousarray(gamma_k.astype(np.float32))
    
    num_q_heads = Q.size // 256  # 8
    num_k_heads = K.size // 256  # 2
    
    c_lib.run_qk_norm_rope_fused(
        Q, K,
        _GAMMA_Q_CACHE[gq_id], _GAMMA_K_CACHE[gk_id],
        int(pos), float(theta_base),
        int(num_q_heads), int(num_k_heads), 256
    )
    return Q, K

# Legacy fallback (still available)
def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    num_q_heads = Q.size // 256
    num_k_heads = K.size // 256
    Q_f32 = Q.astype(np.float32) if Q.dtype != np.float32 else Q
    K_f32 = K.astype(np.float32) if K.dtype != np.float32 else K
    Q_reshaped = Q_f32.reshape(num_q_heads, 256)
    K_reshaped = K_f32.reshape(num_k_heads, 256)
    q_rms = np.sqrt(np.mean(Q_reshaped ** 2, axis=1, keepdims=True) + 1e-6)
    k_rms = np.sqrt(np.mean(K_reshaped ** 2, axis=1, keepdims=True) + 1e-6)
    Q_norm = (Q_reshaped / q_rms) * gamma_q
    K_norm = (K_reshaped / k_rms) * gamma_k
    return Q_norm.flatten(), K_norm.flatten()

def cpu_rope(x, pos, theta_base):
    dim = 256
    num_heads = len(x) // dim
    if x.dtype == np.float32 and x.flags['C_CONTIGUOUS']:
        x_flat = x
    else:
        x_flat = np.ascontiguousarray(x.astype(np.float32).flatten())
    c_lib.run_rope_inplace(x_flat, int(pos), float(theta_base), int(num_heads), int(dim))
    return x_flat

def cpu_update_kv_cache(K_rope, V, token_cnt, layer_idx, K_cache, V_cache):
    pass

# ================================================================
# Fused GQA — entire attention in one C++ call
# ================================================================
_GQA_OUT_BUF = np.empty(2048, dtype=np.float32)  # 8 heads * 256 dim

def cpu_gqa_fused(Q_rope, K_cache_layer, V_cache_layer):
    """Full GQA via C++ fused kernel. K/V cache stay as float16."""
    Q_f32 = Q_rope if (Q_rope.dtype == np.float32 and Q_rope.flags['C_CONTIGUOUS']) \
            else np.ascontiguousarray(Q_rope.astype(np.float32))
    
    seq_len = K_cache_layer.shape[0]
    K_contig = np.ascontiguousarray(K_cache_layer)  # ensure contiguous
    V_contig = np.ascontiguousarray(V_cache_layer)
    
    c_lib.run_gqa_fused(
        Q_f32,
        K_contig.ctypes.data,
        V_contig.ctypes.data,
        _GQA_OUT_BUF,
        seq_len,
        2,    # num_kv_groups
        4,    # heads_per_group  
        256   # head_dim
    )
    return _GQA_OUT_BUF.copy()

# Legacy fallback
def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    Q_reshaped = Q_rope.reshape(2, 4, 256).astype(np.float32)
    K_mat = K_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    V_mat = V_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    K_t = K_mat.transpose(1, 2, 0)
    scores = np.matmul(Q_reshaped, K_t)
    scores -= np.max(scores, axis=-1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= np.sum(scores, axis=-1, keepdims=True)
    V_t = V_mat.transpose(1, 0, 2)
    attn_out = np.matmul(scores, V_t)
    return attn_out.flatten()

def cpu_sample_token(probs):
    return int(np.argmax(probs))
