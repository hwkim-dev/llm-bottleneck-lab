import numpy as np
import CPU_CORE
import safeTensor
import math

import gc
import sys
import ctypes

# see ram usage
import os
import psutil

# ================================================================
# ██████ Configuration ██████
# ================================================================
ACCEL_MODE = "IGPU"
#ACCEL_MODE = "CPU"

# ── Weight Quantization Mode ──
WEIGHT_MODE = "INT4"  # Options: "INT4" | "INT8" | "FP16" | "FP32"

# ── Feature Map (Activation) Quantization Mode ──
FEATURE_MODE = "FP32"  # Options: "INT4" | "INT8" | "BF16" | "FP32"

# ================================================================

_IGPU_WEIGHT_KEYS = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"]
NUM_LAYERS = 35

if ACCEL_MODE == "IGPU":
    import IGPU_CORE as FAST_MATRIX_CORE
elif ACCEL_MODE == "CPU":
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
# -----------------------------------------------------------
# C - DLL porting (load and init set)

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# ><><><><><><><><Parameters><><><><><><><><

# RMS Norm
c_lib.run_RMSNorm_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
c_lib.run_RMSNorm_inplace.restype = None

# Softmax
c_lib.run_softmax_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_float
]
c_lib.run_softmax_inplace.restype = None

# ><><><><><><><><><><><><><><><><><><><><><

# -----------------------------------------------------------

# ================================================================
# ██████ Feature Map Quantization Engine ██████
# ================================================================
def _qf(x: np.ndarray) -> np.ndarray:
    """Quantize Feature: apply feature map precision to activation tensor."""
    if FEATURE_MODE == "FP32":
        return x
    elif FEATURE_MODE == "BF16":
        return x.astype(np.float16).astype(np.float32)
    elif FEATURE_MODE == "INT8":
        abs_max = np.max(np.abs(x))
        if abs_max < 1e-10: return x
        # 고속 아웃라이어 클리핑 (정렬 없이 평균값의 15배를 최대치로 근사)
        clip_max = np.mean(np.abs(x)) * 15.0
        scale = min(abs_max, clip_max) / 127.0
        return (np.clip(np.round(x / scale), -128, 127) * scale).astype(np.float32)
    elif FEATURE_MODE == "INT4":
        abs_max = np.max(np.abs(x))
        if abs_max < 1e-10: return x
        clip_max = np.mean(np.abs(x)) * 10.0
        scale = min(abs_max, clip_max) / 7.0
        return (np.clip(np.round(x / scale), -8, 7) * scale).astype(np.float32)
    return x

# ================================================================
# Pre-allocated activation buffers
# ================================================================
_BUF_2048  = np.empty(2048,  dtype=np.float32)
_BUF_2048b = np.empty(2048,  dtype=np.float32)
_BUF_512   = np.empty(512,   dtype=np.float32)
_BUF_512b  = np.empty(512,   dtype=np.float32)
_BUF_16384 = np.empty(16384, dtype=np.float32)
_BUF_16384b= np.empty(16384, dtype=np.float32)

# gamma buffers (loaded once, reused)
_GAMMA_CACHE = {}
def _ensure_gamma(gamma):
    gid = id(gamma)
    if gid not in _GAMMA_CACHE:
        _GAMMA_CACHE[gid] = np.ascontiguousarray(gamma.astype(np.float32))
    return _GAMMA_CACHE[gid]


def hw_matmul(x, w, use_gelu=False):
    if ACCEL_MODE == "IGPU":
        return FAST_MATRIX_CORE.igpu_matmul_gelu(x, w) if use_gelu else FAST_MATRIX_CORE.igpu_matmul(x, w)
    else:
        if isinstance(w, tuple):
            packed, scale = w
            low = (packed & 0x0F).astype(np.int8)
            low[low > 7] -= 16
            high = (packed >> 4).astype(np.int8)
            high[high > 7] -= 16
            res = np.empty((packed.shape[0], packed.shape[1]*2), dtype=np.float32)
            res[:, 0::2] = low
            res[:, 1::2] = high
            w_real = res * scale[:, np.newaxis]
            out = np.dot(x, w_real.T)
        else:
            out = np.dot(x, w)
        return CPU_CORE.gelu(out) if use_gelu else out

def hw_prefetch(w, buf_idx):
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple):
        FAST_MATRIX_CORE.prefetch_weight(w, buf_idx)

def hw_compute_pingpong(x, w, buf_idx, use_gelu=False, out=None):
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple):
        result = FAST_MATRIX_CORE.compute_pingpong(x, w, buf_idx, out=out)
        return CPU_CORE.gelu(result) if use_gelu else result
    else:
        return hw_matmul(x, w, use_gelu)
    
def rms_norm(x, gamma):
    if x.dtype == np.float32 and x.flags['C_CONTIGUOUS']:
        x_f32 = x.copy()
    else:
        x_f32 = np.ascontiguousarray(x.astype(np.float32))
    gamma_c = _ensure_gamma(gamma)
    c_lib.run_RMSNorm_inplace(x_f32, gamma_c, x_f32.size)
    return x_f32

def rms_norm_no_gamma(x):
    """RMSNorm without learnable gamma (per-head). V is [512], which is 2 heads of 256."""
    if x.dtype != np.float32: x = x.astype(np.float32)
    x_2d = x.reshape(-1, 256)
    rms = np.sqrt(np.mean(x_2d ** 2, axis=1, keepdims=True) + 1e-6)
    return (x_2d / rms).flatten()

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm) / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def forward_one_token(token_id, pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    safe_token_id = int(min(token_id, W_ple_packed.shape[0] - 1))
    x0 = CPU_CORE.embedding(safe_token_id, W_embed[0], W_embed[1])
    x0 = x0 * math.sqrt(2048.0)

    xs = np.zeros((4, 2048), dtype=np.float32)
    xs[0] = x0
    for k in range(3):
        xs[k + 1] = np.dot(x0, altup_projs[k])
        
    x_proj = hw_matmul(x0, W_ple_proj) / math.sqrt(2048.0)
    x_proj = x_proj.reshape(35, 256)
    x_proj_f32 = x_proj if x_proj.dtype == np.float32 else x_proj.astype(np.float32)
    rms_vals   = np.sqrt(np.mean(x_proj_f32 ** 2, axis=1, keepdims=True) + 1e-6)
    x_proj_normed = (x_proj_f32 / rms_vals) * norm_ple

    unpacked_w_ple = CPU_CORE.embedding(safe_token_id, W_ple_packed, W_ple_scale)
    y = unpacked_w_ple.reshape(35, 256) * math.sqrt(256.0)
    pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

    # ── Lightweight profiler (fires once on first decode step) ──────────
    _PROF = {"altup_pred": 0.0, "qkv": 0.0, "qk_rope": 0.0,
             "attn": 0.0, "o_proj": 0.0, "laurel": 0.0,
             "ffn": 0.0, "altup_corr": 0.0, "ple": 0.0}
    import time as _time
    _profile_this = not getattr(forward_one_token, "_profiled", False)

    ping_pong = 0
    hw_prefetch(W["W_q"][0], ping_pong)

    for i in range(NUM_LAYERS):
        _t0 = _time.perf_counter() if _profile_this else 0

        modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
        coef_mat    = np.dot(W["altup_pred"][i], modalities).reshape(4, 4)
        xs_pred     = xs + np.dot(coef_mat, xs)

        if _profile_this: _PROF["altup_pred"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        x                 = xs_pred[0].copy()
        inputs_normalized = rms_norm(x, W["input_ln"][i])

        # --- Q, K, V projections with ping-pong ---
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_k"][i], next_buf)
        Q = hw_compute_pingpong(inputs_normalized, W["W_q"][i], curr_buf, out=_BUF_2048)
        Q = Q.copy()
        ping_pong = next_buf

        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_v"][i], next_buf)
        K = hw_compute_pingpong(inputs_normalized, W["W_k"][i], curr_buf, out=_BUF_512)
        K = K.copy()
        ping_pong = next_buf

        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_o"][i], next_buf)
        V = hw_compute_pingpong(inputs_normalized, W["W_v"][i], curr_buf, out=_BUF_512b)
        V = V.copy()
        ping_pong = next_buf

        if _profile_this: _PROF["qkv"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        # --- QK Norm + RoPE (fused C++) ---
        theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
        Q, K = CPU_CORE.cpu_qk_norm_rope_fused(Q, K, W["gamma_q"][i], W["gamma_k"][i], pos, theta)

        if _profile_this: _PROF["qk_rope"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        if i < 20:
            K_cache[i, pos, :] = K
            V_cache[i, pos, :] = V
            target_k_cache = K_cache[i, :pos + 1, :]
            target_v_cache = V_cache[i, :pos + 1, :]
        else:
            if i % 5 == 4:
                target_k_cache = K_cache[19, :pos + 1, :]
                target_v_cache = V_cache[19, :pos + 1, :]
            else:
                target_k_cache = K_cache[18, :pos + 1, :]
                target_v_cache = V_cache[18, :pos + 1, :]

        # --- Attention (fused C++ kernel: FP16 KV cache → FP32 output) ---
        attn_raw = CPU_CORE.cpu_gqa_fused(Q, target_k_cache, target_v_cache)

        if _profile_this: _PROF["attn"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        # --- O projection ---
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_gate"][i], next_buf)
        attn_output = hw_compute_pingpong(attn_raw, W["W_o"][i], curr_buf, out=_BUF_2048b)
        attn_output = attn_output.copy()
        ping_pong = next_buf

        if _profile_this: _PROF["o_proj"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        laurel_x          = hw_matmul(inputs_normalized, W["laurel_left"][i])
        laurel_x          = hw_matmul(laurel_x, W["laurel_right"][i])
        laurel_out_normed = inputs_normalized + rms_norm(laurel_x, W["laurel_norm"][i])

        attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
        attn_output += x
        attn_output  = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))

        if _profile_this: _PROF["laurel"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])

        # --- Gate, Up, Down projections ---
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_up"][i], next_buf)
        gate_out = hw_compute_pingpong(x_n2, W["W_gate"][i], curr_buf, use_gelu=(i >= 10), out=_BUF_16384)
        gate_out = gate_out.copy()
        ping_pong = next_buf

        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_down"][i], next_buf)
        up_out = hw_compute_pingpong(x_n2, W["W_up"][i], curr_buf, out=_BUF_16384b)
        up_out = up_out.copy()
        ping_pong = next_buf

        if i < 10:
            cutoff      = np.mean(gate_out) + np.std(gate_out) * 1.6448536
            sparse_gate = np.maximum(gate_out - cutoff, 0.0)
            activated   = CPU_CORE.gelu(sparse_gate)
            hidden      = activated * up_out
        else:
            hidden      = gate_out * up_out

        curr_buf = ping_pong; next_buf = 1 - ping_pong
        if i < NUM_LAYERS - 1:
            hw_prefetch(W["W_q"][i+1], next_buf)

        mlp_out = hw_compute_pingpong(hidden, W["W_down"][i], curr_buf, out=_BUF_2048)
        mlp_out = mlp_out.copy()
        ping_pong = next_buf

        outputs  = rms_norm(mlp_out, W["post_ffn_ln"][i])
        outputs += attn_output

        if _profile_this: _PROF["ffn"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        activated  = outputs * W["altup_scale"][i]
        innovation = activated - xs_pred[0]
        mod_corr   = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
        corr_coefs = np.dot(W["altup_corr"][i], mod_corr) + 1.0
        xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation

        if _profile_this: _PROF["altup_corr"] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        pli      = pli_all[i]
        gate_ple = CPU_CORE.gelu(hw_matmul(activated, W["ple_gate"][i])) * pli
        mapped   = rms_norm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
        xs_new[1:] += mapped
        xs = xs_new

        if _profile_this: _PROF["ple"] += _time.perf_counter() - _t0

    if _profile_this:
        forward_one_token._profiled = True
        total = sum(_PROF.values())
        print("\n\n[Profiler] Per-section time (35 layers total):")
        for k, v in sorted(_PROF.items(), key=lambda x: -x[1]):
            print(f"  {k:<15} {v*1000:7.1f} ms  ({v/total*100:4.1f}%)")
        print(f"  {'TOTAL':<15} {total*1000:7.1f} ms")

    return xs

def decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head):
    hw_prefetch(W_lm_head, 0)
    
    target_mag = np.mean(xs[0] ** 2) ** 0.5
    unembedded = [xs[0]]
    for k in range(3):
        proj_x  = np.dot(xs[k + 1], altup_unprojs[k])
        new_mag = np.mean(proj_x ** 2) ** 0.5
        proj_x *= target_mag / max(new_mag, 1e-12)
        unembedded.append(proj_x)
    x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
    x_final = rms_norm(x_final, W_final_norm)
    
    logits = hw_compute_pingpong(x_final, W_lm_head, buf_idx=0)
    return logits

def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rep_penalty: float, generated: list) -> int:
    
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty
                
    if temperature == 0.0: 
        return int(np.argmax(logits))

    if logits.dtype == np.float32 and logits.flags['C_CONTIGUOUS']:
        logits_f32 = logits
    else:
        logits_f32 = np.ascontiguousarray(logits.astype(np.float32))
    
    c_lib.run_softmax_inplace(logits_f32, logits_f32.size, float(temperature))
    probs = logits_f32

    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] < top_p
        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_idx[cutoff_mask]] = probs[sorted_idx[cutoff_mask]]
        if probs_filtered.sum() == 0: probs_filtered[sorted_idx[0]] = 1.0
        probs = probs_filtered / probs_filtered.sum()
        
    return int(np.random.choice(len(probs), p=probs))

def print_ram_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {rss_mb:.2f} MB")


def select_modes():
    global FEATURE_MODE, WEIGHT_MODE

    print("=" * 56)
    print("  Gemma 3N E4B — Inference Configuration")
    print("=" * 56)
    
    feature_options = {"1": "FP32", "2": "BF16", "3": "INT8", "4": "INT4"}
    print("\n  [Feature Map Mode] (activation precision)")
    print("    1) FP32  — Full precision (baseline, recommended)")
    print("    2) BF16  — BFloat16 (half bandwidth)")
    print("    3) INT8  — 8-bit quantized")
    print("    4) INT4  — 4-bit quantized (aggressive)")
    
    choice = input("  Select [1-4, default=1]: ").strip()
    FEATURE_MODE = feature_options.get(choice, "FP32")
    
    print(f"\n  [Weight Mode] (currently fixed: INT4)")
    WEIGHT_MODE = "INT4"
    
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Weight Mode:      {WEIGHT_MODE:>10}   │")
    print(f"  │  Feature Map Mode: {FEATURE_MODE:>10}   │")
    print(f"  │  Accelerator:      {ACCEL_MODE:>10}   │")
    print(f"  └─────────────────────────────────┘")
    print()


def main():
    select_modes()
    
    print_ram_usage("1. Before Model Load")
    
    TEMPERATURE    = 0.65
    TOP_P          = 0.9    
    REP_PENALTY    = 1.15
    MAX_NEW_TOKENS = 2048
    KV_CACHE_DIM   = 512

    FAST_MATRIX_CORE.warmup()
    print(f"\nGemma 3N [W:{WEIGHT_MODE} / A:{FEATURE_MODE}] - Chat Mode")
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    
    print("[Memory] Optimizing weighted VRAM...")
    FAST_MATRIX_CORE.preload_and_free(W, _IGPU_WEIGHT_KEYS)
    FAST_MATRIX_CORE._get_or_upload_weight(W_lm_head)

    print_ram_usage("2. After Model Load")

    K_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
    V_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)

    cur_pos = 0
    print_ram_usage("3. After KV Cache Allocation")

    print("\n--- Start conversation (end: 'exit' or 'quit') ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input.strip(): continue

            prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
            input_tokens = CPU_CORE.tokenize(prompt)
            
            print("Model: ", end="", flush=True)
            
            xs = None
            for token_id in input_tokens:
                xs = forward_one_token(token_id, cur_pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                                       W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
            
            print_ram_usage("4. After Prefill")
            
            generated = []
            STOP_TOKENS = [1, 106]
            printed_text = ""  

            for _ in range(MAX_NEW_TOKENS):
                logits = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
                logits = 30.0 * np.tanh(logits / 30.0)
                next_token = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
                
                if next_token in STOP_TOKENS: break
                generated.append(next_token)

                current_text = CPU_CORE.tokenizer.decode(generated, skip_special_tokens=True)
                new_text = current_text[len(printed_text):]
                print(new_text, end="", flush=True)
                printed_text = current_text

                xs = forward_one_token(next_token, cur_pos, W, W_embed, W_ple_packed, W_ple_scale,
                                       norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
                
            print()
            gc.collect()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    print_ram_usage("5. After Generation")
    print("\n[Complete] The conversation has ended.")

if __name__ == "__main__":
    main()
