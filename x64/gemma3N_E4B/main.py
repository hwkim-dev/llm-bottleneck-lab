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
import time

# ================================================================
# ██████ Continuous Profiling Configuration ██████
# ================================================================
GLOBAL_PROFILE_DATA = []

def generate_profile_html():
    if not GLOBAL_PROFILE_DATA: return
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family: Arial, sans-serif; background:#121212; color:#e0e0e0; padding:20px;}",
        "h2{color:#4fc3f7; text-align:center;}",
        "table{border-collapse: collapse; margin:0 auto; font-size:14px;}",
        "th,td{border: 1px solid #333; padding: 6px 12px; text-align: right;}",
        "th{background: #1f1f1f; text-align: center; color:#b39ddb; position: sticky; top:0;}",
        "tr:hover{filter: brightness(1.2);}",
        ".prefill{background: #111a22;}",
        ".decode{background: #182318;}",
        ".hl{color: #ffd54f; font-weight:bold;}",
        "</style></head><body>",
        f"<h2>Gemma 3N Inference Operation Profiler</h2>",
        "<table><tr><th>#</th><th>Stage</th>"
    ]
    keys = ["ffn", "o_proj", "qkv", "ple", "laurel", "altup_corr", "altup_pred", "qk_rope", "attn"]
    for k in keys: html.append(f"<th>{k}</th>")
    html.append("<th>TOTAL</th><th>Tokens/sec</th></tr>")
    
    total_time_sum = 0
    for i, row in enumerate(GLOBAL_PROFILE_DATA):
        stage = row.get("stage", "Unknown")
        cls = "prefill" if stage == "Prefill" else "decode"
        html.append(f"<tr class='{cls}'><td>{i}</td><td style='text-align:center;'>{stage}</td>")
        
        row_time = row.get("_total", 0)
        total_time_sum += row_time
        
        for k in keys:
            html.append(f"<td>{row.get(k, 0)*1000:.1f}</td>")
        
        tps = 1.0 / row_time if row_time > 0 else 0
        html.append(f"<td class='hl'>{row_time*1000:.1f} ms</td><td class='hl'>{tps:.1f}</td></tr>")
        
    html.append(f"<tr><td colspan='{len(keys)+2}' style='text-align:center'><b>Cumulative Time: {(total_time_sum):.2f} sec</b></td><td colspan='2'></td></tr>")
    html.append("</table></body></html>")
    
    with open("ProfilerReport.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("\n[Profiler] 📊 Saved detailed HTML report to 'ProfilerReport.html'")

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


def _cpu_matmul_from_tuple(x, w):
    """CPU fallback: decode an (int4-packed | int8, scale) tuple and matmul with x."""
    packed, scale = w
    if packed.dtype == np.uint8:
        low = (packed & 0x0F).astype(np.int8)
        low[low > 7] -= 16
        high = (packed >> 4).astype(np.int8)
        high[high > 7] -= 16
        res = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        res[:, 0::2] = low
        res[:, 1::2] = high
        w_real = res * scale[:, np.newaxis]
    else:  # int8
        w_real = packed.astype(np.float32) * scale[:, np.newaxis]
    return np.dot(x, w_real.T)


def hw_matmul(x, w, use_gelu=False):
    # Vulkan GEMV only supports INT4 packed (uint8) weights.  Everything
    # else runs on the CPU path for now — see IGPU_CORE.py for the TODO.
    if isinstance(w, tuple):
        packed, _scale = w
        if ACCEL_MODE == "IGPU" and packed.dtype == np.uint8:
            return (FAST_MATRIX_CORE.igpu_matmul_gelu(x, w) if use_gelu
                    else FAST_MATRIX_CORE.igpu_matmul(x, w))
        out = _cpu_matmul_from_tuple(x, w)
    else:
        # Raw FP16 / FP32 weight stored pre-transposed as [K_in, M_out].
        if ACCEL_MODE == "IGPU":
            out = FAST_MATRIX_CORE.igpu_matmul(x, w)
        else:
            out = np.dot(x, w)
    return CPU_CORE.gelu(out) if use_gelu else out


def hw_prefetch(w, buf_idx):
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple) and w[0].dtype == np.uint8:
        FAST_MATRIX_CORE.prefetch_weight(w, buf_idx)


def hw_compute_pingpong(x, w, buf_idx, use_gelu=False, out=None):
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple) and w[0].dtype == np.uint8:
        result = FAST_MATRIX_CORE.compute_pingpong(x, w, buf_idx, out=out)
        if result.shape[0] == 16384 and w[0].shape[0] == 8192:
            print("WTF? IGPU returned 16384 for an 8192 weight matrix!")
        return CPU_CORE.gelu(result) if use_gelu else result
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

def forward_draft_one_token(token_id, pos, W, W_embed, W_ple_packed, W_ple_scale,
                            norm_ple, W_ple_proj, altup_projs, K_cache, V_cache,
                            num_draft_layers: int = 17, W_draft=None):
    """Layer-skip self-speculative draft or E2B MatFormer draft.

    Quality caveat: without explicit LayerSkip training the early-exit logits
    are noisy. Using a MatFormer E2B sliced draft (via W_draft) significantly
    improves the acceptance rate.
    """
    global NUM_LAYERS
    saved = NUM_LAYERS
    try:
        NUM_LAYERS = max(1, min(num_draft_layers, saved))
        active_W = W_draft if W_draft is not None else W
        return forward_one_token(token_id, pos, active_W, W_embed, W_ple_packed, W_ple_scale,
                                 norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
    finally:
        NUM_LAYERS = saved


def _embed_row(token_id, w):
    """Row-wise embed lookup across INT4 / INT8 / FP16 / FP32 weight formats."""
    if isinstance(w, tuple):
        packed, scale = w
        if packed.dtype == np.uint8:
            return CPU_CORE.embedding(token_id, packed, scale)
        return packed[token_id].astype(np.float32) * scale[token_id]
    return w[token_id].astype(np.float32)


def _embed_row_split(token_id, packed, scale):
    """As above but with pre-split packed/scale (W_ple convention). scale may be None."""
    if scale is None:
        return packed[token_id].astype(np.float32)
    if packed.dtype == np.uint8:
        return CPU_CORE.embedding(token_id, packed, scale)
    return packed[token_id].astype(np.float32) * scale[token_id]


def forward_one_token(token_id, pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    safe_token_id = int(min(token_id, W_ple_packed.shape[0] - 1))
    x0 = _embed_row(safe_token_id, W_embed)
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

    unpacked_w_ple = _embed_row_split(safe_token_id, W_ple_packed, W_ple_scale)
    y = unpacked_w_ple.reshape(35, 256) * math.sqrt(256.0)
    pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

    # ── Continuous Profiler ──────────
    _PROF = {"altup_pred": 0.0, "qkv": 0.0, "qk_rope": 0.0,
             "attn": 0.0, "o_proj": 0.0, "laurel": 0.0,
             "ffn": 0.0, "altup_corr": 0.0, "ple": 0.0}

    ping_pong = 0
    hw_prefetch(W["W_q"][0], ping_pong)

    for i in range(NUM_LAYERS):
        _t0 = time.perf_counter()

        modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
        coef_mat    = np.dot(W["altup_pred"][i], modalities).reshape(4, 4)
        xs_pred     = xs + np.dot(coef_mat, xs)

        _PROF["altup_pred"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

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

        _PROF["qkv"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        # --- QK Norm + RoPE (fused C++) ---
        theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
        Q, K = CPU_CORE.cpu_qk_norm_rope_fused(Q, K, W["gamma_q"][i], W["gamma_k"][i], pos, theta)

        _PROF["qk_rope"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

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

        _PROF["attn"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        # --- O projection ---
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_gate"][i], next_buf)
        attn_output = hw_compute_pingpong(attn_raw, W["W_o"][i], curr_buf, out=_BUF_2048b)
        attn_output = attn_output.copy()
        ping_pong = next_buf

        _PROF["o_proj"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        laurel_x          = hw_matmul(inputs_normalized, W["laurel_left"][i])
        laurel_x          = hw_matmul(laurel_x, W["laurel_right"][i])
        laurel_out_normed = inputs_normalized + rms_norm(laurel_x, W["laurel_norm"][i])

        attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
        attn_output += x
        attn_output  = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))

        _PROF["laurel"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])

        # --- Gate, Up, Down projections ---
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_up"][i], next_buf)
        gate_out = hw_compute_pingpong(x_n2, W["W_gate"][i], curr_buf, use_gelu=(i >= 10), out=None)
        gate_out = gate_out.copy()
        ping_pong = next_buf

        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_down"][i], next_buf)
        up_out = hw_compute_pingpong(x_n2, W["W_up"][i], curr_buf, out=None)
        up_out = up_out.copy()
        ping_pong = next_buf

        if gate_out.shape != up_out.shape:
            print(f"DEBUG LAYER {i} | gate_out.shape: {gate_out.shape} | up_out.shape: {up_out.shape}")
            print(f"W_gate shape: {W['W_gate'][i][0].shape if isinstance(W['W_gate'][i], tuple) else W['W_gate'][i].shape}")
            print(f"W_up shape: {W['W_up'][i][0].shape if isinstance(W['W_up'][i], tuple) else W['W_up'][i].shape}")

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

        _PROF["ffn"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        activated  = outputs * W["altup_scale"][i]
        innovation = activated - xs_pred[0]
        mod_corr   = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
        corr_coefs = np.dot(W["altup_corr"][i], mod_corr) + 1.0
        xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation

        _PROF["altup_corr"] += time.perf_counter() - _t0; _t0 = time.perf_counter()

        pli      = pli_all[i]
        gate_ple = CPU_CORE.gelu(hw_matmul(activated, W["ple_gate"][i])) * pli
        mapped   = rms_norm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
        xs_new[1:] += mapped
        xs = xs_new

        _PROF["ple"] += time.perf_counter() - _t0

    _PROF["_total"] = sum(_PROF.values())
    GLOBAL_PROFILE_DATA.append(_PROF)

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

    installed = {v["mode"] for v in safeTensor.list_variants()}
    if not installed:
        print("\n  ⚠ No weight variants installed under models/.")
        print("    Convert an HF snapshot first:")
        print("      python quantize.py --mode int4 --src /path/to/hf-gemma-3n-E4B-it\n")

    weight_menu = [("1", "INT4", "4-bit packed, Vulkan-accelerated"),
                   ("2", "INT8", "8-bit, CPU matmul"),
                   ("3", "FP16", "half precision, CPU matmul"),
                   ("4", "FP32", "full precision baseline")]
    print("\n  [Weight Mode]")
    for key, mode, desc in weight_menu:
        marker = "✓" if mode.lower() in installed else " "
        print(f"    {key}) [{marker}] {mode:4}  — {desc}")
    choice = input("  Select [1-4, default=1]: ").strip()
    WEIGHT_MODE = {"1": "INT4", "2": "INT8", "3": "FP16", "4": "FP32"}.get(choice, "INT4")

    feature_options = {"1": "FP32", "2": "BF16", "3": "INT8", "4": "INT4"}
    print("\n  [Feature Map Mode] (activation precision)")
    print("    1) FP32  — Full precision (baseline, recommended)")
    print("    2) BF16  — BFloat16 (half bandwidth)")
    print("    3) INT8  — 8-bit quantized")
    print("    4) INT4  — 4-bit quantized (aggressive)")
    choice = input("  Select [1-4, default=1]: ").strip()
    FEATURE_MODE = feature_options.get(choice, "FP32")

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Weight Mode:      {WEIGHT_MODE:>10}   │")
    print(f"  │  Feature Map Mode: {FEATURE_MODE:>10}   │")
    print(f"  │  Accelerator:      {ACCEL_MODE:>10}   │")
    print(f"  └─────────────────────────────────┘")
    print()


def _variant_dir_for_mode(mode: str, model_name: str = "gemma-3n-e4b"):
    """Return an existing model directory matching `mode`, else None."""
    candidate = os.path.join(safeTensor.models_dir, f"{model_name}-{mode.lower()}")
    if os.path.isdir(candidate) and any(
        f.endswith(".npy") for f in os.listdir(candidate)
    ):
        return candidate
    return None


def _any_weights_available(mode: str) -> bool:
    """Return True iff SOME usable weights exist for this mode."""
    if _variant_dir_for_mode(mode) is not None:
        return True
    # legacy fallback: mmap_weights/ works for int4 only
    return (mode.lower() == "int4"
            and os.path.isdir(safeTensor.mmap_dir)
            and any(f.endswith(".npy") for f in os.listdir(safeTensor.mmap_dir)))


def _prompt_download_and_quantize(mode: str) -> bool:
    """Interactive: offer to download + quantize weights when none are installed.

    Returns True on success (weights now available), False if user declined or it failed.
    """
    print()
    print(f"  ✗ No weights found for WEIGHT_MODE={mode.upper()}.")
    print(f"    Expected one of:")
    print(f"      • models/gemma-3n-e4b-{mode.lower()}/")
    if mode.lower() == "int4":
        print(f"      • mmap_weights/   (legacy INT4)")
    answer = input("\n  Download google/gemma-3n-E4B-it from HuggingFace and quantize now? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("  Aborted.  Install manually via:")
        print(f"    python quantize.py --mode {mode.lower()} --hf-id google/gemma-3n-E4B-it")
        return False
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub is not installed.  Run: pip install huggingface_hub")
        return False

    token = input("  HF token (press Enter if you already ran `huggingface-cli login`): ").strip()

    hf_id = "google/gemma-3n-E4B-it"
    cache_dir = os.path.join(os.path.dirname(safeTensor.models_dir),
                             "hf_cache", hf_id.replace("/", "__"))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n  → snapshot_download({hf_id}) into {cache_dir}")
    try:
        snapshot_download(
            repo_id=hf_id, local_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            **({"token": token} if token else {}),
        )
    except Exception as e:
        msg = str(e)
        print(f"\n  ✗ Download failed: {type(e).__name__}: {msg[:200]}")
        if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            print("    This model is gated — visit https://huggingface.co/" + hf_id)
            print("    to accept the license, then get a token from")
            print("    https://huggingface.co/settings/tokens")
        return False

    import quantize as _quantize
    dst = os.path.join(safeTensor.models_dir, f"gemma-3n-e4b-{mode.lower()}")
    print(f"  → quantize --mode {mode.lower()} → {dst}")
    try:
        _quantize.convert(cache_dir, dst, mode.lower())
    except Exception as e:
        print(f"\n  ✗ Quantize failed: {type(e).__name__}: {e}")
        return False
    print("\n  ✓ Weights ready.")
    return True


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

    if not _any_weights_available(WEIGHT_MODE):
        if not _prompt_download_and_quantize(WEIGHT_MODE):
            print("  Exiting.")
            sys.exit(1)

    variant_dir = _variant_dir_for_mode(WEIGHT_MODE)
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights(
            model_dir=variant_dir, mode=WEIGHT_MODE.lower())
    
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
            GLOBAL_PROFILE_DATA.clear()
            
            for token_id in input_tokens:
                xs = forward_one_token(token_id, cur_pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                                       W_ple_proj, altup_projs, K_cache, V_cache)
                GLOBAL_PROFILE_DATA[-1]["stage"] = "Prefill"
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
                GLOBAL_PROFILE_DATA[-1]["stage"] = "Decode"
                cur_pos += 1
                
            print()
            generate_profile_html()
            gc.collect()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    print_ram_usage("5. After Generation")
    print("\n[Complete] The conversation has ended.")

if __name__ == "__main__":
    main()

