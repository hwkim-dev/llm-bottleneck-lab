#!/usr/bin/env python3
"""Thread scaling: measure decode tok/s with different OMP thread counts.
Must be run separately for each thread count since OMP is set at process start.
Usage: OMP_NUM_THREADS=N python3 run_thread_scaling.py
"""
import sys, os, time, csv, gc
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

N_THREADS = os.environ.get("OMP_NUM_THREADS", "6")

import CPU_CORE
import safeTensor

try:
    import IGPU_CORE as FAST_MATRIX_CORE
except:
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE

from main import (forward_one_token, decode_logits, _sample,
                  GLOBAL_PROFILE_DATA, _IGPU_WEIGHT_KEYS, NUM_LAYERS,
                  _variant_dir_for_mode)

WEIGHT_MODE = "INT4"
KV_DIM = 512
TEMPERATURE = 0.65
TOP_P = 0.9
REP_PENALTY = 1.15

PROMPT = "What is on-device AI?"
MAX_OUT = 32
OUTPUT_CSV = "thread_scaling_results.csv"


def load_model():
    FAST_MATRIX_CORE.warmup()
    vdir = _variant_dir_for_mode(WEIGHT_MODE)
    if vdir is None:
        if os.path.isdir(safeTensor.mmap_dir):
            vdir = safeTensor.mmap_dir
        else:
            sys.exit("No weights")
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, \
        altup_projs, altup_unprojs, W_final_norm, W_lm_head, W = \
        safeTensor.load_local_weights(model_dir=vdir, mode="int4")
    FAST_MATRIX_CORE.preload_and_free(W, _IGPU_WEIGHT_KEYS)
    FAST_MATRIX_CORE._get_or_upload_weight(W_lm_head)
    return (W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
            W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head)


def run_once(prompt, max_tokens, weights):
    (W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
     W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head) = weights
    K = np.zeros((NUM_LAYERS, 2048, KV_DIM), dtype=np.float16)
    V = np.zeros((NUM_LAYERS, 2048, KV_DIM), dtype=np.float16)
    pos = 0
    full = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    tokens = CPU_CORE.tokenize(full)
    GLOBAL_PROFILE_DATA.clear()
    for tid in tokens:
        xs = forward_one_token(tid, pos, W, W_embed, W_ple_packed, W_ple_scale,
                               norm_ple, W_ple_proj, altup_projs, K, V)
        GLOBAL_PROFILE_DATA[-1]["stage"] = "Prefill"
        pos += 1
    generated = []
    for _ in range(max_tokens):
        logits = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
        logits = 30.0 * np.tanh(logits / 30.0)
        nt = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
        if nt in [1, 106]: break
        generated.append(nt)
        xs = forward_one_token(nt, pos, W, W_embed, W_ple_packed, W_ple_scale,
                               norm_ple, W_ple_proj, altup_projs, K, V)
        GLOBAL_PROFILE_DATA[-1]["stage"] = "Decode"
        pos += 1
    decode_rows = [r for r in GLOBAL_PROFILE_DATA if r.get("stage") == "Decode"]
    dc_ms = [r.get("_total", 0) * 1000 for r in decode_rows]
    ffn_ms = np.mean([r.get("ffn", 0) * 1000 for r in decode_rows]) if decode_rows else 0
    return {
        "threads": int(N_THREADS),
        "output_tokens": len(generated),
        "decode_mean_ms": round(np.mean(dc_ms), 1) if dc_ms else 0,
        "decode_tok_s": round(1000 / np.mean(dc_ms), 2) if dc_ms and np.mean(dc_ms) > 0 else 0,
        "ffn_ms": round(ffn_ms, 1),
    }


def main():
    print(f"Thread scaling: OMP_NUM_THREADS={N_THREADS}")
    weights = load_model()
    print("Warm-up...")
    run_once("Hi", 4, weights)
    gc.collect()

    print(f"Running with {N_THREADS} threads...")
    r = run_once(PROMPT, MAX_OUT, weights)
    print(f"  decode={r['decode_mean_ms']:.0f}ms/tok ({r['decode_tok_s']:.2f} tok/s) FFN={r['ffn_ms']:.0f}ms")

    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "output_tokens", "decode_mean_ms", "decode_tok_s", "ffn_ms"])
        if write_header:
            w.writeheader()
        w.writerow(r)
    print(f"Appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
