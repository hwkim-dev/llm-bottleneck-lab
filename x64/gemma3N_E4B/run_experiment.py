#!/usr/bin/env python3
"""Automated experiment runner for paper measurements.
Loads model once, runs 5x repeat, saves CSV.
"""
import sys, os, time, csv, gc, shutil
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

os.environ.setdefault("OMP_NUM_THREADS", "6")

import CPU_CORE
import safeTensor

WEIGHT_MODE = "INT4"
FEATURE_MODE = "FP32"

try:
    import IGPU_CORE as FAST_MATRIX_CORE
    ACCEL = "IGPU"
except:
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
    ACCEL = "CPU"

from main import (forward_one_token, decode_logits, _sample,
                  GLOBAL_PROFILE_DATA, generate_profile_html,
                  _IGPU_WEIGHT_KEYS, NUM_LAYERS,
                  _variant_dir_for_mode, _any_weights_available)

PROMPT = "What is on-device AI and why is it important?"
MAX_NEW = 64
KV_DIM = 512
TEMPERATURE = 0.65
TOP_P = 0.9
REP_PENALTY = 1.15
N_RUNS = 5

OUTPUT_CSV = "experiment_results.csv"
PROFILE_DIR = "experiment_profiles"


def load_model():
    FAST_MATRIX_CORE.warmup()
    vdir = _variant_dir_for_mode(WEIGHT_MODE)
    if vdir is None:
        if WEIGHT_MODE.lower() == "int4" and os.path.isdir(safeTensor.mmap_dir):
            vdir = safeTensor.mmap_dir
        else:
            print("ERROR: No weights found for", WEIGHT_MODE)
            sys.exit(1)
    print(f"Loading weights from {vdir} ...")
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, \
        altup_projs, altup_unprojs, W_final_norm, W_lm_head, W = \
        safeTensor.load_local_weights(model_dir=vdir, mode=WEIGHT_MODE.lower())
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
        if nt in [1, 106]:
            break
        generated.append(nt)
        xs = forward_one_token(nt, pos, W, W_embed, W_ple_packed, W_ple_scale,
                               norm_ple, W_ple_proj, altup_projs, K, V)
        GLOBAL_PROFILE_DATA[-1]["stage"] = "Decode"
        pos += 1

    prefill = [r for r in GLOBAL_PROFILE_DATA if r.get("stage") == "Prefill"]
    decode = [r for r in GLOBAL_PROFILE_DATA if r.get("stage") == "Decode"]

    pf_ms = sum(r.get("_total", 0) for r in prefill) * 1000
    dc_ms = [r.get("_total", 0) * 1000 for r in decode]

    keys = ["ffn", "o_proj", "qkv", "attn", "ple"]
    bk = {}
    for k in keys:
        vals = [r.get(k, 0) * 1000 for r in decode]
        bk[k] = np.mean(vals) if vals else 0

    return {
        "input_tokens": len(tokens),
        "output_tokens": len(generated),
        "prefill_ms": round(pf_ms, 1),
        "decode_mean_ms": round(np.mean(dc_ms), 1) if dc_ms else 0,
        "decode_std_ms": round(np.std(dc_ms), 1) if dc_ms else 0,
        "tok_s": round(1000 / np.mean(dc_ms), 2) if dc_ms and np.mean(dc_ms) > 0 else 0,
        "ffn_ms": round(bk["ffn"], 1),
        "qkv_ms": round(bk["qkv"], 1),
        "oproj_ms": round(bk["o_proj"], 1),
        "attn_ms": round(bk["attn"], 1),
        "ple_ms": round(bk["ple"], 1),
    }


def main():
    print("=" * 50)
    print("  llm-lite Experiment Runner")
    print(f"  Mode: W={WEIGHT_MODE} A={FEATURE_MODE} Accel={ACCEL}")
    print(f"  Runs: {N_RUNS}, MaxTokens: {MAX_NEW}")
    print("=" * 50)

    os.makedirs(PROFILE_DIR, exist_ok=True)

    t0 = time.time()
    weights = load_model()
    print(f"Model loaded in {time.time()-t0:.0f}s\n")

    results = []
    for i in range(N_RUNS):
        print(f"--- Run {i+1}/{N_RUNS} ---")
        t1 = time.time()
        r = run_once(PROMPT, MAX_NEW, weights)
        r["run"] = i + 1
        r["elapsed_s"] = round(time.time() - t1, 1)
        results.append(r)

        generate_profile_html()
        if os.path.exists("ProfilerReport.html"):
            shutil.copy2("ProfilerReport.html",
                         os.path.join(PROFILE_DIR, f"run{i+1}_profile.html"))

        print(f"  in={r['input_tokens']} out={r['output_tokens']} "
              f"decode={r['decode_mean_ms']:.0f}ms/tok "
              f"({r['tok_s']:.2f} tok/s) "
              f"FFN={r['ffn_ms']:.0f} attn={r['attn_ms']:.0f} "
              f"[{r['elapsed_s']}s]")
        gc.collect()

    fields = ["run", "input_tokens", "output_tokens", "prefill_ms",
              "decode_mean_ms", "decode_std_ms", "tok_s",
              "ffn_ms", "qkv_ms", "oproj_ms", "attn_ms", "ple_ms", "elapsed_s"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*50}")
    print(f"  DONE — {OUTPUT_CSV}")

    means = {k: np.mean([r[k] for r in results])
             for k in ["decode_mean_ms", "tok_s", "ffn_ms", "attn_ms"]}
    stds = {k: np.std([r[k] for r in results])
            for k in ["decode_mean_ms", "tok_s"]}

    print(f"  Decode: {means['decode_mean_ms']:.0f} +/- {stds['decode_mean_ms']:.0f} ms/tok")
    print(f"  tok/s:  {means['tok_s']:.2f} +/- {stds['tok_s']:.2f}")
    print(f"  FFN:    {means['ffn_ms']:.0f} ms  Attn: {means['attn_ms']:.0f} ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
