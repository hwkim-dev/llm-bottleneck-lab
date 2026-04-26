#!/usr/bin/env python3
"""Context length sweep: measure how input length affects prefill/decode."""
import sys, os, time, csv, gc, shutil
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import CPU_CORE
import safeTensor

try:
    import IGPU_CORE as FAST_MATRIX_CORE
except:
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE

from main import (forward_one_token, decode_logits, _sample,
                  GLOBAL_PROFILE_DATA, generate_profile_html,
                  _IGPU_WEIGHT_KEYS, NUM_LAYERS,
                  _variant_dir_for_mode)

WEIGHT_MODE = "INT4"
KV_DIM = 512
TEMPERATURE = 0.65
TOP_P = 0.9
REP_PENALTY = 1.15
MAX_OUT = 32

SHORT = "What is AI?"
MEDIUM = ("Explain the architectural differences between cloud-based large language model "
          "inference and on-device small language model inference, covering aspects such as "
          "memory bandwidth, KV-cache management, quantization techniques, and NPU scheduling. "
          "Discuss how prefill and decode stages differ in their computational characteristics. ") * 3
LONG = MEDIUM * 4

EXPERIMENTS = [
    {"name": "short", "prompt": SHORT, "label": "short"},
    {"name": "medium", "prompt": MEDIUM, "label": "medium"},
    {"name": "long", "prompt": LONG, "label": "long"},
]

OUTPUT_CSV = "context_sweep_results.csv"


def load_model():
    FAST_MATRIX_CORE.warmup()
    vdir = _variant_dir_for_mode(WEIGHT_MODE)
    if vdir is None:
        if WEIGHT_MODE.lower() == "int4" and os.path.isdir(safeTensor.mmap_dir):
            vdir = safeTensor.mmap_dir
        else:
            sys.exit("No weights")
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
    n_input = len(tokens)

    GLOBAL_PROFILE_DATA.clear()

    t_pf_start = time.time()
    for tid in tokens:
        xs = forward_one_token(tid, pos, W, W_embed, W_ple_packed, W_ple_scale,
                               norm_ple, W_ple_proj, altup_projs, K, V)
        GLOBAL_PROFILE_DATA[-1]["stage"] = "Prefill"
        pos += 1
    t_pf = (time.time() - t_pf_start) * 1000

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

    decode_rows = [r for r in GLOBAL_PROFILE_DATA if r.get("stage") == "Decode"]
    dc_ms = [r.get("_total", 0) * 1000 for r in decode_rows]

    keys = ["ffn", "o_proj", "qkv", "attn", "ple"]
    bk = {}
    for k in keys:
        vals = [r.get(k, 0) * 1000 for r in decode_rows]
        bk[k] = np.mean(vals) if vals else 0

    dc_total = sum(bk.values())
    return {
        "input_tokens": n_input,
        "output_tokens": len(generated),
        "prefill_ms": round(t_pf, 1),
        "prefill_tok_s": round(n_input / (t_pf / 1000), 1) if t_pf > 0 else 0,
        "decode_mean_ms": round(np.mean(dc_ms), 1) if dc_ms else 0,
        "decode_tok_s": round(1000 / np.mean(dc_ms), 2) if dc_ms and np.mean(dc_ms) > 0 else 0,
        "ffn_pct": round(bk["ffn"] / dc_total * 100, 1) if dc_total > 0 else 0,
        "attn_pct": round(bk["attn"] / dc_total * 100, 1) if dc_total > 0 else 0,
        "ffn_ms": round(bk["ffn"], 1),
        "attn_ms": round(bk["attn"], 1),
    }


def main():
    print("=" * 50)
    print("  Context Length Sweep")
    print("=" * 50)

    t0 = time.time()
    weights = load_model()
    print(f"Model loaded in {time.time()-t0:.0f}s\n")

    # warm-up run
    print("--- Warm-up run ---")
    run_once("Hello", 8, weights)
    gc.collect()
    print("  done\n")

    results = []
    for exp in EXPERIMENTS:
        print(f"--- {exp['name']} ---")
        t1 = time.time()
        r = run_once(exp["prompt"], MAX_OUT, weights)
        r["label"] = exp["label"]
        r["elapsed_s"] = round(time.time() - t1, 1)
        results.append(r)

        print(f"  input={r['input_tokens']} output={r['output_tokens']} "
              f"prefill={r['prefill_ms']:.0f}ms ({r['prefill_tok_s']:.1f} tok/s) "
              f"decode={r['decode_mean_ms']:.0f}ms/tok ({r['decode_tok_s']:.2f} tok/s) "
              f"FFN={r['ffn_pct']:.0f}% attn={r['attn_pct']:.0f}%")
        gc.collect()

    fields = ["label", "input_tokens", "output_tokens", "prefill_ms", "prefill_tok_s",
              "decode_mean_ms", "decode_tok_s", "ffn_pct", "attn_pct", "ffn_ms", "attn_ms", "elapsed_s"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*50}")
    print(f"  DONE — {OUTPUT_CSV}")
    for r in results:
        print(f"  [{r['label']:6s}] in={r['input_tokens']:4d}  "
              f"prefill={r['prefill_ms']:8.0f}ms  "
              f"decode={r['decode_mean_ms']:6.0f}ms/tok  "
              f"FFN={r['ffn_pct']:.0f}%  attn={r['attn_pct']:.0f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
