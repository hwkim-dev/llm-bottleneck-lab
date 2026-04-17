import os
import sys
import resource

# mmap_weights 디렉토리에 2000+ 파일을 동시에 열기 때문에 fd 한도를 올림
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(_soft, 65536), max(_hard, 65536)))
except Exception:
    pass

import numpy as np
from flask import Flask, render_template, request, Response, jsonify
import json
import logging
import time
import threading

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

import main as gem_main
import CPU_CORE
import safeTensor

_init_lock = threading.Lock()

model_state = {
    "initialized": False,
    "initializing": False,
    "init_error": None,
    "K_cache": None,
    "V_cache": None,
    "cur_pos": 0,
    "weights": None,
    "init_time": None,
}

MAX_NEW_TOKENS = 2048
KV_CACHE_DIM = 512


def init_model():
    with _init_lock:
        if model_state["initialized"] or model_state["initializing"]:
            return
        model_state["initializing"] = True

    try:
        print("[GUI] Initializing Gemma 3N Model...")
        t0 = time.time()

        gem_main.select_modes = lambda: None
        gem_main.WEIGHT_MODE = "INT4"
        gem_main.FEATURE_MODE = "FP32"
        gem_main.ACCEL_MODE = "IGPU"

        gem_main.FAST_MATRIX_CORE.warmup()

        W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
            W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

        print("[GUI] Optimizing weighted VRAM...")
        gem_main.FAST_MATRIX_CORE.preload_and_free(W, gem_main._IGPU_WEIGHT_KEYS)
        gem_main.FAST_MATRIX_CORE._get_or_upload_weight(W_lm_head)

        NUM_LAYERS = gem_main.NUM_LAYERS
        K_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
        V_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)

        elapsed = time.time() - t0
        model_state.update({
            "initialized": True,
            "initializing": False,
            "init_error": None,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "cur_pos": 0,
            "init_time": elapsed,
            "weights": {
                "W": W,
                "W_embed": W_embed,
                "W_ple_packed": W_ple_packed,
                "W_ple_scale": W_ple_scale,
                "norm_ple": norm_ple,
                "W_ple_proj": W_ple_proj,
                "altup_projs": altup_projs,
                "altup_unprojs": altup_unprojs,
                "W_final_norm": W_final_norm,
                "W_lm_head": W_lm_head,
            },
        })
        print(f"[GUI] Model ready in {elapsed:.1f}s")

    except Exception as e:
        model_state["initializing"] = False
        model_state["init_error"] = str(e)
        print(f"[GUI] Init failed: {e}")
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "initialized": model_state["initialized"],
        "initializing": model_state["initializing"],
        "error": model_state["init_error"],
        "context_pos": model_state["cur_pos"],
        "init_time_sec": round(model_state["init_time"], 2) if model_state["init_time"] else None,
        "model": "Gemma 3N E4B",
        "weight_mode": "INT4",
        "accel": "IGPU",
    })


@app.route("/api/health")
def health():
    return jsonify({"ok": True}), 200


@app.route("/api/reset", methods=["POST"])
def reset():
    if not model_state["initialized"]:
        return jsonify({"error": "Model not initialized"}), 400

    NUM_LAYERS = gem_main.NUM_LAYERS
    model_state["K_cache"] = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
    model_state["V_cache"] = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
    model_state["cur_pos"] = 0
    return jsonify({"ok": True, "message": "Conversation context cleared"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = data.get("prompt", "").strip()
    if not user_input:
        return jsonify({"error": "Empty prompt"}), 400

    temperature = float(data.get("temperature", 0.65))
    top_p = float(data.get("top_p", 0.9))
    rep_penalty = float(data.get("rep_penalty", 1.15))
    max_tokens = min(int(data.get("max_tokens", 512)), MAX_NEW_TOKENS)

    def generate():
        if not model_state["initialized"]:
            try:
                init_model()
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        w = model_state["weights"]
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        input_tokens = CPU_CORE.tokenize(prompt)

        xs = None
        cur_pos = model_state["cur_pos"]

        for token_id in input_tokens:
            xs = gem_main.forward_one_token(
                token_id, cur_pos, w["W"], w["W_embed"], w["W_ple_packed"],
                w["W_ple_scale"], w["norm_ple"], w["W_ple_proj"],
                w["altup_projs"], model_state["K_cache"], model_state["V_cache"],
            )
            cur_pos += 1

        generated = []
        STOP_TOKENS = [1, 106]
        printed_text = ""
        token_count = 0

        for _ in range(max_tokens):
            logits = gem_main.decode_logits(xs, w["altup_unprojs"], w["W_final_norm"], w["W_lm_head"])
            logits = 30.0 * np.tanh(logits / 30.0)
            next_token = gem_main._sample(logits, temperature, top_p, rep_penalty, generated)

            if next_token in STOP_TOKENS:
                break

            generated.append(next_token)
            token_count += 1

            current_text = CPU_CORE.tokenizer.decode(generated, skip_special_tokens=True)
            new_text = current_text[len(printed_text):]
            printed_text = current_text

            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'count': token_count})}\n\n"

            xs = gem_main.forward_one_token(
                next_token, cur_pos, w["W"], w["W_embed"], w["W_ple_packed"],
                w["W_ple_scale"], w["norm_ple"], w["W_ple_proj"],
                w["altup_projs"], model_state["K_cache"], model_state["V_cache"],
            )
            cur_pos += 1

        model_state["cur_pos"] = cur_pos
        yield f"data: {json.dumps({'done': True, 'total_tokens': token_count, 'context_pos': cur_pos})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    init_model()
    print("\n" + "=" * 50)
    print("  Gemma 3N E4B  |  백엔드 실행 중")
    print("  (네이티브 GUI 대기 중...)")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
