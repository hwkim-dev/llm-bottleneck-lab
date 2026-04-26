import os
import sys
import resource

# Bump fd ceiling — the MMAP loader opens 2000+ weight files simultaneously.
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
import shutil
import subprocess
import threading
import uuid

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

import main as gem_main
import CPU_CORE
import safeTensor
import spec_decode

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

# ── Generation cancel flag (shared between /api/chat and /api/chat/stop) ──
_generation_cancel = threading.Event()

# ── Engine configuration (applied on next init_model / reload) ──
ENGINE_CONFIG = {
    "cache_embeddings": False,
    "use_spec_decode": False,
    "spec_gamma": 4,
    "spec_draft_variant": None,   # e.g. "gemma-3n-e2b-int4"
    "use_streaming_llm": False,   # Attention Sinks for infinite context
}

# ── Model manager + GPU info state ──────────────────────────────
_download_jobs = {}       # job_id -> {status, percent, message, mode, hf_id}
_download_lock = threading.Lock()
_gpu_info_cache = None    # lazy-initialized {device, vendor, fp16_faster, raw}

_OLD_IGPU_HINTS = (
    "vega", "renoir", "cezanne", "lucienne",   # AMD Ryzen 4000/5000 APUs
    "hd graphics", "uhd graphics",             # Intel HD/UHD (Gen ≤ 9)
    "videocore",                                # Raspberry Pi
)


def _sanitize_variant_name(name: str) -> str:
    """Reject anything that could traverse out of models/."""
    if not name or "/" in name or ".." in name or name.startswith("."):
        raise ValueError(f"Invalid variant name: {name!r}")
    return name


def _detect_gpu_info():
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache
    device = "unknown"
    vendor = "unknown"
    raw = ""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=5,
        )
        raw = result.stdout
        for line in raw.splitlines():
            line_low = line.strip().lower()
            if "devicename" in line_low.replace(" ", ""):
                device = line.split("=", 1)[-1].strip() if "=" in line else line.split(":", 1)[-1].strip()
                break
            if line_low.startswith("devicename") and "=" in line:
                device = line.split("=", 1)[1].strip()
                break
    except Exception as e:
        raw = f"(vulkaninfo failed: {e})"
    device_low = device.lower()
    if "amd" in device_low or "radeon" in device_low:
        vendor = "AMD"
    elif "intel" in device_low:
        vendor = "Intel"
    elif "nvidia" in device_low or "geforce" in device_low:
        vendor = "NVIDIA"
    fp16_faster = any(h in device_low for h in _OLD_IGPU_HINTS)
    _gpu_info_cache = {
        "device": device,
        "vendor": vendor,
        "fp16_faster": fp16_faster,
        "raw": raw[:400],
    }
    return _gpu_info_cache


@app.route("/api/models/list")
def models_list():
    return jsonify({"variants": safeTensor.list_variants()})


@app.route("/api/models/delete", methods=["POST"])
def models_delete():
    data = request.json or {}
    try:
        name = _sanitize_variant_name(data.get("name", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    target = os.path.join(safeTensor.models_dir, name)
    if not os.path.isdir(target):
        return jsonify({"error": f"variant '{name}' not found"}), 404
    try:
        shutil.rmtree(target)
    except OSError as e:
        return jsonify({"error": f"rmtree failed: {e}"}), 500
    return jsonify({"ok": True, "deleted": name})


def _hf_cache_root() -> str:
    return os.path.join(os.path.dirname(safeTensor.models_dir), "hf_cache")


def _app_root() -> str:
    return os.path.dirname(safeTensor.models_dir)


def _legacy_reference_dirs():
    """Pre-existing model directories that predate the hf_cache convention.

    These are safe to *read* from (to re-quantize) but the UI disallows
    deletion — user can always remove them manually.
    """
    root = _app_root()
    candidates = [
        ("local_gemma_3n_int4", "google/gemma-3n-E4B-it (local, quantized INT4 snapshot)"),
        ("local_gemma_3n",      "google/gemma-3n-E4B-it (local, FP snapshot)"),
        ("local_gemma_3n_fp16", "google/gemma-3n-E4B-it (local, FP16 snapshot)"),
    ]
    out = []
    for sub, label in candidates:
        p = os.path.join(root, sub)
        if os.path.isdir(p):
            out.append((sub, p, label))
    return out


def _list_references():
    """Scan hf_cache/ + any legacy local_gemma_3n_* directory for references."""
    import glob as _glob
    refs = []

    # (1) hf_cache/<repo__name>/
    hf_root = _hf_cache_root()
    if os.path.isdir(hf_root):
        for name in sorted(os.listdir(hf_root)):
            path = os.path.join(hf_root, name)
            if not os.path.isdir(path):
                continue
            safetensors = _glob.glob(os.path.join(path, "*.safetensors"))
            if not safetensors:
                continue
            size = sum(os.path.getsize(p) for p in safetensors)
            refs.append({
                "name": name,
                "repo_id": name.replace("__", "/"),
                "path": path,
                "size_bytes": size,
                "num_shards": len(safetensors),
                "origin": "hf_cache",
                "deletable": True,
            })

    # (2) local legacy dirs
    for sub, path, label in _legacy_reference_dirs():
        safetensors = _glob.glob(os.path.join(path, "*.safetensors"))
        if not safetensors:
            continue
        size = sum(os.path.getsize(p) for p in safetensors)
        refs.append({
            "name": sub,
            "repo_id": label,
            "path": path,
            "size_bytes": size,
            "num_shards": len(safetensors),
            "origin": "legacy",
            "deletable": False,
        })

    return refs


def _resolve_reference_path(ref_name: str):
    """Map a UI-provided reference name → absolute path (or None)."""
    for r in _list_references():
        if r["name"] == ref_name:
            return r
    return None


def _hf_auth_hint(hf_id: str) -> str:
    return (
        f"'{hf_id}' is a gated model on HuggingFace. To access it:\n"
        f"  1. Visit https://huggingface.co/{hf_id} and accept the license.\n"
        f"  2. Get a token from https://huggingface.co/settings/tokens (read access is enough).\n"
        f"  3. Either paste the token into the HF token field in this UI, or run\n"
        f"     'huggingface-cli login' in the shell, then retry."
    )


def _run_ref_download(job_id: str, hf_id: str, hf_token: str = ""):
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        GatedRepoError = RepositoryNotFoundError = Exception  # older huggingface_hub
    state = _download_jobs[job_id]
    try:
        state["status"] = "downloading"
        state["message"] = f"Fetching {hf_id} from HuggingFace..."
        cache_dir = os.path.join(_hf_cache_root(), hf_id.replace("/", "__"))
        os.makedirs(cache_dir, exist_ok=True)
        kw = {}
        if hf_token:
            kw["token"] = hf_token
        snapshot_download(
            repo_id=hf_id,
            local_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            **kw,
        )
        state["status"] = "done"
        state["percent"] = 100
        state["message"] = f"Reference ready: {os.path.basename(cache_dir)}"
    except GatedRepoError:
        state["status"] = "error"
        state["message"] = _hf_auth_hint(hf_id)
    except RepositoryNotFoundError:
        state["status"] = "error"
        state["message"] = f"Repo '{hf_id}' not found on HuggingFace (or you need auth)."
    except Exception as e:
        msg = str(e)
        state["status"] = "error"
        if "401" in msg or "Cannot access gated" in msg or "restricted" in msg.lower():
            state["message"] = _hf_auth_hint(hf_id)
        else:
            state["message"] = f"{type(e).__name__}: {msg[:300]}"


def _run_quantize_job(job_id: str, ref_name: str, mode: str):
    state = _download_jobs[job_id]
    try:
        ref = _resolve_reference_path(ref_name)
        if ref is None:
            raise FileNotFoundError(f"reference not found: {ref_name}")
        src = ref["path"]
        import quantize
        model_name = quantize.infer_model_name(ref.get("repo_id") or ref_name)
        dst_name = f"{model_name}-{mode}"
        dst = os.path.join(safeTensor.models_dir, dst_name)
        state["status"] = "quantizing"
        state["message"] = f"Quantizing {ref_name} → {dst_name} (from {src})..."
        quantize.convert(src, dst, mode)
        state["status"] = "done"
        state["percent"] = 100
        state["message"] = f"Installed {dst_name}"
    except Exception as e:
        state["status"] = "error"
        state["message"] = f"{type(e).__name__}: {e}"


def _run_download_job(job_id: str, hf_id: str, mode: str):
    """Legacy combined: download + quantize in one job (kept for back-compat)."""
    from huggingface_hub import snapshot_download
    state = _download_jobs[job_id]
    try:
        state["status"] = "downloading"
        state["message"] = f"Downloading {hf_id}..."
        cache_dir = os.path.join(_hf_cache_root(), hf_id.replace("/", "__"))
        os.makedirs(cache_dir, exist_ok=True)
        snapshot_download(
            repo_id=hf_id, local_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )
        state["status"] = "quantizing"
        state["message"] = f"Quantizing to {mode}..."
        dst = os.path.join(safeTensor.models_dir, f"gemma-3n-e4b-{mode}")
        import quantize
        quantize.convert(cache_dir, dst, mode)
        state["status"] = "done"
        state["percent"] = 100
        state["message"] = f"Installed {os.path.basename(dst)}"
    except Exception as e:
        state["status"] = "error"
        state["message"] = f"{type(e).__name__}: {e}"


def _new_job(kind: str, **extra) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _download_lock:
        _download_jobs[job_id] = {
            "status": "queued", "percent": 0, "message": "",
            "kind": kind, **extra,
        }
    return job_id


@app.route("/api/models/refs")
def models_refs():
    return jsonify({"refs": _list_references()})


@app.route("/api/models/refs/download", methods=["POST"])
def models_refs_download():
    data = request.json or {}
    hf_id = data.get("hf_id", "google/gemma-3n-E4B-it").strip()
    hf_token = data.get("hf_token", "").strip()
    if not hf_id or "/" not in hf_id:
        return jsonify({"error": "hf_id must be <org>/<repo>"}), 400
    job_id = _new_job("ref_download", hf_id=hf_id)
    threading.Thread(target=_run_ref_download, args=(job_id, hf_id, hf_token), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/models/refs/delete", methods=["POST"])
def models_refs_delete():
    data = request.json or {}
    try:
        name = _sanitize_variant_name(data.get("name", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    ref = _resolve_reference_path(name)
    if ref is None:
        return jsonify({"error": f"reference '{name}' not found"}), 404
    if not ref.get("deletable", False):
        return jsonify({"error": f"'{name}' is a legacy directory — delete it manually from the shell."}), 403
    shutil.rmtree(ref["path"])
    return jsonify({"ok": True, "deleted": name})


@app.route("/api/models/quantize", methods=["POST"])
def models_quantize():
    data = request.json or {}
    try:
        ref = _sanitize_variant_name(data.get("ref", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    mode = data.get("mode", "int4").strip().lower()
    if mode not in ("int4", "int8", "fp16", "fp32"):
        return jsonify({"error": f"unsupported mode '{mode}'"}), 400
    job_id = _new_job("quantize", ref=ref, mode=mode)
    threading.Thread(target=_run_quantize_job, args=(job_id, ref, mode), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/models/download", methods=["POST"])
def models_download():
    # Legacy combined route — kept so old clients don't break.
    data = request.json or {}
    hf_id = data.get("hf_id", "google/gemma-3n-E4B-it").strip()
    mode = data.get("mode", "int4").strip().lower()
    if mode not in ("int4", "int8", "fp16", "fp32"):
        return jsonify({"error": f"unsupported mode '{mode}'"}), 400
    job_id = _new_job("download_and_quantize", hf_id=hf_id, mode=mode)
    threading.Thread(target=_run_download_job, args=(job_id, hf_id, mode), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/models/download_progress")
def models_download_progress():
    job_id = request.args.get("job_id", "")
    if job_id not in _download_jobs:
        return jsonify({"error": "unknown job_id"}), 404

    def stream():
        last_msg = None
        while True:
            state = _download_jobs.get(job_id)
            if state is None:
                break
            payload = {"status": state["status"], "percent": state["percent"],
                       "message": state["message"]}
            if payload != last_msg:
                yield f"data: {json.dumps(payload)}\n\n"
                last_msg = payload
            if state["status"] in ("done", "error"):
                break
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/gpu-info")
def gpu_info():
    return jsonify(_detect_gpu_info())


@app.route("/api/config/engine", methods=["GET", "POST"])
def config_engine():
    """Read or update engine-side options that apply on the next model reload."""
    global ENGINE_CONFIG
    if request.method == "POST":
        data = request.json or {}
        for key in ("cache_embeddings", "use_spec_decode", "spec_gamma", "spec_draft_variant"):
            if key in data:
                ENGINE_CONFIG[key] = data[key]
    return jsonify({
        "config": ENGINE_CONFIG,
        "restart_required": model_state["initialized"],
    })


@app.route("/api/config/reload_model", methods=["POST"])
def config_reload_model():
    """Force the engine to reinitialize with the current ENGINE_CONFIG."""
    model_state["initialized"] = False
    model_state["weights"] = None
    try:
        init_model()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/config/kv_cache", methods=["POST"])
def set_kv_cache():
    global MAX_NEW_TOKENS
    data = request.json or {}
    try:
        new_size = int(data.get("max_tokens", 512))
    except (TypeError, ValueError):
        return jsonify({"error": "max_tokens must be int"}), 400
    new_size = max(128, min(4096, new_size))
    MAX_NEW_TOKENS = new_size
    if model_state["initialized"]:
        NUM_LAYERS = gem_main.NUM_LAYERS
        model_state["K_cache"] = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
        model_state["V_cache"] = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
        model_state["cur_pos"] = 0
    return jsonify({"ok": True, "max_tokens": MAX_NEW_TOKENS,
                    "mem_mb": round(2 * gem_main.NUM_LAYERS * MAX_NEW_TOKENS * KV_CACHE_DIM * 2 / (1024 * 1024), 1)})


def init_model():
    global model_state
    model_state["initialized"] = True
    print(f"Initializing Model: {data.get('modelArch', 'gemma3N')} on {data.get('deviceMode', 'CPU')}"); return jsonify({"status": "success", "msg": "Mocked"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
