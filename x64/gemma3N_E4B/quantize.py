"""
Weight converter for Gemma 3N E4B.

Consumes HuggingFace-style .safetensors files and emits a directory of
per-tensor .npy files ready for the MMAP loader in safeTensor.py.

Modes:
    int4   — 4-bit packed uint8 (2 values/byte) + per-row FP32 scale.
             Consumed by the Vulkan GEMV kernel in IGPU_CORE.py.
    int8   — 8-bit int8 + per-row FP32 scale. No packing. CPU matmul.
    fp16   — raw half precision. No scale. CPU matmul.
    fp32   — full precision baseline. No scale. CPU matmul.

Layout output (dst/):
    model.<tensor_name>.npy               # weight data (one per tensor)
    model.<tensor_name>.scale.npy         # only for int4/int8
    manifest.json                         # {model, mode, created, size_bytes, num_files}

CLI:
    # (a) You already have a HuggingFace snapshot locally:
    python quantize.py --mode fp16 --src /absolute/path/to/hf-gemma-3n-E4B-it

    # (b) You want the script to fetch the snapshot from HuggingFace first:
    python quantize.py --mode fp16 --hf-id google/gemma-3n-E4B-it

    # --dst defaults to  models/gemma-3n-e4b-<mode>
"""

import os
import gc
import glob
import json
import time
import argparse
import numpy as np
import torch
from safetensors.torch import load_file


# Tensors large enough to be worth quantizing (INT4 path).
_BIG_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "per_layer_input_gate.weight",
    "per_layer_model_projection.weight",
    "laurel.linear_left.weight",
    "laurel.linear_right.weight",
)

# Tensor name fragments that need to be stored transposed for the raw FP path
# (safeTensor.py + main.py hw_matmul consume these via `np.dot(x, w)` where
# w is expected as [K_in, M_out]; the HF layout is [M_out, K_in]).
# INT4/INT8 quantized tensors are stored untransposed because the Vulkan GEMV
# kernel and the CPU unpack path both consume them row-major as [M_out, K_in].
_TRANSPOSE_PATTERNS = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    "per_layer_model_projection.weight",
    "altup_projections", "altup_unembed_projections",
    "per_layer_input_gate.weight", "per_layer_projection.weight",
    "laurel.linear_left.weight", "laurel.linear_right.weight",
    "altup.modality_router.weight",
)


def _is_big_2d(name: str, shape) -> bool:
    return len(shape) == 2 and any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)


def _needs_transpose(name: str) -> bool:
    return any(p in name for p in _TRANSPOSE_PATTERNS)


def quantize_to_int4(weight: np.ndarray):
    """Symmetric per-row INT4. Returns (packed_uint8 [N, M//2], scale_fp32 [N])."""
    w_f32 = weight.astype(np.float32)
    max_vals = np.maximum(np.max(np.abs(w_f32), axis=1, keepdims=True), 1e-8)
    scale = (max_vals / 7.0).flatten()
    w_q = np.clip(np.round(w_f32 / max_vals * 7.0).astype(np.int8), -8, 7)
    packed = ((w_q[:, 0::2] & 0x0F) | ((w_q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return packed, scale.astype(np.float32)


def quantize_to_int8(weight: np.ndarray):
    """Symmetric per-row INT8. Returns (w_int8 [N, M], scale_fp32 [N])."""
    w_f32 = weight.astype(np.float32)
    max_vals = np.maximum(np.max(np.abs(w_f32), axis=1, keepdims=True), 1e-8)
    scale = (max_vals / 127.0).flatten()
    w_q = np.clip(np.round(w_f32 / max_vals * 127.0).astype(np.int16), -128, 127).astype(np.int8)
    return w_q, scale.astype(np.float32)


def _convert_tensor(name: str, tensor, mode: str):
    """Return list of (suffix, np_array) pairs to write to disk."""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    arr = tensor.numpy()

    if _is_big_2d(name, arr.shape):
        if mode == "int4":
            packed, scale = quantize_to_int4(arr)
            return [(".npy", packed), (".scale.npy", scale)]
        elif mode == "int8":
            w_q, scale = quantize_to_int8(arr)
            return [(".npy", w_q), (".scale.npy", scale)]
        elif mode == "fp16":
            arr = arr.T if _needs_transpose(name) else arr
            return [(".npy", np.ascontiguousarray(arr).astype(np.float16))]
        elif mode == "fp32":
            arr = arr.T if _needs_transpose(name) else arr
            return [(".npy", np.ascontiguousarray(arr).astype(np.float32))]

    # Small / non-quantized tensors are always saved as FP32 with transpose
    # matching the consumer's expected layout.
    arr = arr.T if _needs_transpose(name) else arr
    return [(".npy", np.ascontiguousarray(arr).astype(np.float32))]


def convert(src_dir: str, dst_dir: str, mode: str):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"--src path does not exist: {src_dir}\n"
            f"  Either pass a real HuggingFace snapshot directory, or use\n"
            f"  --hf-id <repo_id> to fetch it automatically."
        )
    files = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors found in {src_dir}\n"
            f"  Directory contents: {os.listdir(src_dir)[:5]}...\n"
            f"  Expected the HF Gemma-3N snapshot with model-*.safetensors files."
        )

    total_bytes = 0
    count = 0
    t0 = time.time()
    for st_path in files:
        print(f"[{mode}] {os.path.basename(st_path)}")
        tensors = load_file(st_path)
        for name, tensor in tensors.items():
            for suffix, arr in _convert_tensor(name, tensor, mode):
                out_path = os.path.join(dst_dir, name + suffix)
                np.save(out_path, arr)
                total_bytes += arr.nbytes
                count += 1
        del tensors
        gc.collect()

    manifest = {
        "model": "gemma-3n-e4b",
        "mode": mode,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_files": count,
        "size_bytes": total_bytes,
        "source": os.path.abspath(src_dir),
    }
    with open(os.path.join(dst_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    size_gb = total_bytes / (1024 ** 3)
    print(f"✓ {count} tensors, {size_gb:.2f} GB → {dst_dir}  (in {elapsed:.1f}s)")


def infer_model_name(src_dir_or_hf_id: str) -> str:
    """Normalize a HF repo id or hf_cache directory name to a short model id.

        'google/gemma-3n-E4B-it'             → 'gemma-3n-e4b'
        'google__gemma-3n-E2B-it'            → 'gemma-3n-e2b'
        '/path/to/local_gemma_3n_int4'       → 'gemma-3n-e4b' (legacy)
        '/path/to/hf_cache/google__gemma-3n-E4B-it' → 'gemma-3n-e4b'
    """
    leaf = os.path.basename(os.path.normpath(src_dir_or_hf_id))
    name = leaf.replace("__", "/").split("/")[-1]  # strip org
    name = name.lower()
    # Strip common suffixes
    for suf in ("-it", "-pt", "-chat", "-instruct"):
        if name.endswith(suf):
            name = name[:-len(suf)]
    # Legacy naming fall-backs
    if name.startswith("local_gemma_3n"):
        name = "gemma-3n-e4b"
    if not name.startswith("gemma"):
        # safe default when we can't identify the model
        name = "gemma-3n-e4b"
    return name


def _default_dst(mode: str, model_name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models", f"{model_name}-{mode}")


def _hf_cache_dir(hf_id: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "hf_cache", hf_id.replace("/", "__"))


def main():
    ap = argparse.ArgumentParser(description="Convert Gemma 3N weights into an MMAP-ready variant.")
    ap.add_argument("--mode", choices=["int4", "int8", "fp16", "fp32"], default="int4",
                    help="Output precision. Default: int4.")
    ap.add_argument("--src", default=None,
                    help="Source directory containing HuggingFace .safetensors. "
                         "Mutually exclusive with --hf-id.")
    ap.add_argument("--hf-id", default=None,
                    help="HuggingFace repo id (e.g. google/gemma-3n-E4B-it). "
                         "Script snapshot-downloads into hf_cache/<id>/ then quantizes.")
    ap.add_argument("--name", default=None,
                    help="Model id for the output directory (default: inferred from --src/--hf-id). "
                         "E.g. 'gemma-3n-e4b', 'gemma-3n-e2b'.")
    ap.add_argument("--dst", default=None,
                    help="Destination directory. Default: models/<name>-<mode>.")
    args = ap.parse_args()

    if args.src is None and args.hf_id is None:
        ap.error("Either --src <dir> or --hf-id <repo> must be provided.")
    if args.src and args.hf_id:
        ap.error("--src and --hf-id are mutually exclusive.")

    if args.hf_id:
        from huggingface_hub import snapshot_download
        args.src = _hf_cache_dir(args.hf_id)
        print(f"[hf] Fetching {args.hf_id} → {args.src}")
        snapshot_download(
            repo_id=args.hf_id,
            local_dir=args.src,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )

    model_name = args.name or infer_model_name(args.hf_id or args.src)
    convert(args.src, args.dst or _default_dst(args.mode, model_name), args.mode)


if __name__ == "__main__":
    main()
