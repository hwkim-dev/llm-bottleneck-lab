import argparse
import sys
import os
import json

from engine.core.model_config import ModelConfig
import engine.models  # Ensure models are registered
from engine.core.registry import ModelRegistry
from engine.core.runtime import RuntimeContext
from engine.backends.cpu import CPUBackend
from engine.backends.vulkan import VulkanBackend
from engine.backends.npu_uca import NPUBackend

def main():
    parser = argparse.ArgumentParser(description="llm-lite runner")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--backend", type=str, required=True, choices=["cpu", "vulkan", "npu_uca"], help="Backend to use")
    parser.add_argument("--precision", type=str, required=True, choices=["fp16", "int8", "int4", "ternary"], help="Quantization precision")
    parser.add_argument("--arch", type=str, help="Override model architecture (llama, qwen, deepseek-distill, gemma3n, bitnet)")
    parser.add_argument("--dry-run", action="store_true", help="Run without actually loading weights or executing inference")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum number of new tokens to generate")
    parser.add_argument("--prompt", type=str, default="Hello,", help="Input prompt")

    args = parser.parse_args()

    # Determine ModelConfig
    config_path = os.path.join(args.model, "config.json")
    if not os.path.exists(config_path):
        if not args.json:
            print(f"Warning: config.json not found in {args.model}, using stub configuration.", file=sys.stderr)

        # Simple heuristic fallback if model_type is unknown but the folder name contains hints
        model_name_lower = os.path.basename(args.model).lower()
        model_type_inferred = "unknown"
        if "llama" in model_name_lower:
            model_type_inferred = "llama"
        elif "qwen" in model_name_lower:
            model_type_inferred = "qwen"
        elif "deepseek" in model_name_lower:
            model_type_inferred = "deepseek-distill"
        elif "bitnet" in model_name_lower:
            model_type_inferred = "bitnet"
        elif "gemma" in model_name_lower:
            model_type_inferred = "gemma3n"

        config = ModelConfig(
            model_type=model_type_inferred,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=32000,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            torch_dtype="float16"
        )
    else:
        config = ModelConfig.from_file(config_path)

    # Detect model architecture
    model_type = args.arch if args.arch else config.model_type

    # Try getting the model class
    model_cls = ModelRegistry.get_model(model_type)
    if model_cls is None:
        if args.json:
            print(json.dumps({"error": f"Unsupported model type '{model_type}'"}), file=sys.stdout)
        else:
            print(f"Error: Unsupported model type '{model_type}'", file=sys.stderr)
            print(f"Available models: {ModelRegistry.list_models()}", file=sys.stderr)
        sys.exit(1)

    # Initialize RuntimeContext
    runtime = RuntimeContext(backend_name=args.backend, precision=args.precision, dry_run=args.dry_run)
    model_instance = model_cls(config, runtime)

    # Validate Backend and Precision
    adapter_summary = model_instance.dry_run_summary()
    validation_warnings = []
    validation_ok = True

    if args.backend not in adapter_summary["supported_backends"]:
        validation_warnings.append(f"Backend '{args.backend}' is not in adapter's supported backends: {adapter_summary['supported_backends']}")
        validation_ok = False

    if args.precision not in adapter_summary["supported_precisions"]:
        validation_warnings.append(f"Precision '{args.precision}' is not in adapter's supported precisions: {adapter_summary['supported_precisions']}")
        validation_ok = False

    backend_cls = CPUBackend if args.backend == "cpu" else (VulkanBackend if args.backend == "vulkan" else NPUBackend)
    backend_instance = backend_cls()
    if not backend_instance.supports_precision(args.precision):
        validation_warnings.append(f"Precision '{args.precision}' is not supported by backend '{args.backend}'")
        validation_ok = False

    if not validation_ok:
        if args.json:
            print(json.dumps({"error": "Validation failed", "warnings": validation_warnings}), file=sys.stdout)
        else:
            print("Error: Validation failed", file=sys.stderr)
            for warning in validation_warnings:
                print(f" - {warning}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Normalized JSON Output
        summary = {
            "project": "llm-lite",
            "mode": "dry-run" if args.dry_run else "real",
            "model": {
                "path": args.model,
                "architecture": model_instance.architecture_name(),
                "model_type": model_type,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "intermediate_size": config.intermediate_size,
                "vocab_size": config.vocab_size
            },
            "runtime": {
                "backend": args.backend,
                "precision": args.precision,
                "dry_run": args.dry_run,
                "arch_override": args.arch
            },
            "adapter": adapter_summary,
            "validation": {
                "ok": validation_ok,
                "warnings": validation_warnings
            },
            "notes": [
                "dry-run only; no real weights loaded" if args.dry_run else "WIP real weights implementation",
                "not a measured benchmark"
            ]
        }
        print(json.dumps(summary, indent=2))
        if args.dry_run:
            sys.exit(0)
    else:
        if not args.dry_run:
            print(f"Starting run.py with model={args.model}, backend={args.backend}, precision={args.precision}, dry_run={args.dry_run}")
            print(f"Detected model type: {model_type}")
        else:
            print(f"[llm-lite] model: {model_instance.architecture_name()}")
            print(f"[llm-lite] backend: {args.backend}")
            print(f"[llm-lite] precision: {args.precision}")
            print(f"[llm-lite] mode: dry-run")
            print(f"[llm-lite] layers: {config.num_hidden_layers}")
            print(f"[llm-lite] hidden_size: {config.hidden_size}")
            print(f"[llm-lite] status: adapter resolved successfully")

    if not args.dry_run:
        print("Note: Real inference is currently a WIP for most backends/models. Using stub implementation.", file=sys.stderr)
        model_instance.load_weights(args.model)
        # Dummy token prompt based on length
        tokens = model_instance.generate([1, 2, 3], max_new_tokens=args.max_new_tokens)
        print(f"Generated tokens: {tokens}")
        # Explicitly return code 2 for un-implemented WIP as requested
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
