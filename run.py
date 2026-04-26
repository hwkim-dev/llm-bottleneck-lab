import argparse
import sys
import os

from engine.core.model_config import ModelConfig
# Ensure models are registered
import engine.models
from engine.core.registry import ModelRegistry
from engine.core.runtime import RuntimeContext

def main():
    parser = argparse.ArgumentParser(description="llm-lite runner")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--backend", type=str, required=True, choices=["cpu", "vulkan", "npu_uca"], help="Backend to use")
    parser.add_argument("--precision", type=str, required=True, choices=["fp16", "int8", "int4", "ternary"], help="Quantization precision")
    parser.add_argument("--dry-run", action="store_true", help="Run without actually loading weights or executing inference")

    args = parser.parse_args()

    print(f"Starting run.py with model={args.model}, backend={args.backend}, precision={args.precision}, dry_run={args.dry_run}")

    config_path = os.path.join(args.model, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: config.json not found in {args.model}, using stub configuration.")
        config = ModelConfig(
            model_type="unknown",
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

    model_type = config.model_type

    # Simple heuristic fallback if model_type is unknown but the folder name contains hints
    if model_type == "unknown":
        model_name_lower = os.path.basename(args.model).lower()
        if "llama" in model_name_lower:
            model_type = "llama"
        elif "qwen" in model_name_lower:
            model_type = "qwen"
        elif "deepseek" in model_name_lower:
            model_type = "deepseek-distill"
        elif "bitnet" in model_name_lower:
            model_type = "bitnet"
        elif "gemma" in model_name_lower:
            model_type = "gemma3n"

    print(f"Detected model type: {model_type}")

    model_cls = ModelRegistry.get_model(model_type)
    if model_cls is None:
        print(f"Error: Unsupported model type '{model_type}'")
        print(f"Available models: {ModelRegistry.list_models()}")
        sys.exit(1)

    runtime = RuntimeContext(backend_name=args.backend, precision=args.precision)
    model_instance = model_cls(config, runtime)

    if not args.dry_run:
        model_instance.load_weights(args.model)
        # Dummy prompt
        tokens = model_instance.generate([1, 2, 3], max_new_tokens=10)
        print(f"Generated tokens: {tokens}")
    else:
        print("Dry run complete.")

if __name__ == "__main__":
    main()
