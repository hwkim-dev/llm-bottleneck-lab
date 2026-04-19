# Speculative Decoding with Gemma 3N MatFormer — Research Notes

_Status: open problem. Scaffold lives in `x64/gemma3N_E4B/spec_decode.py`._

## Goal

Use **Speculative Decoding** (Leviathan et al., 2023) to accelerate E4B
token generation on the Ryzen 4500U + Vega 6 target.

- **Target model** (slow, accurate): Gemma 3N **E4B** — 35 transformer layers,
  2048 hidden dim, INT4 packed weights, ~8–12 tok/s decode on 4500U.
- **Draft model** (fast, cheaper): Gemma 3N **E2B** — obtained by
  **MatFormer slicing** from E4B.

Expected speedup with γ = 4 and target/draft ratio ≈ 1.8: **1.4–1.9×**
on the 4500U, assuming ~70 % acceptance rate.

## What is MatFormer / Matryoshka?

Gemma 3N ships as a MatFormer ("Matryoshka Transformer"): a single
parameter tensor that can be **truncated at inference time** along
certain dimensions to yield progressively smaller, still-functional
submodels.

For Gemma 3N the published variants are:

| Variant | Active params (approx) | Layers | Hidden dim |
|---|---|---|---|
| E4B | ~4 B | 35 | 2048 |
| E2B | ~2 B | (subset) | (subset) |

The draft model should be *the same weights, read partially*, not a
separately trained small model.

**Open question #1 — what exactly does "E2B" select?**

HuggingFace `transformers` has a `Gemma3nConfig` with a `matformer_config`
/ `elasticity` field; the model code branches on it inside
`Gemma3nModel.forward`. The precise slicing rule (which hidden dims,
which layers, whether FFN intermediate shrinks proportionally) should
be read directly from `transformers/models/gemma3n/modeling_gemma3n.py`
at a pinned commit. This has not been done yet.

## Blocker: slicing INT4 packed weights

Our weights are stored after symmetric per-row INT4 quantization —
`(packed_uint8[N, M/2], scale_fp32[N])`. MatFormer slicing operates on
*weight rows and columns*, which for quantized data means:

1. **Row-wise slice** (output-dim reduction): discard trailing rows of
   `packed` and matching entries of `scale`. This works directly —
   scales are per-row so no re-quantization is needed.
2. **Column-wise slice** (input-dim reduction): discard trailing
   *columns* of the logical INT4 matrix. Because two INT4 values share
   each uint8 byte, a column count that isn't a multiple of 2 needs
   careful unpack→slice→repack. Scales are per-row and stay valid.
3. **FFN hidden-dim slice**: trims both dims of `W_gate`, `W_up`, `W_down`.
   Follows rules 1–2 composed. The residual path (`_sample_gate`
   thresholding in `main.forward_one_token`) must still see a
   contiguous post-slice tensor.
4. **Layer subset**: pick a subset of the 35 target layers. No weight
   surgery required — just call `forward_one_token` with a reduced
   `NUM_LAYERS` and a filtered layer dict.

Rule 4 alone is the simplest first experiment: call the existing
forward loop over e.g. layers `[0, 3, 7, 11, 15, 19, 23, 27, 31, 34]`
and see whether output is coherent. This is **not the correct E2B** but
it gives us a working draft scaffold to measure end-to-end speedup
upper bounds.

## Open question #2 — verification in one pass

The existing `forward_one_token` in `main.py` processes a single token
and writes to `K_cache[:, pos, :]`. Speculative decoding requires
verifying γ draft tokens *in parallel* through the target model,
producing γ logits vectors. Two options:

- **Batched forward**: change all matmuls to accept `x` of shape
  `[γ, hidden]`. The Vulkan GEMV kernel is shape-parameterized per call
  so this is additive work — new pipeline for batch>1, plus changes to
  KV-cache writes to `K_cache[:, pos:pos+γ, :]`.
- **Looped verify**: just call `forward_one_token` γ times with KV
  rollback if any token rejects. Easier to prototype, loses the main
  speedup source (no actual parallelism).

The batched path is what pays off.

## Implementation roadmap

1. Pin a `transformers` commit and read `modeling_gemma3n.py` to learn
   the exact MatFormer slicing rule. Document the slice in this file.
2. Implement `matformer_slice.py`:
   `slice_to_e2b(W: dict, layers_kept: list[int], ...) -> W_sliced: dict`.
3. Land a batched forward path in `main.py` accepting `γ` tokens at once.
4. Implement `SpeculativeDecoder.generate()` with rejection sampling
   (see the skeleton in `spec_decode.py`).
5. Benchmark on the 4500U vs. the plain decoder: wall-clock tok/s,
   acceptance rate, final text divergence.
6. Wire a "Speculative" toggle into `gui_app.py` (`/api/chat?mode=spec`)
   once quality-gated.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>
- Cai et al. *Medusa: Simple LLM Inference Acceleration Framework
  with Multiple Decoding Heads.* 2024. <https://arxiv.org/abs/2401.10774>
- HuggingFace `transformers/models/gemma3n/modeling_gemma3n.py`
- Devoto et al. *Matryoshka Transformer.* 2023. <https://arxiv.org/abs/2310.07707>
