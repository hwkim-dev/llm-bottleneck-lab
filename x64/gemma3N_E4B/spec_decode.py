"""
Speculative Decoding for Gemma 3N E4B.

Two strategies are scaffolded here.  Only (A) is wired end-to-end today —
the scaffolding for (B) is in docs/Speculative_Decoding_Research.md.

(A) Self-speculative / layer-skip draft [IMPLEMENTED]
    Draft = run the first K out of 35 transformer layers of the same E4B model.
    Target = run all 35 layers.
    Requires no extra weights, purely a code path.
    CAVEAT: without LayerSkip training the draft tokens are noisy, so the
    acceptance rate is typically <10%.  You should only expect a speed-up
    if you also fine-tune for early-exit — see the research note.

(B) Dual-model MatFormer draft  [NOT WIRED YET]
    Draft = separate E2B (smaller intermediate_size) model, loaded into its
            own engine with correctly sized FFN buffers.
    Target = E4B.
    This is the "real" MatFormer speculative approach and does give speedup,
    but it requires per-layer buffer resizing which is outside the scope of
    this module right now.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_speculative(
    user_input: str,
    state,                 # gui_app.model_state dict
    main_mod,              # the `main` module (forward_one_token, _sample, etc.)
    CPU_CORE,              # the CPU_CORE extension module (tokenize, tokenizer)
    cancel_event,          # threading.Event — set to abort
    temperature: float = 0.65,
    top_p: float = 0.9,
    rep_penalty: float = 1.15,
    max_tokens: int = 512,
    gamma: int = 4,
    draft_layers: int = 17,
):
    """Generator yielding (kind, payload) events.

    Events:
        ('token', (new_text, total_count))
        ('done',  {'total_tokens': int, 'context_pos': int, 'stopped': bool,
                   'accepted': int, 'drafted': int})
    """
    w = state["weights"]
    
    # Init E2B MatFormer draft weights once per session if missing or draft_layers changed
    if "W_draft" not in w or len(w["W_draft"]["W_gate"]) != draft_layers:
        import matformer_slice
        w["W_draft"] = matformer_slice.slice_to_e2b(w["W"], num_layers=draft_layers, intermediate_size=8192)

    K_cache = state["K_cache"]
    V_cache = state["V_cache"]

    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    input_tokens = CPU_CORE.tokenize(prompt)

    cur_pos = state["cur_pos"]

    # Prefill through the FULL model.
    xs = None
    for token_id in input_tokens:
        if cancel_event.is_set():
            break
        xs = main_mod.forward_one_token(
            token_id, cur_pos, w["W"], w["W_embed"], w["W_ple_packed"],
            w["W_ple_scale"], w["norm_ple"], w["W_ple_proj"],
            w["altup_projs"], K_cache, V_cache,
        )
        cur_pos += 1

    STOP_TOKENS = {1, 106}
    generated: list[int] = []
    printed_text = ""
    token_count = 0
    accepted_total = 0
    drafted_total = 0
    stopped = False
    last_token = input_tokens[-1] if len(input_tokens) > 0 else 2

    while token_count < max_tokens:
        if cancel_event.is_set():
            stopped = True
            break

        # ---- Draft γ tokens via layer-subset forward pass ----
        draft_tokens = []
        draft_xs = xs
        draft_prev = last_token
        for d in range(gamma):
            draft_xs = main_mod.forward_draft_one_token(
                draft_prev, cur_pos + d, w["W"], w["W_embed"], w["W_ple_packed"],
                w["W_ple_scale"], w["norm_ple"], w["W_ple_proj"],
                w["altup_projs"], K_cache, V_cache,
                num_draft_layers=draft_layers, W_draft=w["W_draft"]
            )
            logits = main_mod.decode_logits(draft_xs, w["altup_unprojs"],
                                            w["W_final_norm"], w["W_lm_head"])
            # Greedy to keep the draft deterministic & cheap.
            tok = int(np.argmax(logits))
            if tok in STOP_TOKENS:
                draft_tokens.append(tok)
                break
            draft_tokens.append(tok)
            draft_prev = tok
        drafted_total += len(draft_tokens)

        # ---- Verify each draft token with the FULL model ----
        accepted_this_round = 0
        verify_prev = last_token
        resampled: int | None = None
        for k, dt in enumerate(draft_tokens):
            xs = main_mod.forward_one_token(
                verify_prev, cur_pos, w["W"], w["W_embed"], w["W_ple_packed"],
                w["W_ple_scale"], w["norm_ple"], w["W_ple_proj"],
                w["altup_projs"], K_cache, V_cache,
            )
            logits = main_mod.decode_logits(xs, w["altup_unprojs"],
                                            w["W_final_norm"], w["W_lm_head"])
            logits = 30.0 * np.tanh(logits / 30.0)
            target_tok = main_mod._sample(logits, temperature, top_p, rep_penalty, generated)
            if target_tok == dt:
                accepted_this_round += 1
                generated.append(dt)
                token_count += 1
                cur_pos += 1
                verify_prev = dt
                if dt in STOP_TOKENS:
                    break
            else:
                resampled = target_tok
                cur_pos += 1
                break

        accepted_total += accepted_this_round

        # ---- If nothing was accepted, still move forward with the target's pick ----
        if resampled is not None:
            generated.append(resampled)
            token_count += 1
            last_token = resampled
            if resampled in STOP_TOKENS:
                break
        else:
            # Whole draft accepted.  Next turn's prev is the last draft token.
            last_token = draft_tokens[-1] if draft_tokens else last_token
            if last_token in STOP_TOKENS:
                break

        # Emit any new text since the last flush.
        current_text = CPU_CORE.tokenizer.decode(generated, skip_special_tokens=True)
        new_text = current_text[len(printed_text):]
        printed_text = current_text
        if new_text:
            yield ("token", (new_text, token_count))

    state["cur_pos"] = cur_pos
    yield ("done", {
        "total_tokens": token_count,
        "context_pos": cur_pos,
        "stopped": stopped,
        "accepted": accepted_total,
        "drafted": drafted_total,
    })


# ---------------------------------------------------------------------------
# Legacy stub kept for back-compat with earlier scaffolding
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Thin wrapper preserved for documentation purposes.  Real entry point is
    :func:`generate_speculative` above.
    """

    def __init__(self, draft_weights=None, target_weights=None, gamma: int = 4):
        self.draft_w = draft_weights
        self.target_w = target_weights
        self.gamma = gamma
