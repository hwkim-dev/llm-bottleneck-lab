# BitNet Backend Notes

BitNet (b1.58) utilizes ternary weights (-1, 0, +1).

## Objectives in `llm-lite`

We are **not** aiming to be a production replacement for `bitnet.cpp`. Our goal is to create a low-spec research adapter.

1.  **Ternary Packing:** Investigate optimal packing formats for ternary values to minimize memory footprints while allowing fast decoding.
2.  **Add/Sub Compute:** Ternary weights eliminate multiplication during inference (replacing it with addition/subtraction). We aim to benchmark if this theoretical advantage translates to practical speedups on low-spec CPUs lacking fast ML accelerators.
3.  **Adapter Implementation:** The `BitNetAdapter` is intentionally separated from the standard INT4 paths because the underlying math and data structures are fundamentally different.