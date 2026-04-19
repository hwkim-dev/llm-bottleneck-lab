# native/ — DEPRECATED

_Last active: 2026-04-19._

The Dear ImGui + Vulkan native GUI in this directory is **no longer the
supported front-end** for llm-lite. It was written early in the project when
the web GUI did not yet exist. With the web GUI now feature-complete
(model manager, stop-generation, KV cache slider, GPU warning, Apache-styled
About tab) the native path stopped paying its complexity cost.

## What replaced it

Two tracks only:

1. **Web GUI** — `bash run.sh` → `http://127.0.0.1:5000`.
   Full feature set, modern dark theme, zero native build.
2. **CLI** — `python3 main.py`.
   Headless, ideal for KV260 / Raspberry Pi / SSH sessions / scripting.

See [../README.md](../README.md) for the up-to-date Getting Started section.

## The files in this directory

They still build (`cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`)
and still talk to the Flask backend over localhost, but:

- New features (stop button, model manager, etc.) are not back-ported.
- The installer (`install.sh`) no longer compiles this by default.
- Bug reports against `native/` will be closed as `wontfix`.

If you want an ImGui-based front-end for a specific deployment, fork the
directory and maintain it separately — the code is small (~1k LOC) and the
HTTP/SSE client in `main.cpp` is a useful starting point.
