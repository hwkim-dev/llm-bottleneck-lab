#!/usr/bin/env bash
# llm-lite :: Web GUI launcher — starts the Flask backend + web UI.
# The native Dear ImGui GUI (run-native.sh or build/llm-lite-native)
# connects to this Flask backend over localhost:5000.
set -e

source /home/hwkim/Desktop/github/llm-lite/x64/gemma3N_E4B/pynq_env/bin/activate
cd /home/hwkim/Desktop/github/llm-lite/x64/gemma3N_E4B

echo ""
echo "  llm-lite :: Gemma 3N E4B"
echo "  Web GUI: http://127.0.0.1:5000"
echo "  (native GUI talks to this same backend; start it from native/build/llm-lite-native)"
echo ""

python3 gui_app.py
