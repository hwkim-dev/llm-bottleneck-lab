#!/bin/bash
# Fix pynq_env paths after folder relocation
# Old: /home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env
# New: /home/hwkim/Desktop/github/llm-lite/x64/gemma3N_E4B/pynq_env

OLD_PATH="/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env"
NEW_PATH="/home/hwkim/Desktop/github/llm-lite/x64/gemma3N_E4B/pynq_env"

echo "Fixing venv paths..."
echo "  Old: $OLD_PATH"
echo "  New: $NEW_PATH"

# Fix all text files in bin/
find "$NEW_PATH/bin" -type f -exec sed -i "s|$OLD_PATH|$NEW_PATH|g" {} +

# Verify fix
echo ""
echo "=== Verification ==="
grep -r "TinyNPU-RTL" "$NEW_PATH/bin/" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✓ No old paths remaining - all fixed!"
else
    echo "✗ Some old paths still exist"
fi

echo ""
echo "Now run:"
echo "  source $NEW_PATH/bin/activate"
echo "  cd /home/hwkim/Desktop/github/llm-lite/x64/gemma3N_E4B"
echo "  python3 main.py"
