#!/bin/bash
cd "$(dirname "$0")"
source pynq_env/bin/activate

rm -f thread_scaling_results.csv

for T in 1 2 4 6; do
    echo "========== OMP_NUM_THREADS=$T =========="
    OMP_NUM_THREADS=$T python3 run_thread_scaling.py
    echo ""
done

echo "===== All thread scaling done ====="
cat thread_scaling_results.csv
