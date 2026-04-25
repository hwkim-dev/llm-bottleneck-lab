import ctypes
import numpy as np
import os
import sys

def main():
    dll_path = os.path.join(os.path.dirname(__file__), 'C_DLL', 'my_accelerator.so')

    if not os.path.exists(dll_path):
        print(f"ERROR: DLL not found at {dll_path}")
        sys.exit(1)

    print(f"Loading {dll_path}...")
    try:
        accelerator = ctypes.CDLL(dll_path)
    except Exception as e:
        print(f"ERROR loading DLL: {e}")
        sys.exit(1)

    print("DLL loaded successfully!")

    # Test a simple function that doesn't need external data
    # void run_gelu_inplace(float* __restrict__ x, int length)
    try:
        run_gelu_inplace = accelerator.run_gelu_inplace
        run_gelu_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        run_gelu_inplace.restype = None

        test_arr = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        print(f"Testing run_gelu_inplace with input: {test_arr}")

        c_float_p = test_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        run_gelu_inplace(c_float_p, len(test_arr))

        print(f"GELU output: {test_arr}")
        print("Smoke test passed successfully!")
    except Exception as e:
        print(f"ERROR calling function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
