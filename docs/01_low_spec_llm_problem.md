# The Low-Spec LLM Problem

Even small LLMs (1B-3B parameters) demand significant memory bandwidth and compute resources. On high-end GPUs, these requirements are easily met. However, on low-spec hardware (older laptops, embedded devices, low-power APUs), several bottlenecks emerge:

1. **Memory Bandwidth:** The primary bottleneck. Loading weights for every token generation step saturates the limited bandwidth of DDR4/LPDDR4 memory.
2. **Compute vs. Memory Bound:** While prefill (prompt processing) is often compute-bound, generation (decoding) is strictly memory-bound.
3. **iGPU Overhead:** Integrated GPUs share system memory with the CPU. Offloading compute to the iGPU via APIs like Vulkan introduces dispatch overhead, and the iGPU may not have enough compute units to overcome this overhead for small workloads.
4. **Quantization Decompression:** Formats like INT4 save bandwidth but require the CPU/iGPU to decompress weights to FP16/FP32 for matrix multiplication. If the hardware lacks fast integer-to-float conversion, the decompression becomes the new bottleneck.