#include <cmath>
#include <cstdint> // uint8_t
#include <omp.h>   
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define PORTABLE_FP16 _Float16
#else
    #define PORTABLE_FP16 __fp16
#endif

#define GELU_CONST 0.7978845608028654f

// Force export to C specification so Python ctypes can find function names.
extern "C" {
    // __restrict__: A keyword that swears to the compiler that "this pointer memory is for me only!"
    // Without this, the compiler is afraid and cannot perform SIMD parallelization properly.

    // Overwrite (In-place) operation by receiving only the pointer (*x) and length of the array
    void run_gelu_inplace(float* __restrict__ x, int length) {

        #pragma omp simd
        for (int i = 0; i < length; i++) {
            float val = x[i];
            float cube = val * val * val; 
            float inner = GELU_CONST * (val + 0.044715f * cube);
            
            x[i] = 0.5f * val * (1.0f + std::tanh(inner)); 
        }
    }

    // RMSNorm
    void run_RMSNorm_inplace(float* __restrict__ x, const float* __restrict__ gamma, int length) {
        double sum = 0.0f;

        // separately add and add up after
        #pragma omp simd reduction(+ : sum)        
        for(int i = 0; i < length; i++){ 
            float val = x[i];
            sum += val * val;
        }

        float inv_rms = 1.0f / std::sqrt((sum / (float)length) + 1e-6f);
        
        #pragma omp simd
        for(int i = 0; i < length; i++){
            x[i] = x[i] * inv_rms * gamma[i];
        }    
    }

    void run_unpack_int4_inplace(const uint8_t *__restrict__ packed, float scale, float *__restrict__ out, int packed_length)
    {
        #pragma omp simd
        for (int i = 0; i < packed_length; i++)
        {
            uint8_t p = packed[i];

            // 1. extract low 4bits
            int8_t low = p & 0x0F;
            if (low > 7)
                low -= 16;

            // extract high 4bits
            int8_t high = (p >> 4) & 0x0F;
            if (high > 7)
                high -= 16;

            out[2 * i] = (float)low * scale;
            out[2 * i + 1] = (float)high * scale;
        }
    }

    void run_rope_inplace(float *__restrict__ x, int pos, float theta_base, int num_heads, int dim)
    {
        int half = dim / 2;

        // No matter how many heads there are, the angle is the same, so it is calculated only once (128 times) and placed in the cache.
        float cos_vals[128];
        float sin_vals[128];

        #pragma omp simd
        for (int i = 0; i < half; i++)
        {
            // Frequency calculation: 1.0 / (theta_base ^ (2 * i / dim))
            float exp_val = (2.0f * (float)i) / (float)dim;
            float freq = 1.0f / std::pow(theta_base, exp_val);
            float angle = (float)pos * freq;

            cos_vals[i] = std::cos(angle);
            sin_vals[i] = std::sin(angle);
        }

        // Rotation is applied to each head using the calculated cos and sin values ​​(In-place overwrite)
        for (int h = 0; h < num_heads; h++)
        {
            int head_offset = h * dim;
            float *x_head = x + head_offset;

            #pragma omp simd
            for (int i = 0; i < half; i++)
            {
                float x0 = x_head[i];
                float x1 = x_head[i + half];

                float cos_a = cos_vals[i];
                float sin_a = sin_vals[i];

                x_head[i] = x0 * cos_a - x1 * sin_a;
                x_head[i + half] = x1 * cos_a + x0 * sin_a;
            }
        }
    }

    // Softmax Acceleration (Temperature scaling , In-place overwrite)
    void run_softmax_inplace(float *__restrict__ logits, int length, float temperature)
    {
        // prevent divide by zero
        float temp = (temperature > 1e-8f) ? temperature : 1e-8f;
        float inv_temp = 1.0f / temp;

        float max_val = -INFINITY;

        // 1. Temperature divide(mult inver) & find max val in one loop(fusion)
        #pragma omp simd reduction(max : max_val)
        for (int i = 0; i < length; i++)
        {
            logits[i] *= inv_temp;
            if (logits[i] > max_val)
            {
                max_val = logits[i];
            }
        }

        double sum_exp = 0.0;

        // 2. safe Exp expression
        // fusion Sum Up in single loop (using double for Enhance Precision)
        #pragma omp simd reduction(+ : sum_exp)
        for (int i = 0; i < length; i++)
        {
            logits[i] = std::exp(logits[i] - max_val);
            sum_exp += (double)logits[i];
        }

        // normalize
        float inv_sum = (float)(1.0 / sum_exp);

        #pragma omp simd
        for (int i = 0; i < length; i++)
        {
            logits[i] *= inv_sum;
        }
    }


    void run_gemv_int4(const float *__restrict__ vec, const uint8_t *__restrict__ mat_p, const float *__restrict__ scale, float *__restrict__ out, int M_out, int K_in)
    {
        int K_packed = K_in / 2;

        // use all core
        #pragma omp parallel for
        for (int i = 0; i < M_out; i++)
        {
            float acc = 0.0f;
            const uint8_t *row_p = mat_p + i * K_packed;

            // Using AVX2 SIMD (union of 8 data calc) in single core
            #pragma omp simd reduction(+ : acc)
            for (int k = 0; k < K_packed; k++)
            {
                uint8_t p = row_p[k];

                int8_t low = p & 0x0F;
                if (low > 7)
                    low -= 16;

                int8_t high = (p >> 4) & 0x0F;
                if (high > 7)
                    high -= 16;

                acc += vec[2 * k] * (float)low + vec[2 * k + 1] * (float)high;
            }
            out[i] = acc * scale[i];
        }
    }

    // INT4 GEMV + GeLU fusion (FFN gate, reduce memory Access)
    void run_gemv_int4_gelu(
        const float *__restrict__ vec, 
        const uint8_t *__restrict__ mat_p, 
        const float *__restrict__ scale, 
        float *__restrict__ out, 
        int M_out, 
        int K_in)
    {
        int K_packed = K_in / 2;

        #pragma omp parallel for
        for (int i = 0; i < M_out; i++)
        {
            float acc = 0.0f;
            const uint8_t *row_p = mat_p + i * K_packed;

            #pragma omp simd reduction(+ : acc)
            for (int k = 0; k < K_packed; k++)
            {
                uint8_t p = row_p[k];

                int8_t low = p & 0x0F;
                if (low > 7)
                    low -= 16;

                int8_t high = (p >> 4) & 0x0F;
                if (high > 7)
                    high -= 16;

                acc += vec[2 * k] * (float)low + vec[2 * k + 1] * (float)high;
            }
            float v = acc * scale[i];

            // GeLU
            float cube = v * v * v;
            float inner = GELU_CONST * (v + 0.044715f * cube);
            out[i] = 0.5f * v * (1.0f + std::tanh(inner));
        }
    }

    // ================================================================
    // NEW: Fused GQA (Grouped Query Attention) — Full attention in one C++ call
    // Q: [num_q_heads * head_dim] = [8 * 256] = [2048] float32
    // K_cache: [seq_len, kv_dim] = [seq_len, 512] float16
    // V_cache: [seq_len, kv_dim] = [seq_len, 512] float16
    // out: [num_q_heads * head_dim] = [2048] float32
    // num_kv_groups=2, heads_per_group=4, head_dim=256
    // ================================================================
    void run_gqa_fused(
        const float *__restrict__ Q,
        const uint16_t *__restrict__ K_cache,   // float16 as uint16
        const uint16_t *__restrict__ V_cache,   // float16 as uint16
        float *__restrict__ out,
        int seq_len,
        int num_kv_groups,    // 2
        int heads_per_group,  // 4
        int head_dim)         // 256
    {
        int kv_dim = num_kv_groups * head_dim; // 512

        for (int g = 0; g < num_kv_groups; g++)
        {
            for (int h = 0; h < heads_per_group; h++)
            {
                int q_head_idx = g * heads_per_group + h;
                const float *q_head = Q + q_head_idx * head_dim;
                float *out_head = out + q_head_idx * head_dim;

                // 1. Compute attention scores: Q_head @ K_cache_group^T
                float max_score = -INFINITY;
                float scores_buf[4096];
                float *scores = (seq_len <= 4096) ? scores_buf : new float[seq_len];

                for (int s = 0; s < seq_len; s++)
                {
                    float dot = 0.0f;
                    const uint16_t *k_row = K_cache + s * kv_dim + g * head_dim;
                    #pragma omp simd reduction(+:dot)
                    for (int d = 0; d < head_dim; d++)
                    {
                    // Hardware-correct fp16->fp32 conversion
                    PORTABLE_FP16 k_h;
                    std::memcpy(&k_h, &k_row[d], sizeof(PORTABLE_FP16));
                        dot += q_head[d] * (float)k_h;
                    }
                    scores[s] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // 2. Softmax (numerically stable)
                float sum_exp = 0.0f;
                for (int s = 0; s < seq_len; s++)
                {
                    scores[s] = std::exp(scores[s] - max_score);
                    sum_exp += scores[s];
                }
                float inv_sum = 1.0f / sum_exp;
                for (int s = 0; s < seq_len; s++)
                    scores[s] *= inv_sum;

                // 3. Weighted sum: scores @ V_cache_group
                for (int d = 0; d < head_dim; d++)
                {
                    float acc = 0.0f;
                    for (int s = 0; s < seq_len; s++)
                    {
                        const uint16_t *v_row = V_cache + s * kv_dim + g * head_dim;
                    PORTABLE_FP16 v_h;
                    std::memcpy(&v_h, &v_row[d], sizeof(PORTABLE_FP16));
                        acc += scores[s] * (float)v_h;
                    }
                    out_head[d] = acc;
                }

                if (seq_len > 4096) delete[] scores;
            }
        }
    }

    // ================================================================
    // NEW: Small GEMV for float32 matrices (laurel, ple, altup_proj etc.)
    // Uses OpenMP multicore for non-quantized float16/float32 matrices
    // x: [K_in] float32, mat: [M_out, K_in] float32 (row-major, already transposed)
    // out: [M_out] float32
    // ================================================================
    void run_small_gemv_f32(
        const float *__restrict__ x,
        const float *__restrict__ mat,
        float *__restrict__ out,
        int M_out,
        int K_in)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M_out; i++)
        {
            float acc = 0.0f;
            const float *row = mat + i * K_in;
            
            #pragma omp simd reduction(+ : acc)
            for (int k = 0; k < K_in; k++)
            {
                acc += x[k] * row[k];
            }
            out[i] = acc;
        }
    }

    // ================================================================
    // NEW: Fused QK Norm + RoPE
    // Performs per-head RMS normalization and RoPE rotation in one call
    // q: [q_total] float32 (2048 = 8 heads * 256)
    // k: [k_total] float32 (512 = 2 heads * 256)
    // gamma_q, gamma_k: [head_dim] float32 (256)
    // ================================================================
    void run_qk_norm_rope_fused(
        float *__restrict__ q,
        float *__restrict__ k,
        const float *__restrict__ gamma_q,
        const float *__restrict__ gamma_k,
        int pos,
        float theta_base,
        int num_q_heads,   // 8
        int num_k_heads,   // 2
        int head_dim)      // 256
    {
        int half = head_dim / 2;

        // Pre-compute cos/sin (shared across all heads)
        float cos_vals[128];
        float sin_vals[128];
        
        #pragma omp simd
        for (int i = 0; i < half; i++)
        {
            float exp_val = (2.0f * (float)i) / (float)head_dim;
            float freq = 1.0f / std::pow(theta_base, exp_val);
            float angle = (float)pos * freq;
            cos_vals[i] = std::cos(angle);
            sin_vals[i] = std::sin(angle);
        }

        // Process Q heads: RMS norm + gamma + RoPE
        for (int h = 0; h < num_q_heads; h++)
        {
            float *head = q + h * head_dim;
            
            // RMS norm
            double sum = 0.0;
            #pragma omp simd reduction(+ : sum)
            for (int i = 0; i < head_dim; i++)
            {
                sum += (double)head[i] * (double)head[i];
            }
            float inv_rms = 1.0f / std::sqrt((float)(sum / head_dim) + 1e-6f);
            
            // Apply gamma and store normalized values
            #pragma omp simd
            for (int i = 0; i < head_dim; i++)
            {
                head[i] = head[i] * inv_rms * gamma_q[i];
            }
            
            // RoPE rotation (in-place)
            #pragma omp simd
            for (int i = 0; i < half; i++)
            {
                float x0 = head[i];
                float x1 = head[i + half];
                head[i] = x0 * cos_vals[i] - x1 * sin_vals[i];
                head[i + half] = x1 * cos_vals[i] + x0 * sin_vals[i];
            }
        }

        // Process K heads: RMS norm + gamma + RoPE
        for (int h = 0; h < num_k_heads; h++)
        {
            float *head = k + h * head_dim;
            
            // RMS norm
            double sum = 0.0;
            #pragma omp simd reduction(+ : sum)
            for (int i = 0; i < head_dim; i++)
            {
                sum += (double)head[i] * (double)head[i];
            }
            float inv_rms = 1.0f / std::sqrt((float)(sum / head_dim) + 1e-6f);
            
            // Apply gamma
            #pragma omp simd
            for (int i = 0; i < head_dim; i++)
            {
                head[i] = head[i] * inv_rms * gamma_k[i];
            }
            
            // RoPE rotation
            #pragma omp simd
            for (int i = 0; i < half; i++)
            {
                float x0 = head[i];
                float x1 = head[i + half];
                head[i] = x0 * cos_vals[i] - x1 * sin_vals[i];
                head[i + half] = x1 * cos_vals[i] + x0 * sin_vals[i];
            }
        }
    }
}
    // FP8, BF8, BF16 Fallback Kernels (Simulated)
    void run_gemv_fp8(const float* __restrict__ vec, const int8_t* __restrict__ mat, const float* __restrict__ scale,
                      float* __restrict__ out, int M, int K) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            float row_scale = scale[i];
            const int8_t* row_mat = &mat[i * K];
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < K; j++) {
                sum += vec[j] * ((float)row_mat[j] * row_scale);
            }
            out[i] = sum;
        }
    }

    void run_gemv_bf8(const float* __restrict__ vec, const int8_t* __restrict__ mat, const float* __restrict__ scale,
                      float* __restrict__ out, int M, int K) {
        // Similar fallback for BF8 (simulated with INT8 math for now in absence of native hardware ops)
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            float row_scale = scale[i];
            const int8_t* row_mat = &mat[i * K];
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < K; j++) {
                sum += vec[j] * ((float)row_mat[j] * row_scale);
            }
            out[i] = sum;
        }
    }

    void run_gemv_bf16(const float* __restrict__ vec, const uint16_t* __restrict__ mat, float* __restrict__ out, int M, int K) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            const uint16_t* row_mat = &mat[i * K];
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < K; j++) {
                uint16_t bf = row_mat[j];
                // Convert bf16 to float32
                uint32_t val = ((uint32_t)bf) << 16;
                float fval;
                std::memcpy(&fval, &val, sizeof(float));
                sum += vec[j] * fval;
            }
            out[i] = sum;
        }
    }
