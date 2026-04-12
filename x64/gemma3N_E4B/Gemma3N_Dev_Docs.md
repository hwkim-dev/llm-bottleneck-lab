# Gemma 3N E4B — Inference Optimization & Architectural Docs

이 문서는 Gemma 3N E4B 모델을 커스텀 추론 엔진 (Python + C++ / Vulkan iGPU)에 포팅하고 최적화하는 과정에서 발생한 이슈, 아키텍처 스펙, 메모리 프로파일링 및 최적화 로드맵을 총망라한 문서입니다.

---

## 1. 아키텍처 명세 및 공식 스펙 분해 (HF 기준)
Gemma 3N E4B는 일반적인 LLaMA 베이스 구조에 **Sliding Window와 GQA, AltUp 병목 라우팅**이 섞인 복잡한 구조입니다.

* **num_hidden_layers:** 35
* **hidden_size / intermediate_size:** 2048 / 16384
* **Attention Heads:** Q: 8 / KV: 2 (GQA 4:1 배율)
* **head_dim:** 256
* **KV Cache 공유 (KV Shared Layers):**
  - 총 35개 레이어 중 **마지막 15개(Layer 20~34)는 K/V를 직접 계산하지 않고, 이전 레이어의 KV Cache를 재사용(Reuse)** 합니다.
  - 슬라이딩(Sliding) 레이어는 Layer 18의 KV를, 글로벌(Full) 레이어는 Layer 19의 KV를 참조하도록 강력하게 종속됩니다.
* **비선형 활성화 (Activation):**
  - **Layer 0~9:** `activation_sparsity = 0.95`. 즉, 상위 5% 외에는 모두 잘라내는 Gaussian Top-K 적용 후 `GELU` 통과.
  - **Layer 10~34:** Sparsity 없이 일반 `GELU` 통과.
* **V_norm 주의사항:**
  - 공식 코드는 V Projection 이후 `RMSNorm`을 수행하지만(gamma 없는 순수 정규화), 현재 추출된 MMAP INT4 가중치는 V_norm 배제 상태로 튜닝되었기 때문에 이를 강제로 적용하면 모델의 출력 스케일이 망가지는(환각) 현상을 발견. (의도적 배제)

---

## 2. 메모리 사용량 (Tensor Profiling)
각 텐서가 VRAM / RAM을 얼마나 차지하는지 분석한 로드맵입니다 (Layer 1개 기준 및 전체 데이터).

| 구성요소 | shape | 타입(Type) | 용량(MB) | 비고 |
|---|---|---|---|---|
| **W_q / W_o** | 2048 x 1024 | `INT4` (uint8 packed) | 각 70.2 MB | |
| **W_k / W_v** | 512 x 1024 | `INT4` (uint8 packed) | 각 17.5 MB | GQA 메모리 압축 |
| **W_gate / W_up** | 16384 x 1024 | `INT4` (uint8 packed) | 각 562.1 MB | **모델에서 가장 거대한 부분** |
| **W_down** | 2048 x 8192 | `INT4` (uint8 packed) | 560.2 MB | |
| **W_embed** | 262400 x 1024 | `INT4` (uint8 packed) | 257.2 MB | 어휘 사전 |
| **W_ple** | 262144 x 4480 | `INT4` (uint8 packed) | 1121.0 MB | PLE 보정 대용량 매트릭스 |

> **총평:** 가장 무거운 FFN 구간(Gate, Up, Down)의 가중치가 전체 파라미터의 대부분(계산량 65% 이상)을 차지합니다.

---

## 3. 디버깅 및 최적화 역사 (History)

### 🚨 1. INT8 Feature Map 양자화 실패 및 교훈
* **목표:** NPU 환경을 시뮬레이션하기 위해 Activation(Feature Map)을 INT8로 양자화 (`_qf` 함수 도입)
* **문제점:**
  1. **정확도 붕괴 (일본어/외계어 환각):** 모델 내(특히 FFN `hidden` 캐시)에 존재하는 엄청나게 거대한 **특이값(Outliers)** 때문에 단 하나의 기준점(127 스케일)으로 모든 소수점 데이터를 압축하자 미세 정보가 전부 0으로 뭉그러짐.
  2. **성능 하락:** Python Numpy 단에서 `np.percentile` 등 정렬과 통계 연산을 수행하자 GPU를 쓰는 이점을 다 덮어버릴 만큼 순수 딜레이 발생.
* **해결 및 결론:**
  - 수학적 한계로, SmoothQuant 없이 단순 INT8 Activation 적용은 기각.
  - 모드를 **W4A32 (또는 W4A16)** 즉, 가중치는 INT4, 연산은 FP16/FP32로 유지하도록 복구하여 성능 안정화.

### 🧠 2. FP16 메모리 최적화 및 C++ Fused Kernel 재건
* **C++ GQA 퓨즈드 커널 (`my_accelerator.cpp`)**
  - 기존 수동 비트 조작으로 FP16 → FP32 변환을 하던 과정에서 Subnormal Float 오류가 발생.
  - GCC의 하드웨어 전용 타입인 **`__fp16`** 을 직접 사용하여 캐스팅하도록 커널(C++)을 개편, 완벽한 정확도와 속도 확보.
* **KV Cache FP16 절약**
  - Attention KV Cache를 `float16`으로 유지한 상태로 직접 C++ 커널에 투입, **대략 350MB ~ 1.4GB 이상의 RAM을 성공적으로 다이어트**. (`2800MB → 2442MB` 달성)

---

## 4. 현재 병목 및 향후 최적화 방향 (Next Step)
가벼운 파이썬 타이머로 측정한 레이어당 처리 시간 지표를 분석했습니다 (Ryzen 4500U, Vega 6).

| Module | 소요 시간(ms) | 비율(%) |
|---|---|---|
| **FFN (Gate/Up/Down)** | **1559.6 ms** | **65.4%** |
| o_proj | 450.7 ms | 18.9% |
| qkv | 185.4 ms | 7.8% |
| ple_proj | 128.6 ms | 5.4% |

**병목 진단:**
가장 무거운 FFN 연산에서 초당 1.1GB/s 수준의 처리량밖에 나오지 않고 있습니다. 이는 `APU (통합형 그래픽)` 구조상 Vulkan으로 데이터를 던지고 CPU가 다시 넘겨받기 위해 대기(Synchronous Lock)하는 오버헤드가 전체 메모리 대역폭을 심각하게 제한하고 있기 때문입니다.

**🚀 향후 최적화 솔루션 제안:**
Vulkan(iGPU) 오프로드를 고집하기보다, **CPU의 AVX2 (또는 C++ OpenMP를 활용한 최적화된 INT4 행렬곱)** 을 구현하는 것이 훨씬 효율적입니다. CPU-GPU 간의 데이터 직렬화 손실 없이 L2/L3 캐시에서 즉시 연산해버리면, M-Chip (Mac)과 유사한 구조적 이득을 통해 FFN 처리 시간을 1/3 이상 줄일 수 있습니다.
