# Gemma 3N E4B 추론 엔진 (LLM-Lite)

이 프로젝트는 **Gemma 3N E4B** 모델을 C++ 및 Vulkan을 결합하여 내장 그래픽 코어(iGPU)와 CPU 환경에서 경량화 및 가속하기 위한 로컬 추론 엔진입니다. INT4로 사전에 압축된 가중치를 MMAP으로 읽어 램을 절약하고, FP16/FP32 커스텀 행렬 연산을 결합해 동작합니다.

## 시스템 요구사항 (System Requirements)
- **운영체제:** Linux (Ubuntu 등)
- **컴파일러:** GCC/G++ (`__fp16` 및 OpenMP 지원)
- **그래픽 가속:** Vulkan 드라이버

## 1. 사전 의존성 설치 (Prerequisites)

먼저 C++ 커널과 Vulkan 셰이더 컴파일을 위한 시스템 필수 패키지가 필요합니다.
```bash
sudo apt update
sudo apt install build-essential libvulkan-dev glslang-tools
```

파이썬 환경에 필요한 패키지는 다음과 같습니다. 가상 환경(`pynq_env`)을 사용하시는 것을 권장합니다.
```bash
pip install -r requirements.txt
```

## 2. 엔진 빌드 (Building Engines)

Python이 호출할 최적화된 C++ 공유 라이브러리(`.so`)를 빌드해야 합니다.
지원하는 하드웨어 (설정된 Target: Ryzen 5 4500U `znver2`) 에 맞춰 컴파일됩니다.

```bash
bash build.sh
```
> **참고:** 빌드가 끝나면 `C_DLL/` 폴더에 `my_accelerator.so`와 `vulkan_core.so`가 생성됩니다.

## 3. 실행 방법 (Running the Inference)

빌드가 완료되었다면 핵심 구동 파일인 `main.py`를 실행하여 챗봇 모드로 진입합니다.

```bash
python3 main.py
```

### 환경 안내 (Configuration Modes)
실행 시 다음과 같은 메뉴가 노출됩니다:
```text
  [Feature Map Mode] (activation precision)
    1) FP32  — Full precision (baseline, recommended)
    2) BF16  — BFloat16 (half bandwidth)
    3) INT8  — 8-bit quantized
    4) INT4  — 4-bit quantized (aggressive)
```

아웃라이어(특이값) 보정 및 모델 품질 확보, 그리고 최적의 실행 속도를 보장하기 위해 현재 **기본값인 `1 (FP32)`을 권장**합니다. 모델 가중치(Weights) 자체는 언제나 RAM을 극적으로 줄이는 **INT4 (MMAP)** 환경으로 구동됩니다.

## 주요 구조 (Architecture)
- `main.py`: 모델 로드, 메모리 프로파일링, 추론 파이프라인 관리 모듈
- `CPU_CORE.py`: C++ 퓨즈드 커널 기능(KV Cache 최적화, RoPE, GQA 포함)과 파이썬 매핑 모듈
- `IGPU_CORE.py`: Vulkan 셰이더 기반 VRAM 데이터 전송 및 가속 관리 모듈
- `safeTensor.py`: 가중치의 가상 RAM 맵핑 (Zero-copy 로딩)
- `Gemma3N_Dev_Docs.md`: 개발 역사, 구조 스펙, 텐서 프로파일링 등에 대한 통합 기술 문서
