import os

docs_dir = "/home/hwkim/Desktop/github/llm-lite/docs"
target_file = os.path.join(docs_dir, "Gemma3N_Reference_Manual.md")

# 파일 분류
theory_files = ["Attention_RoPE.md", "FFN_Sparsity.md", "PLE_LAuReL.md"]
pipeline_file = "Gemma3N_Pipeline_EN.md"
code_file = "GEMMA_3N_E4B.md"

def read_file(name):
    with open(os.path.join(docs_dir, name), 'r') as f:
        return f.read()

try:
    merged = "# Gemma 3N E4B Comprehensive Reference Manual\n\n"
    merged += "이 문서는 Gemma 3N E4B의 수학적 아키텍처 이론, 전체 데이터 파이프라인 수식, 그리고 저수준(C++/Vulkan) 코드 구현 명세서를 하나로 통합한 공식 매뉴얼입니다.\n\n"

    merged += "---\n\n## 📚 PART 1: Core Architecture & Mechanisms\n\n"
    merged += "Gemma 3N 모델을 구성하는 핵심 특수 메커니즘들입니다.\n\n"
    for f in theory_files:
        merged += read_file(f) + "\n\n---\n\n"

    merged += "## 📐 PART 2: Operational Pipeline\n\n"
    merged += read_file(pipeline_file) + "\n\n---\n\n"

    merged += "## 💻 PART 3: Code Implementation & Acceleration Layer\n\n"
    merged += read_file(code_file) + "\n\n"

    with open(target_file, "w") as out:
        out.write(merged)

    # 오래된 파일 삭제
    all_files = theory_files + [pipeline_file, code_file]
    for f in all_files:
        os.remove(os.path.join(docs_dir, f))

    print(f"✅ [성공] 5개의 문서가 '{target_file}'로 완벽히 병합되었으며 기존 파일들은 삭제되었습니다.")
    
except Exception as e:
    print(f"❌ [에러] 파일 병합 중 오류가 발생했습니다: {e}")
