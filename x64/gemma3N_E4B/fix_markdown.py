import re
import os

file_path = "/home/hwkim/Desktop/github/llm-lite/docs/Gemma3N_Reference_Manual.md"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 언어 없는 빈 백틱(```)을 ```text 로 변경
# (줄의 시작이 ``` 이고 뒤에 아무것도 없는 경우)
content = re.sub(r'^```\s*$', '```text', content, flags=re.MULTILINE)

# 2. Mermaid 다이어그램 도형 텍스트 안보임 수정 (다크모드 지원을 위해 color:#000 추가)
content = re.sub(r'(style\s+\w+\s+fill:#[a-fA-F0-9]+,stroke:#[a-fA-F0-9]+(:?,stroke-width:\d+px)?)$', 
                 r'\1,color:#000', 
                 content, 
                 flags=re.MULTILINE)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ [성공] 마크다운 가독성 및 다크/라이트 모드 정상 지원 패치가 완료되었습니다!")
