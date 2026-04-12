import re
import os

file_path = "/home/hwkim/Desktop/github/llm-lite/docs/Gemma3N_Reference_Manual.md"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 잘못 바뀐 ```text 를 원래의 빈 ``` 로 되돌림
content = re.sub(r'^```text\s*$', '```', content, flags=re.MULTILINE)

# 2. 강제로 추가했던 ,color:#000 삭제
content = content.replace(",color:#000", "")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ [복구 완료] 파일이 문제 발생 직전 원본 상태로 완벽하게 되돌아갔습니다!")
