"""OpenRouter 키를 화면에 보이지 않게 입력받아 ~/.openrouter_key 에 저장한다 (나만 읽기 권한 600).

    .venv/bin/python ai_judge/save_key.py
    → '키를 붙여 넣고 Enter' 가 나오면 키를 붙여 넣는다 (Cmd+V). 입력한 글자는 화면에 보이지 않는다.

키는 저장소 밖(홈 폴더)에만 저장되고, 화면·명령 기록·git 에 남지 않는다. 키를 채팅에 붙여 넣지 않는다.
"""
import getpass
import os
from pathlib import Path

p = Path.home() / ".openrouter_key"
k = getpass.getpass("OpenRouter 키를 붙여 넣고 Enter (화면에 보이지 않습니다): ").strip()
if not k:
    raise SystemExit("입력이 비어 있어 저장하지 않았습니다. 다시 실행해 주세요.")
p.write_text(k, encoding="utf-8")
os.chmod(p, 0o600)
ok = k.startswith("sk-or-") and not any(c.isspace() for c in k)
print(f"저장했습니다 ({p}) · 형식: {'정상 (sk-or- 로 시작)' if ok else '확인 필요 — sk-or- 로 시작하는 키를 다시 붙여 넣어 주세요'}")
