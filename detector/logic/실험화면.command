#!/bin/bash
# 로직 탐지기 실험 화면: 더블클릭하면 서버가 뜨고 브라우저(http://localhost:8766)가 열림. 끝내려면 이 창에서 Ctrl+C
cd "$(dirname "$0")/.."   # 저장소 맨 위로 (이 파일은 logic_detector/ 안에 있음)
PY=.venv/bin/python
[ -x "$PY" ] || PY=../medical-doc-forgery/.venv/bin/python   # 이 레포 가상환경이 없으면 옆 폴더 것 사용
"$PY" logic_detector/web/__main__.py "$@"
