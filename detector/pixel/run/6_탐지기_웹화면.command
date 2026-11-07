#!/bin/bash
# 더블클릭하면 탐지기 웹 화면이 브라우저로 열림. 끄려면 이 터미널 창에서 Ctrl+C 또는 창 닫기.
cd "$(dirname "$0")/../../.."   # 저장소 루트로
./.venv/bin/python -m detector.pixel.web
