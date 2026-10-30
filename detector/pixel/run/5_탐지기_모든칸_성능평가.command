#!/bin/bash
# 탐지기를 좌표 없이 문서의 모든 글자 칸(RapidOCR)에 돌려서 성능 다시 재기 (2026-10-30)
# 끊겨도 다시 실행하면 이어서 함. 결과: outputs/reports/locate_evaluation_rapid.json
cd "$(dirname "$0")/../scripts/detect"   # run/ 에서
export LOCATE_OCR=rapid
../../.venv/bin/python evaluate_locate.py --collect --max-seconds 100000 || exit 1
echo; echo "===== 모든 글자 칸 ====="
../../.venv/bin/python evaluate_locate.py --only all
echo; echo "===== 숫자 있는 칸만 ====="
../../.venv/bin/python evaluate_locate.py --only digits
echo; echo "===== 금액·날짜 모양 칸만 ====="
../../.venv/bin/python evaluate_locate.py --only money
read -p "끝. 엔터로 닫기"
