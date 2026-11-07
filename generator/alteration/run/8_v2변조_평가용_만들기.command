#!/bin/bash
# 탐지기 v2 평가용: 고도화 변조기(v2)로 칸 하나만 바꾼 위조 300장 만들기 (튜닝용 150 + 시험용 150, 2026-11-06)
# VLM·VLM 안 씀 (A 검사는 글자·픽셀만 보므로 합계 맞추기 불필요). SD 사용, 장당 20~40초 -> 2~3시간
# 끊겨도 다시 실행하면 이어서 함. 끝나면 detector/pixel/run/7 을 다시 돌리면 평가셋에 자동으로 들어감
cd "$(dirname "$0")/../../.."   # 저장소 루트로
PY=.venv/bin/python
$PY generator/alteration/forge/forge.py --src outputs/평가셋/v2원본_300 --out v2_한칸_평가용 --logic 끔 --limit 0 --no-compare --resume || exit 1
echo "결과: outputs/고도화/v2_한칸_평가용/AI/"
read -r -p "끝. 엔터로 닫기"
