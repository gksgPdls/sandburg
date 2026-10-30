#!/bin/bash
# VLM 논리 맞추기 2단계: <사진이름>_vlm.json 답을 계산으로 확인 -> 통과한 것만 위조 이미지 생성
cd "$(dirname "$0")/.."   # run/ 에서 저장소 루트로
PY=.venv/bin/python
read -r -p "이름 (Enter면 vlm): " NAME; NAME=${NAME:-vlm}
$PY scripts/generate/1_vlm/vlm_2_apply_답확인.py --name "$NAME"
echo "확인 결과: outputs/고도화/_vlm/$NAME/확인결과.csv"
read -r -p "통과한 것으로 위조 이미지 만들까요? (y/n): " GO
[ "$GO" = y ] && $PY scripts/generate/forge.py --src "outputs/고도화/원본목록_$NAME.csv" --out "VLM_$NAME"
