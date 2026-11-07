#!/bin/bash
# VLM 논리 맞추기 1단계: 사진 폴더 -> 사진마다 OCR 칸 목록 + 프롬프트 (outputs/고도화/_vlm/<이름>/)
cd "$(dirname "$0")/../../.."   # 저장소 루트로
PY=.venv/bin/python
$PY -c "import rapidocr_onnxruntime" 2>/dev/null || .venv/bin/pip install rapidocr-onnxruntime
echo "사진 폴더를 이 창에 끌어다 놓고 Enter:"; read -r SRC; SRC="${SRC%\'}"; SRC="${SRC#\'}"; SRC="${SRC% }"
read -r -p "이름 (Enter면 vlm): " NAME; NAME=${NAME:-vlm}
read -r -p "몇 장 (Enter면 10, 전부는 0): " N; N=${N:-10}
$PY generator/alteration/forge/1_vlm/vlm_1_prep_칸목록.py --src "$SRC" --name "$NAME" --limit "$N"
open "outputs/고도화/_vlm/$NAME" 2>/dev/null || true
