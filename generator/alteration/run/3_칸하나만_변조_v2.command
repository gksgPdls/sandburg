#!/bin/bash
# 위조 이미지 만들기 (AI 인페인팅 방식 전용. 동현님 방식은 dataset/scripts 코드를 따로 실행). 더블클릭하면 터미널이 열리고 사진 폴더(또는 목록 CSV)를 물어봄
# 같은 일을 터미널에서: .venv/bin/python scripts/generate/forge.py --src <폴더 또는 CSV> --out <결과 이름> [--limit N]
# 자세한 건 docs/고도화_사용법.md
set -e
cd "$(dirname "$0")/.."   # run/ 에서 저장소 루트로
mkdir -p outputs/고도화
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
PY=.venv/bin/python
[ -x "$PY" ] || { echo ".venv 가 없음. docs/고도화_사용법.md 1절 준비부터"; read -p "Enter로 닫기"; exit 1; }
$PY -c "import cv2" 2>/dev/null || .venv/bin/pip install opencv-python
$PY -c "import skimage" 2>/dev/null || .venv/bin/pip install scikit-image
$PY -c "import rapidocr_onnxruntime" 2>/dev/null || .venv/bin/pip install rapidocr-onnxruntime

if [ -z "$SRC" ]; then
  echo "원본 사진 폴더(또는 목록 CSV)를 이 창에 끌어다 놓고 Enter:"
  read -r SRC
  SRC="${SRC%\"}"; SRC="${SRC#\"}"; SRC="${SRC%\'}"; SRC="${SRC#\'}"; SRC="${SRC% }"
fi
if [ -z "$OUT" ]; then
  read -r -p "결과 폴더 이름 (Enter면 '위조결과'): " OUT
  OUT=${OUT:-위조결과}
fi
if [ -z "$LIMIT" ]; then
  read -r -p "몇 장 만들까요? (Enter면 10, 전부는 0): " LIMIT
  LIMIT=${LIMIT:-10}
fi
exec > >(tee "outputs/고도화/실행로그_$OUT.txt") 2>&1
$PY scripts/generate/forge.py --src "$SRC" --out "$OUT" --limit "$LIMIT" $EXTRA
echo "결과: outputs/고도화/$OUT/ (위조 이미지는 AI/, 비교는 비교이미지/)"
open "outputs/고도화/$OUT" 2>/dev/null || true
