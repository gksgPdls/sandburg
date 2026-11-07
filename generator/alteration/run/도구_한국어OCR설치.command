#!/bin/bash
# 한국어 OCR(EasyOCR) 설치하고 원본 앞 60장을 미리 읽어 둠 (진료비 영수증 칸 이름 찾기용). 모델은 GitHub에서 자동으로 받음
set -e
cd "$(dirname "$0")/../../.."   # 저장소 루트로
exec > >(tee outputs/고도화/실행로그_한국어OCR.txt) 2>&1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
PY=.venv/bin/python
$PY -c "import easyocr" 2>/dev/null || .venv/bin/pip install easyocr "torch==2.8.0" "torchvision==0.23.0"  # torch는 그대로 두고 맞는 torchvision만
$PY generator/alteration/forge/3_logic/ocr_ko.py --src dataset/original/bankbook --limit ${LIMIT:-60}
echo 완료
