#!/bin/bash
# A 검사(픽셀·글자) 평가 준비: 평가셋 만들기 -> 모든 이미지 칸 찾기(RapidOCR) -> 지금 탐지기 기준선 성능 (2026-10-30)
# 끊겨도 다시 실행하면 이어서 함. 결과: outputs/평가셋/ (manifest.csv, ocr.jsonl, 평가결과_<날짜>.csv)
cd "$(dirname "$0")/../../.."   # 저장소 루트로
PY=.venv/bin/python
$PY -c "import rapidocr_onnxruntime" 2>/dev/null || .venv/bin/pip install rapidocr-onnxruntime
echo "===== 1/3 평가셋 만들기 (v1 위조본은 원본과 같은 JPEG 압축으로 다시 저장) ====="
$PY detector/pixel/scripts/v2/build_evalset.py || exit 1
echo; echo "===== 2/3 모든 이미지 칸 찾기 (RapidOCR, 2300장 정도, 1시간 안팎) ====="
$PY detector/pixel/scripts/v2/ocr_cache.py --workers 4 || exit 1
echo; echo "===== 3/3 기준선: 지금 탐지기(A0) 성능 ====="
$PY detector/pixel/scripts/v2/evaluate.py --checks A0_기존탐지기
read -r -p "끝. 엔터로 닫기"
