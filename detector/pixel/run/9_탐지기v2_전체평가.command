#!/bin/bash
# 탐지기 v2(A 검사) 전체 평가 (2026-11-06). run/7(칸 찾기)·run/8(v2 변조 300장) 다음에 실행. 끊겨도 다시 실행하면 이어서.
# 결과: outputs/평가셋/평가결과_<날짜>.csv + 화면 표 (정상 오탐률, 묶음별 정탐·미탐·재현율·AUC·위치 적중)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "===== 1/4 평가셋 다시 만들기 (run/8 결과 포함) ====="
$PY scripts/detect/v2/build_evalset.py || exit 1
echo; echo "===== 2/4 새 이미지 칸 찾기 (이미 한 건 건너뜀) ====="
$PY scripts/detect/v2/ocr_cache.py --workers 4 || exit 1
echo; echo "===== 3/4 검사별 평가 (A11 다시그린칸 = 주 검사, A5 흰박스, A9 질감, A10 같은숫자, A0 기존) ====="
$PY scripts/detect/v2/evaluate.py --checks A11_다시그린칸,A5_흰박스,A9_질감,A10_같은숫자,A0_기존탐지기
echo; echo "===== 4/4 합쳐 보기 ====="
$PY scripts/detect/v2/combine.py --checks A11_다시그린칸
$PY scripts/detect/v2/combine.py --checks A11_다시그린칸,A0_기존탐지기
read -r -p "끝. 엔터로 닫기"
