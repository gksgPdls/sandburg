#!/bin/bash
# 같은 8장을 세 방식(논리 끔 / 이전 논리맞춤 / VLM)으로 최종 고도화 변조(SD)하고 결과를 한 폴더로 합침 (2026-10-30)
# 결과: outputs/고도화/통합_8장/ (비교 이미지, 통합점수.csv)
cd "$(dirname "$0")/../../.."   # 저장소 루트로
PY=.venv/bin/python
$PY generator/alteration/forge/forge.py --src outputs/고도화/원본목록_one_8.csv --out 한칸_8 || exit 1
$PY generator/alteration/forge/forge.py --src outputs/고도화/원본목록_old_8.csv --out 이전논리_8 || exit 1
$PY generator/alteration/forge/forge.py --src outputs/고도화/원본목록_vlm_10.csv --out VLM_8 || exit 1
$PY generator/alteration/forge/4_compare/merge_runs.py --out 통합_8장 --runs 한칸_8:논리끔 이전논리_8:이전논리맞춤 VLM_8:VLM
read -p "끝. 엔터로 닫기"
