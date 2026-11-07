#!/bin/bash
# 탐지기 v2 결과 뷰어 (2026-11-06): 문서별 의심 칸·실제 고친 칸·정탐/오탐과 수치표를 브라우저로.
# 문서 작은 사본을 outputs/평가셋/뷰어/img/ 에 만들어(git 제외) 파일로 바로 열림. 이미지는 이 Mac 밖으로 안 나감.
cd "$(dirname "$0")/../../.."   # 저장소 루트로
.venv/bin/python detector/pixel/scripts/v2/make_viewer.py --check "${1:-A11_다시그린칸}" && open "outputs/평가셋/뷰어/index.html"
