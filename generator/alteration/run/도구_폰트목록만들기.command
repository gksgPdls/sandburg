#!/bin/bash
# 4단계 준비: 맥에 있는 폰트 목록을 outputs/고도화/폰트목록.txt로 저장
cd "$(dirname "$0")/../../.."   # 저장소 루트로
mkdir -p outputs/고도화
{ ls /System/Library/Fonts /System/Library/Fonts/Supplemental /Library/Fonts ~/Library/Fonts 2>/dev/null; } > outputs/고도화/폰트목록.txt
echo "완료: outputs/고도화/폰트목록.txt"
