#!/bin/bash
# 4단계 후보 글꼴 확인: 맥에서 찾은 글꼴 목록을 outputs/고도화/글꼴후보.txt로 저장
cd "$(dirname "$0")/../../.."   # 저장소 루트로
.venv/bin/python -c "
import sys; sys.path.insert(0,'generator/alteration/forge'); import genpath; import font_match as FM
c=FM.candidates(); open('outputs/고도화/글꼴후보.txt','w').write('\n'.join(f'{n} | {p}' for p,i,n in c)); print(len(c),'개')"
echo 완료
