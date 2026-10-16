# normal — 정상 대조군 (조작 없음)

주 탐지 축: **없음** (`target_axis=none`) · ID: `NM_xxx` · 스크립트: [`scripts/make_normals.py`](../../scripts/make_normals.py)

`real_docs` 원본을 조작하지 않고 그대로 넣었다. 조작본·마스크는 없다.

- **왜 필요한가:** 조작본만 있으면 "전부 위조"라고 답하는 엔진이 만점을 받는다. 정상 문서에서 엔진이 반응하는 비율(오탐률)을 잰다
- **문서 종류 비율:** 조작본과 비슷하게 맞췄다 (영수증 DOC210 20, 퇴원·세부내역 계열 DOC205 4, 진단서 DOC203 6). 문서 종류만 보고 정상·조작을 맞히지 못하게 하기 위해서다
- **원본 선택:** 조작본의 원본으로 쓴 문서는 빼고 무작위로 골랐다


```bash
cd dataset/scripts
../.venv/bin/python make_normals.py DOC210=20 DOC205=4 DOC203=6 --seed 0
```
