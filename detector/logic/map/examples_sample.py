"""판 예시 목록 틀. 복사해서 examples_local.py 로 만들고 내 데이터의 문서 번호로 채운다 (examples_local.py 는 git 에 안 올림).

EX     : (묶음 A~D, 문서 번호, 결과 tp·off·fp·fn·tn, 보여 줄 규칙 또는 None, 한 줄 설명)
B_EX   : (문서 번호, 출처 b6·b2·b2x·b2f, 규칙, 한 줄 설명)   — B(수가·가산) 예시
FX     : (문서 번호, hit·orig, 띠 문구, 설명)                 — 주목 칸 예시
LOCAL_EXTRA : 로컬 OCR 결과만 있을 때 덧붙일 예시 (EX 와 같은 모양)
NOTES  : {묶음: 묶음 장 아래 안내 문장}
"""
EX = []
B_EX = []
FX = []
LOCAL_EXTRA = []
NOTES = {}
