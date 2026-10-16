# date_conflict — 끝 날짜를 시작 날짜보다 앞으로

주 탐지 축: **콘텐츠 로직** · ID: `CL_xxx` · 스크립트: [`scripts/auto_receipt.py`](../../scripts/auto_receipt.py) `date_conflict`

## 조작 방식

시작~끝 날짜가 한 칸에 있는 영수증(진료기간, 입원~퇴원)에서 **끝 날짜의 일(日)을 시작 날짜보다 1~5일 앞으로** 바꾼다. 방법론 문서의 케이스 10(날짜 논리 모순)이다. 두 날짜가 같은 달일 때만 한다.

일 두 자리만 바꾸는 방식과 다시 읽기 검증은 [`date_shift`](../date_shift/README.md)와 같다. 검증에서는 시작 날짜가 그대로이고 끝 날짜가 새 값으로 읽히는지 확인한다.

```bash
cd dataset/scripts
../.venv/bin/python auto_receipt.py date_conflict 5 --pool ../_work/pool/DOC205 ../_work/pool/DOC210 --seed 0
```

한 칸에 날짜가 두 개인 영수증은 퇴원 영수증 계열(DOC205)에 많아서 그쪽을 먼저 쓴다.

## 탐지 원리

- **콘텐츠 로직:** 끝(퇴원·종료)일 < 시작(입원·시작)일. 규칙 기반으로 결정적 판정이 가능하고 구현 비용이 낮다. 회의에서 "재촬영해도 논리 모순은 남는다"고 한 케이스라 재촬영 샘플의 재료로도 쓴다
- **글리프:** 다시 쓴 두 숫자의 폰트가 다르다
