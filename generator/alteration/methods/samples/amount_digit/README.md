# amount_digit — 금액 자릿수 키우기

주 탐지 축: **콘텐츠 로직** · ID: `CL_xxx` · 스크립트: [`scripts/auto_receipt.py`](../../scripts/auto_receipt.py) `digit`

## 조작 방식

금액 칸 하나의 끝에 `0`을 붙여 10배로 만든다(`15,000` → `150,000`). 방법론 문서의 케이스 2b(자릿수 조작)다. 지우고 다시 쓰는 방식은 `amount_sum_break`와 같다. 새 글자가 칸 안에 들어가지 않으면 그 칸은 건너뛴다.

```bash
cd generator/alteration/methods/scripts
../.venv/bin/python auto_receipt.py digit 5 --seed 0
```

## 탐지 원리

- **콘텐츠 로직:** 합계 불일치에 더해, 같은 항목 대비 금액이 10배라 이상치다
- **글리프:** 다시 쓴 숫자의 폰트가 다르고, 한 자리 늘어 칸 안 글자 폭이 주변과 달라진다
- 합계까지 10배 차이만큼 고치면 산술로는 못 잡고 "금액이 그럴듯한가"를 판단해야 한다 (후속 변형 후보)
