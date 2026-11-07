# amount_sum_break — 금액 하나만 변경 (합계 불일치)

주 탐지 축: **콘텐츠 로직** · ID: `CL_xxx` · 스크립트: [`scripts/auto_receipt.py`](../../scripts/auto_receipt.py) `sum_break`

## 조작 방식

진료비 영수증의 금액 칸 하나를 **주변 종이로 지우고(inpaint + 종이 결 잡음) 새 금액을 다시 쓴다.** 흰 박스는 쓰지 않는다. 다른 금액·합계 칸은 그대로 둔다.

- 금액 칸은 로컬 OCR(tesseract) 신뢰도 85 이상인 것만 고른다
- 새 금액은 자릿수가 같고 백 원 단위를 유지한다
- 새 글자는 원본 숫자의 높이·오른쪽 끝·진하기를 재서 맞춘다. 폰트는 맥 기본 폰트(Apple SD Gothic Neo)

```bash
cd generator/alteration/methods/scripts
../.venv/bin/python auto_receipt.py sum_break 5 --seed 0
```

## 탐지 원리

- **콘텐츠 로직:** 정상 영수증에서 금액 하나만 바뀌면 그 칸이 들어가는 합계(열 합계, 진료비 총액 등)와 반드시 어긋난다. 탐지 엔진의 "필드를 읽고 합계를 다시 계산"하는 구조가 잡아야 하는 기준선이다
- **글리프:** 다시 쓴 숫자의 폰트가 주변과 다르다
- **copy-move:** 복사가 없어서 못 잡는다

합계 칸을 OCR로 찾아 확인하지는 않았다. 이 사진 품질에서는 열 전체를 정확히 읽는 경우가 드물다.
