# logic_detector — 진료비 영수증·진단서 로직 위변조 탐지기

서류 사진 1장을 넣으면 글자를 읽고(판독), 공식 근거 규칙(서식 산식, 심평원 단가, 상병마스터, 날짜·번호 체계)과 대조해
**정상 / 확인 필요 / 위변조 의심** 셋 중 하나로 판정합니다.

- 위변조 의심: 정상 서류에서는 나올 수 없는 모순 (확정·강함 등급)
- 확인 필요: 정상 서류도 어긋날 수 있는 규칙이 걸렸거나 판독이 불확실 (참고 등급, 이유 표시)
- 어떤 규칙을 적용·제외했는지: [logic/규칙_적용기준.md](logic/규칙_적용기준.md)

## 폴더

```
logic_detector/logic/        탐지기 본체
  __init__.py             입구: inspect(사진) → 판정·근거
  ocr.py                  OCR 두 개 (Apple Vision = 한글, RapidOCR = 숫자). 두 OCR이 다르게 읽은 칸은 '판독 불확실'
  read_receipt.py         진료비 계산서·영수증 판독
  read_diagnosis.py       진단서 판독
  reading.py              판독값 형식 (칸 이름·값·위치·불확실)
  refdata.py              공식 데이터 불러오기
  rules.py                규칙 A 산식 · B 수가·부담률 · C 상병마스터 · D 날짜·번호
  focus.py                규칙 F: 보험금이 걸린 진단서 칸
  data/                   심평원 단가표 2024~2026, 상병마스터 2025.9, 부담률·공휴일·세무서 코드, F 규칙 근거
  규칙_적용기준.md          적용·제외 기준
logic_detector/web/          실험 화면 (이 맥 안에서만 도는 서버, http://localhost:8766)
logic_detector/demo/         실험 화면 샘플 사진·원본·정답.csv (사진은 git에 안 올라감)
logic_detector/실험화면.command   더블클릭하면 실험 화면이 열림
```

> 11-06 폴더 정리: `detector_v2/` → `logic_detector/` (소영 님 `detector/v2` 와 이름이 헷갈려서), 맨 위 `demo/` · `탐지기_실험화면.command` 를 이 폴더 안으로. 코드는 이름만 바뀜

평가 스크립트·설계 문서는 실행에 필요 없어 이 브랜치에는 넣지 않았습니다.
OCR 결과는 `outputs/detect_v2/ocr_cache/`에 자동으로 다시 쌓입니다.

## 설치·실행

맥에서만 돕니다 (한글 OCR이 macOS Apple Vision).

```bash
python3 -m venv .venv && .venv/bin/pip install -r logic_detector/requirements.txt
```

```python
from logic_detector.logic import inspect
r = inspect("영수증.jpg")   # 진료비 계산서·영수증 / 진단서 자동 판별
r["verdict"]                # 정상 / 확인 필요 / 위변조 의심
r["findings"]               # [{rule, grade(확정·강함·참고), title, boxes, expected, actual, source, caveat}]
r["focus"]                  # 보험금이 걸린 칸 — '여기를 보라'
```

실험 화면: `logic_detector/실험화면.command` 더블클릭, 또는 `.venv/bin/python logic_detector/web/__main__.py`
