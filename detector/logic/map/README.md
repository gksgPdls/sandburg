# 로직 엔진 지도 (판)

로직 엔진(`detector/logic`)이 **무엇을 근거로 어떻게 판단하는지**를 한 장의 판으로 보여 주는 화면을 만든다.
끌고 확대하는 판 위에 로직 엔진 → 기준 묶음(A 서식 산식·합계 · B 수가·가산 · C 상병코드 · D 날짜·번호 · 주목 칸) → 묶음별 탐지 결과 예시 카드가 놓인다.

- 묶음 장: **근거**(요양급여규칙 별지 제6호서식, 심평원 상병마스터·수가 고시, 공휴일 규정, 국세청·주민번호 검증번호, 표준약관 …) · **흐름** · **기준**(규칙·등급)
- 예시 카드: 실제 서류 예시를 **정탐 · 이유 다름 · 오탐 · 미탐 · 정답** 탭으로 골라 봄. 서류 위에 얇은 상자로 표시
  - 초록 = 실제로 바뀐 곳, 빨강 = 탐지기가 의심한 칸, 보라 = 주목 칸(보험금이 걸린 칸)
- VLM 판독에는 칸 위치가 없어서, 읽은 값을 로컬 OCR 글자 위치에서 찾아 상자를 그린다 (`locate_local.py`, 외부 호출 없음)

## 이 저장소에 없는 것

회사 실제 서류와 거기서 나온 것은 **올리지 않는다** (`.gitignore`): 서류 사진(`img/`), 만들어진 판(`index.html`), 판독 결과(`*.jsonl`, `*.json`),
변조본 폴더(`b2_fee/` `b2_full/` `b6_dates/`), 예시 목록 `examples_local.py`(실제 서류의 금액·날짜·코드 설명).
`examples_sample.py` 를 복사해 `examples_local.py` 로 만들고 내 데이터의 문서 번호로 채운다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `build_board.py` | 판 만들기 → `index.html` + `img/`. 예시는 `examples_local.py` 에서 읽음 |
| `template_board.html` | 판 화면 틀 (배치 · 탭 · 색 · 끌기/확대) |
| `locate_local.py` | VLM 이 읽은 값 → 사진 위 위치 (로컬 OCR) |
| `run_logic.py` | 실제 서류 폴더를 로컬 OCR 로직 엔진으로 일괄 검사 |
| `make_b6_dates.py` | B6 예시용: 공휴일 진료 영수증의 진료일만 앞 평일로 옮긴 변조본 |
| `make_b2.py` · `make_b2_full.py` | B2 예시용: 진찰료를 부풀린 변조본 (합계까지 다 맞춘 것 포함) |
| `scan_holiday.py` | 토·일·공휴일 진료 영수증 찾기 |
| `catalog.py` | 규칙 목록 |
| `_compat.py` | 로직 엔진 패키지 찾기 (팀 저장소 `logic_detector.logic` 또는 이 저장소 `detector.logic`) |

스크립트 안의 데이터 경로(`~/Desktop/SCRYN/...`)는 만든 사람 맥 기준이라 각자 바꿔 쓴다.

```bash
python build_board.py && python -m http.server 8772    # http://localhost:8772
```
