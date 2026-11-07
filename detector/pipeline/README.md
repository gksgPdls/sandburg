# pipeline — 합친 탐지기 (VLM 읽기 + 로직 · 픽셀 병렬 · OR 판정)

> 🚧 개발 중. 전체 설명은 [`detector/README.md`](../README.md)

```
사진 ─┬─ VLM 읽기 (OpenRouter Qwen, 1회) → 판독값(Reading) → 로직 엔진 (detector/logic 규칙) ─┐
      └─ 픽셀 엔진 (detector/pixel v2 A11 '다시 그린 칸', 읽기와 상관없이 바로 시작) ──────────────────┴→ 등급형 OR → 판정 · 근거 · 위치
```

- **읽기는 VLM 만**: 로컬 OCR 판독을 쓰지 않습니다. `detector/reader/ai_reader.READ_PROMPT` 에 로직 규칙이 쓰는 칸(요양기관 종류 · 입원/외래 · 야간/공휴 · 환자 구분 · 주민번호 · 입원/퇴원일)만 덧붙여 한 번에 읽고, 로직 엔진 판독 형식(`detector/logic/reading.py`)으로 바꿉니다
- **두 엔진 병렬**: 픽셀은 VLM 응답을 기다리지 않고 동시에 돕니다. 엔진 코드는 고치지 않고 불러만 씁니다
  - 픽셀 엔진은 글자 칸 **위치**를 찾을 때만 로컬 RapidOCR 를 씁니다 (값을 읽는 데는 안 씀)
- **판정 = OR** (둘 중 하나라도 걸리면 잡음), 근거 세기로 등급만 나눔

| 로직 | 픽셀 | 판정 |
|---|---|---|
| 확정 · 강함 | 아무거나 | 의심 |
| 아무거나 | 정상 오탐 2% 기준 넘음 | 의심 |
| 걸림(참고) | 걸림(10% 기준) | 의심 |
| 걸림(참고) | 안 걸림 | 확인필요 |
| 안 걸림 | 걸림(10% 기준) | 확인필요 |
| 안 걸림 | 안 걸림 | 정상 |

한 번 돌리면 **등급형OR · 순수OR · 로직만 · 픽셀만** 네 가지 답안이 같이 나옵니다 (같은 결과로 계산, 추가 비용 없음).

## 파일

| 파일 | 하는 일 |
|---|---|
| `vlm_read.py` | VLM 읽기 (캐시) · VLM 판독값 → `Reading` 변환 · 걸린 칸 위치 묻기 |
| `engines.py` | 로직 · 픽셀 엔진을 같은 결과 형식(`level` 의심/확인필요/정상/보류, `strong`, `boxes`)으로 감쌈 |
| `decide.py` | 결합 방식 (`graded` 등급형 OR · `or` · `logic` · `pixel`) |
| `pipeline.py` | 서류 한 장: `inspect(사진, 문서종류)` |
| `run.py` | 폴더 · 평가셋 실행기 → 판정 · 답안 4개 · 채점 |
| `web/` | 실험 화면 (`python -m detector.pipeline.web`) |
| `calibrate_pixel.py` | 픽셀 기준값 정하기 (정상 서류만, 로컬) → `pixel_thresholds.json` |

## 실행

```bash
# 0. 한 번만: 키 저장, 픽셀 기준값 (개발지 정상 서류로 — 평가셋으로 정하지 않음)
.venv/bin/python detector/reader/save_key.py
.venv/bin/python detector/pipeline/calibrate_pixel.py detector/eval/evalsets/개발지_v1

# 1. 평가셋 풀기 (예: 처음 10장으로 먼저 확인)
.venv/bin/python detector/pipeline/run.py detector/eval/evalsets/평가셋_v3 --처음 10
.venv/bin/python detector/pipeline/run.py detector/eval/evalsets/평가셋_v3

# 2. 기준값 · 결합 방식을 바꿔 다시 판정 (호출 없음)
.venv/bin/python detector/pipeline/run.py --다시판정 outputs/combine/평가셋_v3_<시각>
```

- 결과: `outputs/combine/<폴더>_<시각>/` — `문서별.jsonl`(두 엔진 원래 결과 · 판독값 · 비용) · `판정.csv` · `답안_<방식>.csv` · `요약.md`(평가셋이면 채점까지)
- 평가셋이면 `<평가셋>/답안/답안_합친_<방식>_<버전>.csv` 로도 복사 → `detector/eval/score_testset.py` · `merge_answers.py` 그대로 사용
- 비용: VLM 읽기 1회/장 (+ 로직이 걸린 서류만 위치 묻기 1회, `--위치없이` 로 끔). 원본 응답은 `outputs/combine/vlm_cache/` 에 남아 **같은 사진을 다시 돌리면 0원**
- 옵션: `--모델 qwen27b` · `--크기 2500` · `-j 6`(VLM 동시 요청) · `--캐시없이` · `--픽셀없이`

## 실험 화면 (detector/logic/web 과 같은 화면)

```bash
.venv/bin/python -m detector.pipeline.web --demo <샘플 폴더>             # http://localhost:8771 — 미리 계산한 결과만 (외부 호출 없음)
.venv/bin/python -m detector.pipeline.web --demo <샘플 폴더> --실시간     # 결과가 없는 사진은 VLM 으로 읽어 검사 (합성 서류만)
```

- 샘플 폴더(git 제외): `samples/` · `originals/` · `정답.csv` · `info.json` · `결과.json`(사진 sha1 → 화면 결과, `detector/pipeline/web/view.py` 형식)
- 오른쪽에 최종 판정, 엔진 두 줄(로직 · 픽셀), 걸린 규칙과 픽셀 흔적 칸. 원본이 있으면 바뀐 곳(초록 점선)과 나란히
- 로직 근거 상자는 VLM 위치 묻기(`vlm_read.locate`)를 거친 결과에만 있음. 픽셀 근거는 언제나 상자가 있음

## 꼭 지킬 것

- **외부 AI 에는 합성 서류만.** 보내기 전에 `or_client.refuse_real()` 이 실서류(dataset · real_docs · 실서류 이름 형식)를 막습니다
- 픽셀 기준값이 없으면 픽셀은 '보류'(판정에 안 들어감)이고, 점수는 저장되니 기준값을 만든 뒤 `--다시판정` 하면 됩니다


## 아직 안 한 것

- 이 실행기(`run.py`)로 OpenRouter 를 직접 부른 적은 아직 없습니다. 처음엔 `--처음 10` 으로 판독값 변환이 맞는지 `문서별.jsonl` 의 `판독` 을 확인하세요
- 저해상도(1,500px) 서류의 VLM 판독 보강
