# 의료 서류 위변조: 생성기 · 탐지기

진료비 계산서·영수증과 진단서를 대상으로 한 프로젝트입니다.

| 폴더 | 무엇 | 상태 |
|---|---|---|
| [**`generator/forgery/`**](generator/forgery/) | **위조 문서 자동 생성기**: 존재하지 않는 서류를 가상 정보로 새로 만듦 (공식 서식 + 심평원 단가 · 상병마스터) | 완료 |
| [**`generator/alteration/`**](generator/alteration/) | **변조 문서 자동 생성기**: 실제 서류에서 한 부분만 바꿈 (획 따라 지우기 → 비슷한 글씨체로 다시 쓰기 → 블렌딩), 정답 마스크 · 라벨 포함 | 완료 |
| [**`detector/`**](detector/) | **위변조 탐지기**: OpenRouter VLM 으로 문서 값을 읽고, **로직(공식 근거 규칙)** 과 **이미지 픽셀(다시 그린 칸)** 두 엔진으로 동시에 검사 | 🚧 **개발 중** |

```
generator/forgery      공식 서식 + 공식 기준 자료 ──→ 가상 진단서 · 영수증 (위조)
generator/alteration   실제 서류 사진 ──→ 한 칸만 바꾼 변조본 + 마스크 + 라벨 (변조)
                                                     │  평가셋
detector/              서류 사진 ──┬─ OpenRouter VLM 문서 값 읽기 → 로직 엔진 ─┐
                                   └─ 이미지 픽셀 엔진 ─────────────────────────┴→ 위변조 의심 / 확인 필요 / 정상 + 근거 위치
```

## 설치

macOS 기준입니다 (로직 엔진의 한글 OCR 이 Apple Vision 을 씁니다).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r detector/logic/requirements.txt pyyaml pymupdf openpyxl
brew install tesseract
```

OpenRouter 키는 환경변수 `OPENROUTER_API_KEY` 나 `.venv/bin/python detector/reader/save_key.py` 로 넣습니다 (`~/.openrouter_key`, 저장소에 들어가지 않음).

## 공개 범위

- 실제 서류 사진 · 라벨 · 평가 결과는 개인정보라 넣지 않았습니다. 코드와 공개 기준 자료(심평원 수가 · 상병마스터 · 공휴일 · 세무서 코드 · 법제처 서식)만 있습니다.
- 위조 생성기의 환자 정보(`generator/forgery/config/patient.yaml`)는 가짜 값입니다.
- 외부 AI(OpenRouter)에는 합성 서류만 보냅니다. `detector/reader/or_client.refuse_real()` 이 실서류로 보이는 경로 · 파일 이름을 보내기 전에 막습니다.
- 윈도우 글꼴은 라이선스 때문에 넣지 않았습니다 ([`generator/alteration/fonts/README.md`](generator/alteration/fonts/README.md)).

## 만든 사람

| 사람 | 맡은 부분 |
|---|---|
| 혜인 ([gksgPdls](https://github.com/gksgPdls)) | 위조 생성기 `generator/forgery`, 로직 엔진 `detector/logic`, 합친 탐지기 `detector/pipeline` (VLM 읽기 · 병렬 결합 · 웹 화면) |
| 허소영 ([soyoungheo429](https://github.com/soyoungheo429)) | 변조 생성기 `generator/alteration/forge`, 픽셀 엔진 `detector/pixel` |
| 김동현 ([nuzz3800](https://github.com/nuzz3800)) | 조작 방법별 변조 `generator/alteration/methods`, VLM 판독기 `detector/reader`, 채점 `detector/eval` |
