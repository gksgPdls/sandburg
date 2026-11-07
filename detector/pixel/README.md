# detector — 문서 위변조 탐지기

| 폴더 | 내용 | 상태 |
|---|---|---|
| `v1/` | 픽셀 기반: 칸 위치를 주면 그 칸의 노이즈·FFT·픽셀 통계를 같은 문서의 비슷한 주변 칸과 비교. 기준값 `v1/config.json`, 자세한 기준·성능은 `v1/README.md` | 완료 (2026-10-16) |
| `v2/` | 글자·픽셀 일관성 (A 검사): 굵기·자간·높이·획 테두리·흰 박스·기울기·카피앤무브 거르기·형식 규칙. 같은 문서의 같은 종류 칸끼리 비교, 원본 없이 문서 전체 검사 | 개발 중 (2026-11~) |
| `common/` | 공통: 글자 칸 찾기 `ocr.py` (RapidOCR·tesseract), 같은 종류 칸 묶기 `cells.py` | |
| `web/` | 웹 화면 (`detector/pixel/run/6_탐지기_웹화면.command`) | v1 |

예전 이름은 그대로 동작한다: `from detector.pixel import detect, measure` → v1, `from detector.core import ...` → `detector.v1.core`.

평가 스크립트: v1 `detector/pixel/scripts/v1/` (evaluate, tune, evaluate_locate ...), v2 `detector/pixel/scripts/v2/` (평가셋, 칸 찾기 캐시, 평가).
v1 상태를 그대로 보려면 git 태그 `detector-v1`.
