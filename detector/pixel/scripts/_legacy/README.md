# 예전 탐지 스크립트 (보관용)

2026-10-16 탐지기를 `detector/` 패키지로 모듈화하면서 아래 스크립트들은 새 스크립트로 대체됨.
기록용으로 남겨둠 — 이 폴더에서는 import 경로가 맞지 않아 그대로는 실행되지 않음.

| 예전 | 대체 |
|---|---|
| validate.py (정상 원본 오탐률) | `scripts/detect/evaluate.py` |
| evaluate_detectors.py (혼동행렬 표) | `scripts/detect/evaluate.py` |
| measure_auc.py (탐지법별 AUC) | `scripts/detect/evaluate.py` (AUC 열) |
| analyze_thresholds.py, tune_noise_threshold.py (기준값 분포·튜닝) | `scripts/detect/tune.py` |
| detectors.py (구버전 탐지 함수, 예전 기준값) | `detector/` 패키지 — export_catalog.py도 2026-10-16부터 패키지 사용 |
