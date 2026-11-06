"""문서 위변조 탐지기.

  detector/v1/      픽셀 기반 (노이즈·FFT·픽셀 통계) — 칸 위치를 주면 판정. 기준값 v1/config.json
  detector/v2/      글자·픽셀 일관성 (A 검사) — 같은 문서의 같은 종류 칸끼리 비교 (개발 중)
  detector/common/  공통: 글자 칸 찾기(ocr), 같은 종류 칸 묶기(cells)
  detector/web/     웹 화면

예전 코드가 쓰던 이름은 그대로 동작한다 (2026-11-06 폴더 정리 전과 같음):
  from detector import detect, measure, judge, load_config   -> v1
  from detector.core / methods / features / locate import ... -> detector.v1.*
"""
import sys as _sys

from .v1.core import detect, judge, load_config, measure  # noqa: F401
from .v1 import core, features, locate, methods  # noqa: F401
from .common import cells  # noqa: F401

for _name, _mod in (("core", core), ("methods", methods), ("features", features), ("locate", locate), ("cells", cells)):
    _sys.modules[f"{__name__}.{_name}"] = _mod
