"""탐지기 v1: 픽셀 기반 (노이즈·FFT·픽셀 통계). 칸 위치를 주면 그 칸을 같은 문서의 비슷한 주변 칸과 비교한다. README.md 참고."""
from .core import detect, judge, load_config, measure  # noqa: F401
