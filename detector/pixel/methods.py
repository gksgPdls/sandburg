"""
탐지법 정의. 탐지법 하나 = 함수 하나 + METHODS에 등록.

모든 탐지법은 같은 형식으로 측정값을 돌려준다:
    {"score": 판정에 쓰는 값, "inside": 안쪽 측정값, "outside": 비교 대상 측정값}
판정(기준값과 비교)은 여기서 하지 않고 config.json의 규칙으로 core.py에서 한다.
→ 기준값을 바꿀 때 코드 수정 없이 config.json만 고치면 된다.

새 탐지법 추가: 아래처럼 함수를 만들고 METHODS에 등록 → config.json에 규칙 추가.
"""
import numpy as np

from . import features as F


class Context:
    """이미지 1장 + 의심영역에 대한 공통 준비물(여러 탐지법이 재사용, 한 번만 계산)."""

    def __init__(self, rgb_image, mask_arr, bbox, params=None):
        self.rgb = rgb_image
        self.gray = np.asarray(rgb_image.convert("L"))
        self.mask = mask_arr
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.params = params or {}  # 탐지법별 설정 (config.json의 comparison 등)
        self._ranked = None

    @property
    def ranked(self):
        """(안쪽, 비슷한 순 후보 패치들, 후보 엣지밀도) — 한 번만 계산해서 탐지법들이 공유."""
        if self._ranked is None:
            self._ranked = F.ranked_outside_patches(self.gray, self.bbox)
        return self._ranked

    @property
    def patches(self):
        """(안쪽, 가장 비슷한 비교 패치 1개)."""
        inside, ranked, _ = self.ranked
        return inside, ranked[0]

    def comparison(self, key):
        """탐지법 key의 비교 설정(top_k, min_edge_ratio)대로 고른 (안쪽, [비교 패치들])."""
        c = self.params.get(key, {}).get("comparison", {})
        inside, ranked, edges = self.ranked
        return inside, F.select_comparison(inside, ranked, edges,
                                           c.get("top_k", 1), c.get("min_edge_ratio", 0.0))


def ela(ctx):
    """압축흔적_ELA: 다시 JPEG 압축했을 때 의심영역 손실이 마스크 밖 전체 평균보다 큰지."""
    diff = F.ela_diff_map(ctx.rgb)
    inside, outside = diff[ctx.mask > 0], diff[ctx.mask == 0]
    if inside.size == 0 or outside.size == 0:
        return None
    i, o = float(inside.mean()), float(outside.mean())
    return {"score": F.safe_ratio(i, o), "inside": i, "outside": o}


def noise(ctx):
    """신호패턴_노이즈분석: 웨이블릿으로 분리한 노이즈 크기, 안쪽 ÷ 비교 패치들의 중앙값.
    (2026-10-16~ 비교 패치 5개 중앙값 + 빈 종이 칸 제외 — config.json comparison)"""
    inside, outs = ctx.comparison("noise")
    outs = [o for o in outs if o.shape == inside.shape]
    if inside.size == 0 or not outs:
        return None
    i = F.wavelet_noise_std(inside)
    o = float(np.median([F.wavelet_noise_std(p) for p in outs]))
    return {"score": F.safe_ratio(i, o), "inside": i, "outside": o}


def fft(ctx):
    """신호패턴_FFT주파수: 고주파 에너지 비율 차이 (안쪽 - 비슷한 주변 패치)."""
    inside, outside = ctx.patches
    if inside.size == 0 or outside.shape != inside.shape:
        return None
    i, o = F.high_freq_energy_ratio(inside), F.high_freq_energy_ratio(outside)
    return {"score": i - o, "inside": i, "outside": o}


def boundary(ctx):
    """영역불일치_합성경계분석: 마스크 테두리 1px의 밝기 변화 ÷ 배경 전체 평균."""
    grad = F.sobel_magnitude(ctx.gray)
    ring, dilated = F.boundary_ring(ctx.mask > 0)
    if ring.sum() == 0:
        return None
    r = float(grad[ring].mean())
    b = float(grad[~dilated].mean()) if (~dilated).sum() else 0.0
    return {"score": F.safe_ratio(r, b), "inside": r, "outside": b}


def pixel_std(ctx):
    """AI탐지_픽셀통계: 픽셀 밝기 표준편차, 안쪽 ÷ 비슷한 주변 패치.
    (예전 '노이즈지문분석'과 '패치통계이상'은 같은 이 값에 기준만 달랐음 → 하나로 통합)"""
    inside, outside = ctx.patches
    if inside.size == 0 or outside.shape != inside.shape:
        return None
    i, o = float(inside.std()), float(outside.std())
    return {"score": F.safe_ratio(i, o), "inside": i, "outside": o}


METHODS = {
    "ela": ela,
    "noise": noise,
    "fft": fft,
    "boundary": boundary,
    "pixel_std": pixel_std,
}
