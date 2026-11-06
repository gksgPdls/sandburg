"""
⚠️ 구버전 — 새 코드는 프로젝트 루트의 `detector` 패키지를 쓸 것 (기준값: detector/v1/config.json).
이 파일은 scripts/export/export_catalog.py(카탈로그 생성)가 아직 쓰고 있어서 남겨둠.
여기 THRESHOLDS는 예전 기준이라 detector/v1/config.json과 다름 → 카탈로그 갱신 때 함께 교체 예정.
"""
"""
탐지기 핵심 모듈 — 위조 "생성" 방식과 무관하게, 이미지 한 장 + 의심 영역 좌표만
있으면 돌릴 수 있는 픽셀 레벨 탐지 함수 모음.

각 함수는 (탐지여부: bool, 측정값_안쪽, 측정값_바깥쪽)을 반환한다.
"안쪽"은 의심 영역(bbox) 내부, "바깥쪽"은 바로 옆의 같은 크기 영역(또는 배경) —
즉 전부 "이 영역이 같은 문서의 다른 부분과 다르게 생겼는가"만 보고 판단하며,
메타데이터검사 하나만 예외적으로 원본 이미지가 따로 있어야 한다
(→ 실서비스에선 원본이 없으므로 이 함수는 검증 전용, 실배포 불가 — README 참고).

노이즈라고 판단하는 기준 (임계치 전부 아래에 모아둠, 근거는 analyze_thresholds.py로
정상문서 분포를 직접 재서 백분위수 기준으로 잡은 것):
  - "안쪽/바깥쪽 비율"이 THRESHOLDS 범위를 벗어나면 "이상하다(위조 의심)"로 판단
  - 비율이 1에 가까울수록 "안쪽과 바깥쪽이 똑같이 생겼다" = 정상
  - 비율이 1에서 멀어질수록(너무 크거나 너무 작으면) = 이상 신호
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops
from skimage.restoration import denoise_wavelet

ELA_JPEG_QUALITY = 90

# 임계치 — 지금은 아직 감으로 정한 초기값. validate.py로 측정한 오탐률이
# 노이즈분석/FFT주파수에서 80%대로 나와서 analyze_thresholds.py로 정상문서
# 실제 분포를 재서 재조정하는 작업이 다음 단계로 남아있음.
THRESHOLDS = {
    "ela_ratio_min": 1.5,          # 안쪽 ELA값이 바깥쪽의 몇 배 넘으면 이상한지
    "ela_abs_min": 5.0,            # 그리고 안쪽 ELA값 자체가 최소 이 값 이상이어야 함
    # 웨이블릿 노이즈 비율 기준 (2026-10-02 실측, 286장: 정상 원본 vs 위조본).
    # 위조 영역은 거의 항상 "노이즈가 더 적은(매끈한)" 쪽으로 나와서 아래쪽만 본다.
    # 0.82 = 정상 원본 분포의 10백분위수 → 오탐 약 10%, 탐지 약 66%.
    # (오탐 5%로 낮추려면 0.72 → 탐지 약 63%)
    "noise_ratio_high": float("inf"),  # 위쪽은 사용 안 함 (정상문서에서도 6~11배까지 나옴)
    "noise_ratio_low": 0.82,
    "fft_diff_min": 0.05,          # 고주파 에너지 비율 차이가 이거 넘으면 이상
    "boundary_ratio_min": 1.6,     # 마스크 테두리 그래디언트가 배경의 몇 배 넘으면 이상
    "noise_fp_ratio_high": 1.8,
    "noise_fp_ratio_low": 0.6,
    "patch_clean_ratio_max": 0.65,  # 안쪽이 바깥쪽보다 "너무 매끈"한지(단방향)
}


def mask_bbox_from_mask(mask_arr):
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def compute_ela_detect(image_path, mask_arr):
    """압축흔적_ELA: JPEG로 다시 압축했을 때, 의심영역이 주변보다 손실이 큰지 본다."""
    original = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    original.save(buf, format="JPEG", quality=ELA_JPEG_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.asarray(ImageChops.difference(original, resaved)).astype(np.float32).mean(axis=2)

    inside = diff[mask_arr > 0]
    outside = diff[mask_arr == 0]
    if inside.size == 0 or outside.size == 0:
        return False, 0.0, 0.0
    inside_mean, outside_mean = float(inside.mean()), float(outside.mean())
    detectable = inside_mean > outside_mean * THRESHOLDS["ela_ratio_min"] and inside_mean > THRESHOLDS["ela_abs_min"]
    return detectable, inside_mean, outside_mean


def laplacian_variance(gray_arr):
    """더 이상 노이즈분석에 안 쓰임(웨이블릿으로 교체) — 엣지밀도 계산 등에 재사용 가능해 남겨둠."""
    out = np.zeros_like(gray_arr, dtype=np.float32)
    out[1:-1, 1:-1] = (
        gray_arr[:-2, 1:-1] + gray_arr[2:, 1:-1] + gray_arr[1:-1, :-2] + gray_arr[1:-1, 2:]
        - 4 * gray_arr[1:-1, 1:-1]
    )
    return float(np.var(out))


def wavelet_noise_std(gray_arr):
    """웨이블릿 디노이징으로 "진짜 내용"(글자/선)과 "노이즈"를 분리해서, 노이즈 성분의
    표준편차만 반환한다. 라플라시안과 달리 "복잡한 글자"와 "진짜 센서 노이즈"를
    구분할 수 있다 (라플라시안은 이 둘을 구분 못 하고 똑같이 "울퉁불퉁함"으로 뭉뚱그림)."""
    patch = gray_arr.astype(np.float32) / 255.0
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0.0
    denoised = denoise_wavelet(patch, channel_axis=None, rescale_sigma=True)
    residual = patch - denoised
    return float(residual.std() * 255.0)


def _sample_outside_patch(gray, bbox, search_radius=80, num_distances=4):
    """"바깥쪽" 비교 패치를 고른다.

    예전 방식(바로 옆 5px)은 표의 "옆 칸"을 잡아버려서, 위조 여부와 상관없이
    옆 칸 내용이 다르다는 이유만으로 항상 차이가 크게 나왔다 (오탐률 80%대 원인).

    1차 수정("가장 밝은 빈 종이" 선택 + 3개 평균)은 더 나빠졌다 — 백지는 잉크가
    없어서 안쪽(글자)과는 원천적으로 질감이 다르고, 평균까지 내면 질감 자체가
    인위적으로 뭉개져서 비교가 무의미해짐 (오탐률 100%로 악화, 실측 확인).

    2차 수정(밝기만 매칭)으로 역전은 없어졌지만, 오탐률이 여전히 49%대로 높다 —
    밝기가 같아도 생김새(글자/표선/도장)가 다르면 질감이 전혀 다르기 때문.
    예: 숫자칸, 표 테두리 선, 도장은 평균 밝기가 비슷해도 구조는 완전히 다름.

    지금 방식: 밝기 + "엣지 밀도"(그래디언트 크기 평균, 선이 얼마나 많고 복잡한지)
    두 가지를 함께 매칭한다. 둘 다 비슷한 후보를 고르면 "숫자칸은 숫자칸끼리,
    표선은 표선끼리" 비교될 확률이 올라가서 콘텐츠 종류 차이로 인한 오탐이 준다.
    """
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    inside = gray[y0:y1, x0:x1]
    if inside.size == 0:
        return inside, inside

    candidates = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in directions:
        for step in range(1, num_distances + 1):
            dist = step * max(bw, bh, 10)
            if dist > search_radius + max(bw, bh):
                continue
            ox0 = x0 + dx * dist
            oy0 = y0 + dy * dist
            ox1, oy1 = ox0 + bw, oy0 + bh
            if ox0 < 0 or oy0 < 0 or ox1 > gray.shape[1] or oy1 > gray.shape[0]:
                continue
            patch = gray[oy0:oy1, ox0:ox1]
            if patch.shape != inside.shape:
                continue
            candidates.append(patch)

    if not candidates:
        # 후보를 못 찾으면(이미지 가장자리 등) 예전 방식으로 폴백
        ox0 = max(min(x1 + 5, gray.shape[1] - bw), 0)
        return inside, gray[y0:y1, ox0:ox0 + bw]

    # 안쪽과 "밝기 + 엣지밀도"가 함께 가장 비슷한 후보 하나를 그대로 사용
    # (평균 내지 않음 — 평균 내면 그 패치 고유의 질감/노이즈가 사라져버림)
    def edge_density(patch):
        gx = np.abs(np.diff(patch.astype(np.float32), axis=1)).mean()
        gy = np.abs(np.diff(patch.astype(np.float32), axis=0)).mean()
        return float(gx + gy)

    inside_mean = float(inside.mean())
    inside_edge = edge_density(inside)

    # 밝기값(0~255대)과 엣지밀도(보통 0~30대) 스케일이 달라서, 각각을 후보군
    # 안에서의 분산으로 정규화한 뒤 더해 하나의 거리값으로 합친다.
    means = np.array([float(p.mean()) for p in candidates])
    edges = np.array([edge_density(p) for p in candidates])
    mean_scale = means.std() if means.std() > 1e-6 else 1.0
    edge_scale = edges.std() if edges.std() > 1e-6 else 1.0

    dist = (
        np.abs(means - inside_mean) / mean_scale
        + np.abs(edges - inside_edge) / edge_scale
    )
    best_idx = int(np.argmin(dist))
    outside = candidates[best_idx]
    return inside, outside


def compute_noise_detect(gray, mask_arr, bbox):
    """신호패턴_노이즈분석: 웨이블릿으로 분리한 "진짜 노이즈 성분"이 안쪽/바깥쪽 간 다른지."""
    inside, outside = _sample_outside_patch(gray, bbox)
    if inside.size == 0 or outside.size == 0 or outside.shape != inside.shape:
        return False, 0.0, 0.0
    v_in = wavelet_noise_std(inside)
    v_out = wavelet_noise_std(outside)
    ratio = v_in / v_out if v_out > 1e-6 else float("inf")
    detectable = ratio > THRESHOLDS["noise_ratio_high"] or ratio < THRESHOLDS["noise_ratio_low"]
    return detectable, v_in, v_out


def high_freq_energy_ratio(gray_patch):
    if gray_patch.size == 0:
        return 0.0
    f = np.fft.fft2(gray_patch.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    low_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    total = mag.sum() + 1e-6
    high = mag[~low_mask].sum()
    return float(high / total)


def compute_fft_detect(gray, bbox):
    """신호패턴_FFT주파수 / AI탐지_FFT주파수: 고주파 성분(질감의 세밀함) 차이."""
    inside, outside = _sample_outside_patch(gray, bbox)
    if inside.shape != outside.shape or inside.size == 0:
        return False, 0.0, 0.0
    hf_in = high_freq_energy_ratio(inside)
    hf_out = high_freq_energy_ratio(outside)
    diff = abs(hf_in - hf_out)
    detectable = diff > THRESHOLDS["fft_diff_min"]
    return detectable, hf_in, hf_out


def compute_noise_fingerprint_detect(gray, bbox):
    """AI탐지_노이즈지문분석: 픽셀 표준편차 비율 — AI생성 픽셀엔 센서 노이즈가 없다는 가정."""
    inside, outside = _sample_outside_patch(gray, bbox)
    if inside.shape != outside.shape or inside.size == 0:
        return False, 0.0, 0.0
    std_in, std_out = float(inside.std()), float(outside.std())
    ratio = std_in / std_out if std_out > 1e-6 else float("inf")
    detectable = ratio < THRESHOLDS["noise_fp_ratio_low"] or ratio > THRESHOLDS["noise_fp_ratio_high"]
    return detectable, std_in, std_out


def compute_patch_cleanliness_detect(gray, bbox):
    """AI탐지_패치통계이상: 안쪽이 바깥쪽보다 "너무 매끈한지"만 본다 (단방향)."""
    inside, outside = _sample_outside_patch(gray, bbox)
    if inside.shape != outside.shape or inside.size == 0:
        return False, 0.0, 0.0
    std_in, std_out = float(inside.std()), float(outside.std())
    ratio = std_in / std_out if std_out > 1e-6 else float("inf")
    detectable = ratio < THRESHOLDS["patch_clean_ratio_max"]
    return detectable, std_in, std_out


def compute_boundary_detect(gray, mask_arr):
    """영역불일치_합성경계분석: 마스크 바로 테두리에서 밝기 변화가 갑자기 튀는지."""
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (
        g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:] - g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2]
    )
    gy[1:-1, 1:-1] = (
        g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:] - g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:]
    )
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    mask_bool = mask_arr > 0
    dilated = np.zeros_like(mask_bool)
    dilated[:-1, :] |= mask_bool[1:, :]
    dilated[1:, :] |= mask_bool[:-1, :]
    dilated[:, :-1] |= mask_bool[:, 1:]
    dilated[:, 1:] |= mask_bool[:, :-1]
    dilated |= mask_bool
    boundary_ring = dilated & ~mask_bool

    if boundary_ring.sum() == 0:
        return False, 0.0, 0.0

    ring_mean = float(grad_mag[boundary_ring].mean())
    background_mean = float(grad_mag[~dilated].mean()) if (~dilated).sum() else 0.0
    ratio = ring_mean / background_mean if background_mean > 1e-6 else float("inf")
    detectable = ratio > THRESHOLDS["boundary_ratio_min"]
    return detectable, ring_mean, background_mean


def compute_metadata_detect(original_path, candidate_path):
    """파일정보_메타데이터검사.

    ⚠️ 실서비스 미적용: 원본(original_path)이 있어야 작동하는 유일한 탐지법인데,
    실제 서비스는 의심 문서 1장만 받고 "원본"이 없다. 테스트셋 검증 전용.
    """
    original = Image.open(original_path)
    candidate = Image.open(candidate_path)
    original_has_exif = bool(getattr(original, "_getexif", lambda: None)()) or bool(original.info)
    candidate_has_exif = bool(getattr(candidate, "_getexif", lambda: None)()) or bool(
        {k: v for k, v in candidate.info.items() if k.lower() not in ("dpi",)}
    )
    detectable = original_has_exif and not candidate_has_exif
    return detectable, original_has_exif, candidate_has_exif
