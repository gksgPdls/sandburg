"""
측정 계산식 모음 (판정 없음, 숫자만 계산).

detector/pixel/scripts/v1/_legacy/detectors.py에 있던 계산을 그대로 옮긴 것 — 결과값이 기존과 동일해야 한다.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops
from skimage.restoration import denoise_wavelet

ELA_JPEG_QUALITY = 90


def ela_diff_map(rgb_image):
    """JPEG(품질 90)로 다시 저장했을 때 픽셀별 손실량(채널 평균)."""
    buf = BytesIO()
    rgb_image.save(buf, format="JPEG", quality=ELA_JPEG_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    return np.asarray(ImageChops.difference(rgb_image, resaved)).astype(np.float32).mean(axis=2)


def wavelet_noise_std(gray_patch):
    """웨이블릿 디노이징으로 '내용'과 '노이즈'를 분리해 노이즈 성분의 표준편차만 반환."""
    patch = gray_patch.astype(np.float32) / 255.0
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0.0
    denoised = denoise_wavelet(patch, channel_axis=None, rescale_sigma=True)
    return float((patch - denoised).std() * 255.0)


def high_freq_energy_ratio(gray_patch):
    """FFT 스펙트럼에서 저주파(중심 원) 밖 에너지 비율 = 질감이 얼마나 세밀한지."""
    if gray_patch.size == 0:
        return 0.0
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_patch.astype(np.float32))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    low_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return float(mag[~low_mask].sum() / (mag.sum() + 1e-6))


def _edge_density(patch):
    p = patch.astype(np.float32)
    return float(np.abs(np.diff(p, axis=1)).mean() + np.abs(np.diff(p, axis=0)).mean())


def ranked_outside_patches(gray, bbox, search_radius=80, num_distances=4):
    """의심영역(bbox)과 같은 크기의 주변 후보 패치들을 '비슷한 순서'로 정렬해서 반환.

    후보: 8방향 × 최대 4거리. 비슷함 = 밝기 차이 + 엣지밀도(선의 양) 차이 (각각 후보 내 표준편차로 정규화).
    반환: (안쪽 패치, [후보 패치들], [각 후보 엣지밀도])  — 후보가 없으면 오른쪽 5px 옆 영역 1개.
    """
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    inside = gray[y0:y1, x0:x1]
    if inside.size == 0:
        return inside, [inside], [None]

    candidates = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for step in range(1, num_distances + 1):
            dist = step * max(bw, bh, 10)
            if dist > search_radius + max(bw, bh):
                continue
            ox0, oy0 = x0 + dx * dist, y0 + dy * dist
            ox1, oy1 = ox0 + bw, oy0 + bh
            if ox0 < 0 or oy0 < 0 or ox1 > gray.shape[1] or oy1 > gray.shape[0]:
                continue
            patch = gray[oy0:oy1, ox0:ox1]
            if patch.shape == inside.shape:
                candidates.append(patch)

    if not candidates:
        ox0 = max(min(x1 + 5, gray.shape[1] - bw), 0)
        return inside, [gray[y0:y1, ox0:ox0 + bw]], [None]

    means = np.array([float(p.mean()) for p in candidates])
    edges = np.array([_edge_density(p) for p in candidates])
    mean_scale = means.std() if means.std() > 1e-6 else 1.0
    edge_scale = edges.std() if edges.std() > 1e-6 else 1.0
    dist = (np.abs(means - float(inside.mean())) / mean_scale
            + np.abs(edges - _edge_density(inside)) / edge_scale)
    order = np.argsort(dist, kind="stable")
    return inside, [candidates[i] for i in order], [float(edges[i]) for i in order]


def select_comparison(inside, ranked, edges, top_k=1, min_edge_ratio=0.0):
    """비교 패치 고르기.
    min_edge_ratio: 엣지밀도가 의심영역의 이 비율 미만인 후보(빈 종이 등)는 제외 — 다 제외되면 전체 사용.
    top_k: 남은 후보 중 가장 비슷한 k개 (측정값은 각각 재서 중앙값을 씀)."""
    ie = _edge_density(inside) if inside.size else 0.0
    keep = [i for i, e in enumerate(edges) if e is None or e >= min_edge_ratio * ie] or list(range(len(ranked)))
    return [ranked[i] for i in keep[:top_k]]


def sample_outside_patch(gray, bbox, search_radius=80, num_distances=4):
    """(호환용) 가장 비슷한 비교 패치 1개 = ranked_outside_patches의 1등."""
    inside, ranked, _ = ranked_outside_patches(gray, bbox, search_radius, num_distances)
    return inside, ranked[0]


def sobel_magnitude(gray):
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
                      - g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2])
    gy[1:-1, 1:-1] = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
                      - g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:])
    return np.sqrt(gx ** 2 + gy ** 2)


def boundary_ring(mask_bool):
    """마스크 바로 바깥 1px 테두리(4방향 팽창 - 마스크)."""
    dilated = np.zeros_like(mask_bool)
    dilated[:-1, :] |= mask_bool[1:, :]
    dilated[1:, :] |= mask_bool[:-1, :]
    dilated[:, :-1] |= mask_bool[:, 1:]
    dilated[:, 1:] |= mask_bool[:, :-1]
    dilated |= mask_bool
    return dilated & ~mask_bool, dilated


def safe_ratio(a, b):
    return a / b if b > 1e-6 else float("inf")
