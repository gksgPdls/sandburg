"""
칸 안 글자(숫자) 특징 재기 — A1~A4·A6 공통 (2026-11-06).

    from detector.pixel.v2.glyph import cell_features
    f = cell_features(gray, (x, y, w, h))      # gray: 문서 전체 흑백 numpy, bbox: OCR 칸 상자
    f -> {"n_digits", "height", "stroke_w", "ink", "contrast", "edge_w", "halo", "gap", "gap_comma",
          "bottom", "angle", "digit_boxes"}   (못 재면 None)

원본을 쓰지 않는다. 값은 같은 문서의 같은 종류 칸과 비교할 때만 의미가 있다 (checks.py).
모든 길이는 글자 높이로 나누어 크기·해상도 차이를 뺀다. 밝기는 그 칸 둘레 종이 밝기로 나누어 조명 차이를 뺀다.

  height    숫자 높이(px)                         A3
  bottom    숫자 바닥선 y(px, 문서 좌표)            A3 (같은 줄 비교)
  stroke_w  획 두께 / 높이 (거리 변환 중앙선)       A1
  ink       잉크 밝기 / 종이 밝기                  A1
  contrast  (종이 - 잉크) / 종이                   A1
  edge_w    획 가장자리 전이 폭 / 높이 (선명도)     A4
  halo      글자 둘레 2~4px 띠 잔결 / 바깥 종이 잔결 A4
  gap       숫자 사이 틈 중앙값 / 높이               A2
  gap_comma 쉼표 앞뒤 틈 / 높이                     A2
  angle     숫자 바닥점 직선 기울기(도)              A6
"""
import cv2
import numpy as np


def _paper_and_ink(crop):
    """Otsu로 잉크/종이 나누기. (잉크 마스크, 종이 밝기, 잉크 밝기)"""
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    t, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = m > 0
    if ink.sum() < 8 or (~ink).sum() < 8:
        return None, None, None
    return ink, float(np.median(crop[~ink])), float(np.median(crop[ink]))


def _remove_lines(ink):
    """칸 폭 대부분을 가로지르는 가로선·칸 높이를 다 가로지르는 세로선(표 선) 빼기."""
    h, w = ink.shape
    m = ink.astype(np.uint8)
    hl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.6), 8), 1)))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h, 8))),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8)) > 0
    return ink & ~lines


def _segments(ink):
    """연결 요소를 가로로 겹치는 것끼리 합쳐 글자 덩어리로. [(x0, y0, x1, y1, mask)]"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    # 표 선 조각(납작하고 긴 것)은 뺌. 쉼표·점은 폭이 좁아서 남음
    comps = sorted([k for k in range(1, n) if st[k, 4] >= 3 and not (st[k, 3] <= 3 and st[k, 2] >= 6)],
                   key=lambda k: st[k, 0])
    groups = []
    for k in comps:
        a0, a1 = st[k, 0], st[k, 0] + st[k, 2]
        if groups and min(a1, groups[-1][1]) - max(a0, groups[-1][0]) > 0.5 * min(a1 - a0, groups[-1][1] - groups[-1][0]):
            g = groups[-1]; g[0] = min(g[0], a0); g[1] = max(g[1], a1); g[2].append(k)
        else:
            groups.append([a0, a1, [k]])
    out = []
    for a0, a1, ks in groups:
        sel = np.isin(lab, ks)
        ys, xs = np.nonzero(sel)
        out.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, sel))
    return out


def _split_wide(sg, Hd, aspect=0.62):
    """붙어 버린 숫자(33, 100)를 나눈다: 폭이 숫자 하나보다 넓으면 예상 개수만큼, 세로 잉크가 가장 적은 열에서 자름."""
    x0, y0, x1, y1, sel = sg
    w = x1 - x0
    n = int(round(w / (aspect * Hd)))
    if w <= 0.85 * Hd or n < 2:
        return [sg]
    col = sel[:, x0:x1].sum(0).astype(float)
    cuts = []
    for i in range(1, n):
        c = int(round(i * w / n))
        lo, hi = max(1, c - int(0.2 * Hd)), min(w - 1, c + int(0.2 * Hd) + 1)
        cuts.append(lo + int(np.argmin(col[lo:hi])) if hi > lo else c)
    out, edges = [], [0] + cuts + [w]
    for a, b in zip(edges[:-1], edges[1:]):
        part = np.zeros_like(sel)
        part[:, x0 + a:x0 + b] = sel[:, x0 + a:x0 + b]
        ys, xs = np.nonzero(part)
        if xs.size:
            out.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, part))
    return out


def _resid_std(patch):
    p = patch.astype(np.float32)
    return float((p - cv2.GaussianBlur(p, (0, 0), 1.0)).std())


def cell_features(gray, bbox, pad=3):
    x, y, w, h = [int(v) for v in bbox]
    H, W = gray.shape[:2]
    X0, Y0, X1, Y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
    crop = gray[Y0:Y1, X0:X1]
    if crop.size < 64:
        return None
    ink, paper, inkv = _paper_and_ink(crop)
    if ink is None:
        return None
    ink = _remove_lines(ink)
    segs = _segments(ink)
    if not segs:
        return None
    hs = np.array([s[3] - s[1] for s in segs], float)
    Hd = float(np.median(hs[hs >= 0.6 * hs.max()]))           # 숫자 높이
    digits = []
    for sg in segs:
        if (sg[3] - sg[1]) < 0.6 * Hd:
            continue
        digits += _split_wide(sg, Hd)
    small = [s for s in segs if (s[3] - s[1]) < 0.6 * Hd]      # 쉼표·점
    if len(digits) < 3 or Hd < 6:
        return None
    dmask = np.zeros_like(ink)
    for s in digits:
        dmask |= s[4]
    # A1 굵기·진하기
    dist = cv2.distanceTransform(dmask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[(dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6) & dmask]
    stroke = float(np.median(2 * ridge)) if ridge.size else None
    core = cv2.erode(dmask.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    inkm = float(np.median(crop[core])) if core.sum() >= 5 else inkv
    # A4 가장자리 전이 폭: 잉크·종이 사이 중간 밝기 픽셀 수 / 가장자리 길이
    lo, hi = inkm + 0.15 * (paper - inkm), paper - 0.15 * (paper - inkm)
    near = cv2.dilate(dmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mid = near & (crop > lo) & (crop < hi)
    perim = cv2.morphologyEx(dmask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).sum()
    edge_w = float(mid.sum() / max(perim, 1))
    # A4 글자 둘레 띠 잔결 / 바깥 종이 잔결
    r2 = cv2.dilate(dmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    r4 = cv2.dilate(dmask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    band = r4 & ~r2 & ~ink
    outer = ~cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    halo = None
    if band.sum() >= 20 and outer.sum() >= 20:
        rs = crop.astype(np.float32) - cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.0)
        so = float(rs[outer].std())
        halo = float(rs[band].std() / so) if so > 0.3 else None
    # A2 자간
    ds = sorted(digits, key=lambda s: s[0])
    gaps = [ds[i + 1][0] - ds[i][2] for i in range(len(ds) - 1)]
    gap = float(np.median(gaps)) / Hd if gaps else None
    gc = []
    for c in small:
        left = [s for s in ds if s[2] <= c[0] + 1]
        right = [s for s in ds if s[0] >= c[2] - 1]
        if left and right:
            gc.append(((c[0] - left[-1][2]) + (right[0][0] - c[2])) / Hd)
    # A3 높이·바닥선, A6 기울기
    bottoms = np.array([(s[0] + s[2]) / 2 for s in ds]), np.array([s[3] for s in ds], float)
    angle = None
    if len(ds) >= 4:
        k = np.polyfit(bottoms[0], bottoms[1], 1)[0]
        angle = float(np.degrees(np.arctan(k)))
    return {"n_digits": len(digits), "height": Hd, "bottom": float(Y0 + np.median(bottoms[1])),
            "stroke_w": stroke / Hd if stroke else None, "ink": inkm / max(paper, 1), "contrast": (paper - inkm) / max(paper, 1),
            "edge_w": edge_w / Hd, "halo": halo, "gap": gap, "gap_comma": float(np.median(gc)) if gc else None,
            "angle": angle,
            "digit_boxes": [(int(X0 + s[0]), int(Y0 + s[1]), int(s[2] - s[0]), int(s[3] - s[1])) for s in ds]}
