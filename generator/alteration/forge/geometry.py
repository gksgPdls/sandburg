"""B1: 숫자 칸 주변의 기울기 재기와 표 선 빼기.

- 표 선 찾기: OpenCV LSD(선분 검출)로 칸 주변의 모든 각도 직선을 찾는다 (사진 문서의 기울어진 선 포함)
- 표 선 빼기: 찾은 긴 선분을 굵게 그린 마스크를 글자 획 마스크에서 뺀다
- 글자 밑선 기울기: 숫자 획 덩어리마다 아랫점을 모아 직선을 맞춘다 (fitLine)
- 글자 옆 기울어짐: 획을 여러 각도로 기울여 보고 세로 획이 가장 곧게 서는 각도를 찾는다
"""
import cv2
import numpy as np

_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector()
    return _LSD


def ink_mask(gray_box):
    paper = float(np.median(gray_box))
    return (gray_box < paper - 35).astype(np.uint8) * 255


def find_lines(gray, box, margin=None, min_len=None):
    """칸(box) 주변에서 긴 직선 선분 [(x1, y1, x2, y2, 각도)]. 원본 좌표.
    min_len 기본: 칸 높이의 2배와 30px 중 큰 값 (숫자 획보다 확실히 긴 것만)."""
    x, y, w, h = box
    m = margin if margin is not None else max(h, 20)
    H, W = gray.shape
    x0, y0, x1, y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, W), min(y + h + m, H)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    lines = _lsd().detect(crop)[0]
    if lines is None:
        return []
    min_len = min_len or max(2 * h, 30)
    out = []
    for (a, b, c, d) in np.asarray(lines).reshape(-1, 4):
        L = float(np.hypot(c - a, d - b))
        if L < min_len:
            continue
        ang = float(np.degrees(np.arctan2(d - b, c - a)))
        out.append((a + x0, b + y0, c + x0, d + y0, ang, L))
    return out


def line_mask(shape, lines, thickness=3):
    m = np.zeros(shape[:2], np.uint8)
    for a, b, c, d, *_ in lines:
        cv2.line(m, (int(round(a)), int(round(b))), (int(round(c)), int(round(d))), 255, thickness)
    return m


def doc_angle(lines):
    """가로에 가까운 선분들의 길이 가중 중앙 각도(도). 선이 없으면 None."""
    hs = [(((ang + 90) % 180) - 90, L) for *_, ang, L in lines if abs(((ang + 90) % 180) - 90) < 20]
    if not hs:
        return None
    angs = np.array([a for a, _ in hs]); ws = np.array([l for _, l in hs])
    order = np.argsort(angs); cw = np.cumsum(ws[order])
    return float(angs[order][np.searchsorted(cw, cw[-1] / 2)])


def stroke_mask(gray, box):
    """칸 안 글자 획 마스크 (표 선은 어느 각도든 빼고). (획 마스크, 선 마스크) 칸 좌표 기준."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w])
    lines = find_lines(gray, box)
    lm = line_mask(gray.shape, lines, thickness=3)[y:y + h, x:x + w]
    # 예전 방식(칸 폭 대부분을 가로지르는 수평·수직 획)도 함께 뺀다
    if w >= 8:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.7), 8), 1))
        lm |= cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if h >= 8:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.9), 8)))
        lm |= cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    lm = cv2.dilate(lm, np.ones((3, 3), np.uint8))
    m[lm > 0] = 0
    # 선을 빼고 남은 칸 가장자리 부스러기(아주 작은 덩어리) 제거
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        if st[i, 4] <= 2:
            m[lab == i] = 0
    return m, lm


def baseline_angle(mask):
    """획 덩어리별 아랫점으로 맞춘 밑선 각도(도)와 맞춘 점 수. 숫자보다 작은 덩어리(쉼표 등)는 뺀다."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 2:
        return None, 0
    hs = st[1:, 3]
    tall = np.median(hs[hs >= np.percentile(hs, 50)])
    pts = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 0.7 * tall:
            continue
        ys, xs = np.nonzero(lab == i)
        pts.append((float(xs.mean()), float(ys.max())))
    if len(pts) < 3:
        return None, len(pts)
    vx, vy, _, _ = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return float(np.degrees(np.arctan2(vy, vx))), len(pts)


def slant_angle(mask, max_deg=15, step=1):
    """세로 획이 가장 곧게 서는 기울임 각도(도). 양수는 글자 윗부분이 오른쪽으로 기운 것."""
    h, w = mask.shape
    if mask.sum() == 0 or h < 6:
        return 0.0
    best, best_s = 0.0, -1
    pad = int(np.ceil(h * np.tan(np.radians(max_deg)))) + 2
    src = cv2.copyMakeBorder((mask > 0).astype(np.float32), 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
    for deg in np.arange(-max_deg, max_deg + 1e-6, step):
        t = np.tan(np.radians(deg))
        M = np.float32([[1, t, -t * (h - 1)], [0, 1, 0]])   # 아랫줄 고정, 윗줄을 옆으로 민다
        sh = cv2.warpAffine(src, M, (src.shape[1], h))
        score = float((sh.sum(axis=0) ** 2).sum())
        if score > best_s:
            best_s, best = score, deg
    return float(-best)


def measure(gray, box):
    """칸의 기하 정보: 문서 기울기(표 선), 글자 밑선 기울기, 글자 옆 기울어짐."""
    lines = find_lines(gray, box)
    m, _ = stroke_mask(gray, box)
    base, npts = baseline_angle(m)
    return {"doc_angle": doc_angle(lines), "baseline_angle": base, "baseline_pts": npts,
            "slant": slant_angle(m), "n_lines": len(lines)}


# ---------- B2-2: 원근 (비스듬히 찍은 측면 효과) ----------

def _line_eq(seg):
    """선분 -> 직선 (a, b, c), ax + by = c (단위 법선)."""
    x1, y1, x2, y2 = seg[:4]
    a, b = y2 - y1, x1 - x2
    n = np.hypot(a, b) or 1.0
    a, b = a / n, b / n
    return a, b, a * x1 + b * y1


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    return np.array([(c1 * b2 - c2 * b1) / d, (a1 * c2 - a2 * c1) / d], np.float32)


def _y_at(seg, x):
    x1, y1, x2, y2 = seg[:4]
    if abs(x2 - x1) < 1e-6:
        return None
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _x_at(seg, y):
    x1, y1, x2, y2 = seg[:4]
    if abs(y2 - y1) < 1e-6:
        return None
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def cell_quad(gray, box):
    """숫자 칸을 둘러싼 표 선 4개로 만든 사다리꼴 네 모서리 [좌상, 우상, 우하, 좌하] (원본 좌표).
    위·아래 가로선은 꼭 있어야 하고, 좌·우 세로선이 없으면 숫자 영역 양옆에서 가로선 기울기에 수직인 선으로 대신한다.
    못 찾으면 None. 두 번째 값은 찾은 방식 설명."""
    x, y, w, h = box
    m = max(2 * h, 30)
    lines_h = find_lines(gray, box, margin=m, min_len=max(int(0.8 * w), 25))
    lines_v = find_lines(gray, (x - 3 * w, y, 7 * w, h), margin=m, min_len=max(int(1.2 * h), 12))   # 세로선은 칸 높이 정도로 짧음
    cx, cy = x + w / 2, y + h / 2
    hor = [l for l in lines_h if abs(((l[4] + 90) % 180) - 90) < 20]
    ver = [l for l in lines_v if abs(((l[4] + 90) % 180) - 90) > 70]
    above = [(cy - _y_at(l, cx), l) for l in hor if _y_at(l, cx) is not None and _y_at(l, cx) < y + 0.3 * h]
    below = [(_y_at(l, cx) - cy, l) for l in hor if _y_at(l, cx) is not None and _y_at(l, cx) > y + 0.7 * h]
    if not above or not below:
        return None, "위·아래 가로선 없음"
    top = min(above, key=lambda t: t[0])[1]
    bot = min(below, key=lambda t: t[0])[1]
    left = [(cx - _x_at(l, cy), l) for l in ver if _x_at(l, cy) is not None and _x_at(l, cy) < x]
    right = [(_x_at(l, cy) - cx, l) for l in ver if _x_at(l, cy) is not None and _x_at(l, cy) > x + w]
    lt, rt = _line_eq(top), _line_eq(bot)
    how = "가로선 2 + 세로선 2"
    if left and right and min(left)[0] < 3 * w and min(right)[0] < 3 * w:
        ll = _line_eq(min(left, key=lambda t: t[0])[1])
        rl = _line_eq(min(right, key=lambda t: t[0])[1])
    else:
        # 세로선이 없으면 숫자 양옆에서, 위·아래 가로선의 평균 방향에 수직인 선
        how = "가로선 2 (세로선 없음)"
        ang = np.radians((top[4] + bot[4]) / 2 if abs(top[4] - bot[4]) < 90 else top[4])
        dx, dy = np.cos(ang), np.sin(ang)
        a, b = dx, dy   # 법선 = 가로 방향
        ll = (a, b, a * (x - 2) + b * cy)
        rl = (a, b, a * (x + w + 2) + b * cy)
    pts = [_intersect(lt, ll), _intersect(lt, rl), _intersect(rt, rl), _intersect(rt, ll)]
    if any(p is None for p in pts):
        return None, "교차점 없음"
    q = np.array(pts, np.float32)
    # 상식 검사: 사다리꼴이 숫자 영역을 감싸고, 너무 찌그러지지 않아야 함
    if not (q[:, 0].min() <= x + 2 and q[:, 0].max() >= x + w - 2 and q[:, 1].min() <= y + 2 and q[:, 1].max() >= y + h - 2):
        return None, "사다리꼴이 숫자를 안 감쌈"
    if (q[:, 1].max() - q[:, 1].min()) > 6 * h or (q[:, 0].max() - q[:, 0].min()) > 12 * w:
        return None, "사다리꼴이 너무 큼"
    return q, how


def perspective_strength(quad):
    """사다리꼴이 평행사변형에서 벗어난 정도(도): 위·아래 변 각도 차이, 좌·우 변 각도 차이 중 큰 값."""
    tl, tr, br, bl = quad
    ang = lambda p, q: np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
    d1 = abs(ang(tl, tr) - ang(bl, br))
    d2 = abs(((ang(tl, bl) - ang(tr, br)) + 180) % 360 - 180)
    return float(max(d1, d2))


def line_curve(gray, box, min_span=2.0):
    """B6: 칸 위아래 표 선의 휘어짐. 휘어진 종이를 찍으면 표 선이 직선이 아니라 완만한 곡선이 된다.
    칸 주변 띠에서 가로로 긴 선 픽셀을 모아 2차 곡선 y = a(x - xc)^2 + b(x - xc) + c 를 맞추고,
    칸 위 선과 아래 선의 a를 평균한다. (a, 칸 폭 절반에서 처지는 정도 px, 쓴 선 수). 선이 없으면 (0, 0, 0)."""
    x, y, w, h = box
    H, W = gray.shape
    bx0, bx1 = max(x - w, 0), min(x + 2 * w, W)
    by0, by1 = max(y - 4 * h, 0), min(y + 5 * h, H)
    band = gray[by0:by1, bx0:bx1]
    if band.size == 0 or band.shape[1] < 10:
        return 0.0, 0.0, 0
    paper = cv2.GaussianBlur(band, (0, 0), 15).astype(np.float32)
    ink = ((paper - band.astype(np.float32)) > 30).astype(np.uint8) * 255
    # 휘어진 선도 남도록 짧은 가로 막대로 거르고(세로로 살짝 불려서), 길게 이어진 덩어리만 선으로 본다
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    hl = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(hl)
    xc = x + w / 2
    fits = []
    for i in range(1, n):
        cw = st[i, cv2.CC_STAT_WIDTH]
        if cw < min_span * w or st[i, cv2.CC_STAT_HEIGHT] > max(4 * h, 8):
            continue
        ys, xs = np.nonzero((lab == i) & (ink > 0))
        # 선 두께 가운데: x마다 y 평균
        xu = np.unique(xs)
        yc = np.array([ys[xs == v].mean() for v in xu])
        X = xu + bx0 - xc
        Y = yc + by0
        if len(X) < 10:
            continue
        p = np.polyfit(X, Y, 2)
        res = np.abs(np.polyval(p, X) - Y)
        if np.median(res) > 1.0:   # 곡선으로도 안 맞으면 선이 아님 (글자 줄 등)
            continue
        cy = np.polyval(p, 0)
        if not (y - 2 * h <= cy <= y + 3 * h):
            continue
        fits.append((p[0], cw))
    if not fits:
        return 0.0, 0.0, 0
    a = float(np.average([f[0] for f in fits], weights=[f[1] for f in fits]))
    return a, a * (w / 2) ** 2, len(fits)

