"""
A5 흰 박스(덧칠) 검사 (2026-11-06): 칸 안 종이와 칸 바로 바깥(테두리 고리) 종이를 비교.
v1 위조(티나는 변조)는 칸을 단색에 가까운 박스로 덮고 글자를 써서
  - 칸 안 종이 잔결(고주파 표준편차)이 바깥보다 작고        noise = log(바깥/안)
  - 칸 안 종이 밝기가 바깥과 다르다                          bright = |안 중앙값 - 바깥 중앙값|
글자·표 선은 Otsu 잉크(5px 넓힘)로 빼고 잰다. 원본·변조 위치는 쓰지 않는다.
첫 측정(정상 77장 vs v1 위조칸, 고리 8px 하나): 칸 AUC noise 0.74, bright 0.78
"""
import cv2
import numpy as np

from .cellfilter import money_like


def ring_features(gray, bbox, offsets=(2, 8, 16, 26), width=6):
    """칸(OCR 상자) 안 종이 vs 바깥 고리 종이. 덧칠 박스가 OCR 상자보다 크면 가까운 고리도 박스 안이라
    고리를 여러 거리(offsets px 바깥, 폭 width)로 재고 차이가 가장 큰 고리를 쓴다."""
    x, y, w, h = [int(v) for v in bbox]
    H, W = gray.shape
    R = max(offsets) + width
    X0, Y0, X1, Y1 = max(x - R, 0), max(y - R, 0), min(x + w + R, W), min(y + h + R, H)
    cr = gray[Y0:Y1, X0:X1].astype(np.float32)
    if cr.size < 100:
        return None
    blur = cv2.GaussianBlur(cr, (3, 3), 0).astype(np.uint8)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.dilate((m > 0).astype(np.uint8), np.ones((5, 5))) > 0
    hp = cr - cv2.GaussianBlur(cr, (0, 0), 1.0)
    lx, ly = x - X0, y - Y0
    inside = np.zeros_like(dark)
    inside[ly + 1:ly + h - 1, lx + 1:lx + w - 1] = True
    a = inside & ~dark
    if a.sum() < 20:
        return None
    ni, mi = float(hp[a].std()), float(np.median(cr[a]))

    def box(d):
        b = np.zeros_like(dark)
        b[max(ly - d, 0):ly + h + d, max(lx - d, 0):lx + w + d] = True
        return b
    best = None
    for d in offsets:
        ring = box(d + width) & ~box(d) & ~dark
        if ring.sum() < 20:
            continue
        no = float(hp[ring].std())
        mo = float(np.median(cr[ring]))
        f = {"noise": float(np.log((no + 0.05) / (ni + 0.05))), "bright": abs(mi - mo), "bright_s": mi - mo, "ring": d}
        if best is None or max(f["noise"] / 0.38, f["bright"] / 5.0) > max(best["noise"] / 0.38, best["bright"] / 5.0):
            best = f
    return best


def whitebox_scores(gray, cells, kinds=("money", "date")):
    out = {}
    for c in cells:
        if not (money_like(c) or c.get("kind") in kinds):
            continue
        f = ring_features(gray, c["bbox"])
        if f:
            f["score"] = max(f["noise"] / 0.38, f["bright"] / 5.0)   # 정상 칸 2876개의 95% 지점(0.38, 5.0)으로 나눠 단위를 맞춤
            out[c["id"]] = f
    return out


def box_edges(gray, bbox, R=30):
    """덧칠 박스 테두리: 글자를 지운(inpaint) 종이에서 칸 둘레에 '곧게 이어진 밝기 계단'이 있는지.
    왼·오른쪽은 칸 높이 구간의 세로 방향 중앙값 기울기, 위·아래는 칸 폭 구간의 가로 방향 중앙값 기울기를 보고
    가장 센 계단 두 개의 세기(배경 기울기 흔들림 대비) 합을 돌려준다. 표 선(어두운 선)은 inpaint로 지워서 빼고 잰다."""
    x, y, w, h = [int(v) for v in bbox]
    H, W = gray.shape
    X0, Y0, X1, Y1 = max(x - R, 0), max(y - R, 0), min(x + w + R, W), min(y + h + R, H)
    cr = gray[Y0:Y1, X0:X1]
    if cr.shape[0] < 2 * h or cr.shape[1] < w + 10:
        return None
    blur = cv2.GaussianBlur(cr, (3, 3), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.dilate(m, np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(cr, dark, 3, cv2.INPAINT_TELEA).astype(np.float32)
    clean = cv2.GaussianBlur(clean, (0, 0), 1.0)
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3) / 8
    lx, ly = x - X0, y - Y0
    ok = dark == 0
    def prof(g, axis, lo, hi):
        sl = g[lo:hi, :] if axis == 0 else g[:, lo:hi].T
        mk = ok[lo:hi, :] if axis == 0 else ok[:, lo:hi].T
        v = np.where(mk, sl, np.nan)
        good = np.mean(mk, axis=0) > 0.6
        p = np.nanmedian(v, axis=0)
        p[~good] = 0
        return np.nan_to_num(p)
    pc = prof(gx, 0, ly + 1, ly + h - 1)          # 열마다
    pr = prof(gy, 1, lx + 1, lx + w - 1)          # 행마다
    noise = 1.4826 * np.median(np.abs(np.concatenate([pc, pr]))) + 0.15
    sides = []
    for p, a, b in ((pc, 0, lx + 3), (pc, lx + w - 3, len(pc)), (pr, 0, ly + 2), (pr, ly + h - 2, len(pr))):
        seg = np.abs(p[a:b])
        sides.append(float(seg.max()) / noise if seg.size else 0.0)
    s = sorted(sides, reverse=True)
    return {"edge": s[0] + s[1], "sides": [round(v, 1) for v in sides]}


def box_shape(gray, bbox, search=(-2, 8)):
    """덧칠 박스 모양 확인 (11-06): 글자를 지운 종이에서, 칸(OCR 상자) 네 변 바로 바깥 search px 안에
    '곧게 이어진 밝기 계단'이 있는지. 색 칠한 칸·음영은 표 칸 전체에 걸쳐 글자 상자 근처에 계단이 없음.
    변마다: 그 변 위치를 search 범위에서 옮겨 보며, 변을 따라 기울기 부호가 같은 비율(c)과 평균 세기(m) 중 m·c가 가장 큰 곳.
    안쪽이 밝으면 +, 어두우면 -. 네 변 중 같은 방향 세 변 이상의 세기(3번째로 큰 값)를 돌려줌 (배경 기울기 흔들림으로 나눔)."""
    x, y, w, h = [int(v) for v in bbox]
    H, W = gray.shape
    R = search[1] + 4
    X0, Y0, X1, Y1 = max(x - R, 0), max(y - R, 0), min(x + w + R, W), min(y + h + R, H)
    cr = gray[Y0:Y1, X0:X1]
    if cr.shape[0] < h + 6 or cr.shape[1] < w + 6:
        return None
    blur = cv2.GaussianBlur(cr, (3, 3), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.dilate(m, np.ones((3, 3), np.uint8))
    clean = cv2.GaussianBlur(cv2.inpaint(cr, dark, 3, cv2.INPAINT_TELEA).astype(np.float32), (0, 0), 0.8)
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3) / 8
    ok = dark == 0
    lx, ly = x - X0, y - Y0
    noise = 1.4826 * float(np.median(np.abs(np.concatenate([gx[ok], gy[ok]])))) + 0.1

    def side(g, fixed_axis, pos_list, lo, hi, sign):
        best = (0.0, 0.0)
        for p in pos_list:
            if fixed_axis == "x":
                if not (0 <= p < g.shape[1]): continue
                v, k = g[lo:hi, p], ok[lo:hi, p]
            else:
                if not (0 <= p < g.shape[0]): continue
                v, k = g[p, lo:hi], ok[p, lo:hi]
            v = v[k] * sign
            if v.size < 4: continue
            mm = float(np.median(v)); c = float(np.mean(np.sign(v) == np.sign(mm))) if mm else 0.0
            if abs(mm) * c > abs(best[0]) * best[1]:
                best = (mm, c)
        return best[0] * best[1] / noise
    d0, d1 = search
    s = [side(gx, "x", [lx - d for d in range(d0, d1 + 1)], ly + 1, ly + h - 1, +1),          # 왼쪽: 안쪽(오른쪽)이 밝으면 +
         side(gx, "x", [lx + w - 1 + d for d in range(d0, d1 + 1)], ly + 1, ly + h - 1, -1),  # 오른쪽
         side(gy, "y", [ly - d for d in range(d0, d1 + 1)], lx + 1, lx + w - 1, +1),          # 위
         side(gy, "y", [ly + h - 1 + d for d in range(d0, d1 + 1)], lx + 1, lx + w - 1, -1)]  # 아래
    pos = sorted([v for v in s if v > 0], reverse=True)
    neg = sorted([-v for v in s if v < 0], reverse=True)
    third = lambda a: a[2] if len(a) >= 3 else 0.0
    return {"box": max(third(pos), third(neg)), "bright_box": third(pos), "sides": [round(v, 2) for v in s]}
