"""
칸 안 숫자를 하나씩 잘라 0~9 이름 붙이기 (OCR 글자를 길잡이로, 2026-11-06).

    from detector.pixel.v2.segment import segment_digits
    segs = segment_digits(gray, bbox, "13,160")   # [{"ch": "1", "box": (x, y, w, h), "mask": bool배열(box 크기)}, ...] 또는 None

방법
  1. 칸을 잘라 Otsu로 잉크를 나누고 표 선(칸 폭을 가로지르는 가로선, 칸 높이를 가로지르는 세로선)을 뺀다
  2. 쉼표·점(바닥 쪽 작은 덩어리)을 찾아 OCR 글자의 쉼표 자리와 맞춘다 -> 쉼표 사이 구간마다 숫자 개수가 정해짐
  3. 구간마다 세로 잉크 합이 0인 열로 자연스럽게 나눈 뒤
       덩어리가 많으면(끊긴 0 등) 틈이 가장 좁은 이웃끼리 합치고, 적으면(붙은 33 등) 가장 넓은 덩어리를 잉크가 가장 적은 열에서 자른다
     맨 앞에 큰 틈을 두고 떨어진 덩어리(앞의 '⑤' 같은 번호)는 숫자 개수보다 많을 때 먼저 뺀다
  4. 숫자 폭이 글자 높이의 0.15~1.1배를 벗어나면 잘못 자른 것으로 보고 None (판정에서 뺌)
OCR이 '⑤13,160'을 '013,160'으로 읽은 칸은 앞의 0을 뺀다.
"""
import re

import cv2
import numpy as np


def _norm_text(text):
    t = text.replace(" ", "").replace(".", ",").strip("원")
    t = re.sub(r"[^0-9,\-]", "", t)
    if re.match(r"^0\d", t):
        t = t[1:]
    return t


def _ink(crop):
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (m > 0).astype(np.uint8)
    h, w = ink.shape
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.6), 8), 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h - 2, 8))),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    hl2 = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.35), 8), 1)))
    lines = hl | vl | hl2
    ink[cv2.dilate(lines, np.ones((3, 1), np.uint8)) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink, bool)
    for k in range(1, n):
        x, y, bw, bh, a = st[k]
        if a < 3 or (bh <= 3 and bw >= 6):
            continue
        if (x <= 1 or x + bw >= w - 1) and bw <= 3:      # 칸 가장자리 세로선 조각
            continue
        keep |= lab == k
    return keep


def _blobs(cols):
    """세로 잉크 합이 0인 열로 나눈 덩어리 [(x0, x1)]"""
    out, x0 = [], None
    for i, v in enumerate(list(cols) + [0]):
        if v > 0 and x0 is None:
            x0 = i
        elif v == 0 and x0 is not None:
            out.append([x0, i]); x0 = None
    return out


def _fit(blobs, k, cols, Hd):
    blobs = [list(b) for b in blobs]
    while len(blobs) > k:
        gaps = [blobs[i + 1][0] - blobs[i][1] for i in range(len(blobs) - 1)]
        i = int(np.argmin(gaps))
        blobs[i] = [blobs[i][0], blobs[i + 1][1]]; del blobs[i + 1]
    while len(blobs) < k:
        i = int(np.argmax([b[1] - b[0] for b in blobs]))
        a, b = blobs[i]
        if b - a < 4:
            return None
        lo, hi = a + max(1, int(0.25 * (b - a))), b - max(1, int(0.25 * (b - a)))
        c = lo + int(np.argmin(cols[lo:hi])) if hi > lo else (a + b) // 2
        blobs[i:i + 1] = [[a, c], [c, b]]
    return blobs


_WEIGHT = {"1": 0.6}


def _assign(blobs, part, cols):
    """붙은 숫자 나누기: 글자별 예상 폭(1은 좁게)으로 덩어리마다 몇 글자인지 정하고,
    예상 경계 근처(±15%)에서 잉크가 가장 적은 열로 자른다. 덩어리가 글자보다 많으면 틈이 좁은 것부터 합친다."""
    blobs = [list(b) for b in blobs]
    k = len(part)
    while len(blobs) > k:
        gaps = [blobs[i + 1][0] - blobs[i][1] for i in range(len(blobs) - 1)]
        i = int(np.argmin(gaps))
        blobs[i] = [blobs[i][0], blobs[i + 1][1]]; del blobs[i + 1]
    wts = [_WEIGHT.get(c, 1.0) for c in part]
    unit = sum(b - a for a, b in blobs) / sum(wts)
    out, ci = [], 0
    for bi, (a, b) in enumerate(blobs):
        left = len(blobs) - bi - 1
        take, acc = 0, 0.0
        while ci + take < k - left:
            nxt = wts[ci + take] * unit
            if take > 0 and acc + 0.5 * nxt > (b - a):
                break
            acc += nxt; take += 1
        if bi == len(blobs) - 1:
            take = k - ci
        if take == 0:
            return None
        ws = wts[ci:ci + take]; tot = sum(ws); x = a
        for j in range(take - 1):
            ideal = a + (b - a) * sum(ws[:j + 1]) / tot
            r = max(1, int(0.15 * (b - a) / take))
            lo, hi = int(max(x + 1, ideal - r)), int(min(b - 1, ideal + r + 1))
            c = lo + int(np.argmin(cols[lo:hi])) if hi > lo else int(ideal)
            out.append([x, c]); x = c
        out.append([x, b]); ci += take
    return out if ci == k else None


def segment_digits(gray, bbox, text, pad=2):
    t = _norm_text(text)
    digits = re.sub(r"\D", "", t)
    if len(digits) < 2:
        return None
    x, y, w, h = [int(v) for v in bbox]
    H, W = gray.shape[:2]
    X0, Y0, X1, Y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
    crop = gray[Y0:Y1, X0:X1]
    if crop.size < 64:
        return None
    ink = _ink(crop)
    if ink.sum() < 10:
        return None
    rows = np.nonzero(ink.sum(1))[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    hs = [st[k, 3] for k in range(1, n)]
    if not hs:
        return None
    mx = max(hs)
    Hd = float(np.median([v for v in hs if v >= 0.6 * mx]))
    if Hd < 6:
        return None
    # 쉼표·점: 높이가 숫자의 0.5배 미만이고 아래쪽에 있는 작은 덩어리
    top = np.median([st[k, 1] for k in range(1, n) if st[k, 3] >= 0.7 * Hd]) if any(st[k, 3] >= 0.7 * Hd for k in range(1, n)) else rows.min()
    syms = sorted([(st[k, 0], st[k, 0] + st[k, 2]) for k in range(1, n)
                   if st[k, 3] < 0.5 * Hd and st[k, 1] + st[k, 3] / 2 > top + 0.55 * Hd and st[k, 2] < 0.5 * Hd], key=lambda s: s[0])
    n_sym = t.count(",") + t.count("-")
    if len(syms) > n_sym:     # 남는 작은 점은 버리되, 숫자 사이에 있는 것부터 쓰도록 가운데 쪽을 남김
        syms = sorted(sorted(syms, key=lambda s: -s[1] + s[0])[:n_sym], key=lambda s: s[0]) if n_sym else []
    if len(syms) != n_sym:
        return _fallback(ink, st, n, Hd, top, digits, X0, Y0)
    sym_mask = np.zeros_like(ink)
    for a, b in syms:
        sym_mask[:, a:b] = True
    dink = ink & ~sym_mask
    cols = dink.sum(0)
    parts = re.split(r"[,\-]", t)
    edges = [0] + [s for a, b in syms for s in (a, b)] + [ink.shape[1]]
    out = []
    for pi, part in enumerate(parts):
        k = len(part)
        if k == 0:
            continue
        a, b = edges[2 * pi], edges[2 * pi + 1]
        bl = [[p + a, q + a] for p, q in _blobs(cols[a:b])]
        if not bl:
            return None
        if pi == 0 and len(bl) > k and bl[1][0] - bl[0][1] > 0.5 * Hd:   # 앞에 떨어진 번호
            bl = bl[1:]
        bl = _assign(bl, part, cols)
        if bl is None:
            return None
        for ch, (p, q) in zip(part, bl):
            m = dink[:, p:q]
            ys = np.nonzero(m.sum(1))[0]
            if ys.size == 0:
                return None
            wd = q - p
            if not (0.12 * Hd <= wd <= 1.25 * Hd):
                return None
            y0, y1 = ys.min(), ys.max() + 1
            out.append({"ch": ch, "box": (X0 + p, Y0 + y0, wd, y1 - y0), "mask": m[y0:y1].copy()})
    return out if len(out) == len(digits) else None


def _fallback(ink, st, n, Hd, top, digits, X0, Y0):
    """쉼표를 못 찾은 칸: 작은 바닥 덩어리(쉼표·점 후보)를 모두 지우고 숫자 덩어리만 숫자 개수에 맞춤"""
    dink = ink.copy()
    for k in range(1, n):
        x, y, bw, bh, _ = st[k]
        if bh < 0.6 * Hd:
            dink[y:y + bh, x:x + bw] = False
    cols = dink.sum(0)
    bl = _blobs(cols)
    if not bl:
        return None
    # 쉼표 자리에는 틈이 크므로 숫자 덩어리 합치기는 그대로 _fit에 맡김
    if len(bl) > len(digits) and bl[1][0] - bl[0][1] > 0.5 * Hd:
        bl = bl[1:]
    bl = _assign(bl, digits, cols)
    if bl is None:
        return None
    out = []
    for ch, (p, q) in zip(digits, bl):
        m = dink[:, p:q]
        ys = np.nonzero(m.sum(1))[0]
        wd = q - p
        if ys.size == 0 or not (0.12 * Hd <= wd <= 1.25 * Hd):
            return None
        y0, y1 = ys.min(), ys.max() + 1
        out.append({"ch": ch, "box": (X0 + p, Y0 + y0, wd, y1 - y0), "mask": m[y0:y1].copy()})
    return out
