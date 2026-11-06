"""
A7 복사-붙여넣기 검사 (2026-11-06). 위치를 모르는 상태에서, 같은 문서 안 두 칸이 '노이즈까지 똑같은지' 본다.
진짜 인쇄는 같은 숫자를 두 번 찍어도 종이 결·카메라 노이즈가 서로 다르다. 복사한 칸은 원래 칸과 픽셀까지 같다.

  from detector.v2.copymove import copymove_scores
  copymove_scores(gray, cells) -> {cell_id: {"score": 1 - 차이, "with": 짝 칸 id, "pairs": 거의 같은 쌍 수}}

1. 금액·날짜·형식 깨진 금액 칸 중 높이 비가 0.75~1.33인 겹치지 않는 두 칸: 한 칸의 글자 부분을 다른 칸 주변 ±8px에서 밀어 보며 가장 잘 맞는 자리를 찾는다 (OCR 상자 흔들림 대응, 11-06)
2. 고주파 잔결(원본 - 가우시안 1px)의 차이 에너지 / 두 칸 에너지 합 = 차이 (0 = 똑같음, 1 = 무관). 가장 작은 값
3. 디지털 문서(PDF 캡처 등)는 같은 값이 원래 픽셀까지 같아 '거의 같은 쌍'(차이 < 0.15)이 여러 개 나온다
   -> 그런 쌍이 DIGITAL_PAIRS개 넘는 문서는 점수를 0으로 (복사인지 원래 같은지 가를 수 없음)
11-06 튜닝셋 시험 (원본 칸 위치로 한 빠른 확인): 차이 < 0.1 & 거의 같은 쌍 1개 이하 → 복사 58%, 정상 2.2%
"""
import cv2
import numpy as np

from .cellfilter import column_numeric, date_like, money_like

DIGITAL_PAIRS = 2


def _crop(gray, b, pad=1):
    x, y, w, h = [int(v) for v in b]
    return gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad].astype(np.float32)


def _diff(a, b, shift=2):
    ha = a - cv2.GaussianBlur(a, (0, 0), 1.0)
    hb = b - cv2.GaussianBlur(b, (0, 0), 1.0)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    B = hb[b.shape[0] - h:, b.shape[1] - w:]
    best = None
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            ay0, ax0 = max(a.shape[0] - h + dy, 0), max(a.shape[1] - w + dx, 0)
            hh, ww = min(h, a.shape[0] - ay0), min(w, a.shape[1] - ax0)
            if hh * ww < 60:
                continue
            A_, B_ = ha[ay0:ay0 + hh, ax0:ax0 + ww], B[:hh, :ww]
            v = float(((A_ - B_) ** 2).sum() / ((A_ ** 2).sum() + (B_ ** 2).sum() + 1e-6))
            best = v if best is None else min(best, v)
    return best


def _hp(a):
    return a - cv2.GaussianBlur(a, (0, 0), 1.0)


def _diff_search(gray, a_box, b_box, shrink=2, margin=8):
    """a 칸 글자 부분(가장자리 shrink px 뺌)을 b 칸 주변(margin px 넓힘)에서 밀어 보며 가장 잘 맞는 자리의 차이 (11-06).
    OCR 상자 크기·위치가 흔들려도 같은 내용이면 맞는 자리를 찾는다."""
    H, W = gray.shape
    ax, ay, aw, ah = [int(v) for v in a_box]
    bx, by, bw, bh = [int(v) for v in b_box]
    t = gray[ay + shrink:ay + ah - shrink, ax + shrink:ax + aw - shrink].astype(np.float32)
    X0, Y0, X1, Y1 = max(bx - margin, 0), max(by - margin, 0), min(bx + bw + margin, W), min(by + bh + margin, H)
    r = gray[Y0:Y1, X0:X1].astype(np.float32)
    if t.shape[0] < 6 or t.shape[1] < 10 or r.shape[0] < t.shape[0] or r.shape[1] < t.shape[1]:
        return None
    T, R = _hp(t), _hp(r)
    sq = cv2.matchTemplate(R, T, cv2.TM_SQDIFF)
    er = cv2.boxFilter(R * R, -1, (t.shape[1], t.shape[0]), normalize=False, borderType=cv2.BORDER_CONSTANT)
    oy, ox = t.shape[0] // 2, t.shape[1] // 2
    er = er[oy:oy + sq.shape[0], ox:ox + sq.shape[1]] if er.shape[0] >= oy + sq.shape[0] and er.shape[1] >= ox + sq.shape[1] else None
    if er is None or er.shape != sq.shape:
        er = np.full_like(sq, float((R * R).mean() * t.size))
    d = sq / (float((T * T).sum()) + er + 1e-6)
    return float(d.min())


def copymove_scores(gray, cells):
    C = [c for c in cells if money_like(c) or date_like(c) or column_numeric(c, cells)]
    crops = {c["id"]: _crop(gray, c["bbox"]) for c in C}
    best, pairs = {}, 0
    for i, a in enumerate(C):
        for b in C[i + 1:]:
            ax, bx = a["bbox"], b["bbox"]
            if not (0.75 <= ax[3] / max(bx[3], 1) <= 1.33):
                continue
            if not (ax[0] + ax[2] <= bx[0] or bx[0] + bx[2] <= ax[0] or ax[1] + ax[3] <= bx[1] or bx[1] + bx[3] <= ax[1]):
                continue
            ds = [v for v in (_diff_search(gray, ax, bx), _diff_search(gray, bx, ax)) if v is not None]
            if not ds:
                continue
            d = min(ds)
            pairs += d < 0.15
            for c, o in ((a, b), (b, a)):
                if d < best.get(c["id"], (9, None))[0]:
                    best[c["id"]] = (d, o["id"])
    digital = pairs > DIGITAL_PAIRS
    return {cid: {"score": 0.0 if digital else 1.0 - d, "with": o, "pairs": int(pairs)} for cid, (d, o) in best.items()}
