"""
같은 숫자끼리 모양 비교 (0은 0끼리 ~ 9는 9끼리, 위치를 모르는 상태에서, 2026-11-06).

    from detector.v2.digit_shape import digit_scores
    per_cell = digit_scores(gray, cells)    # {cell_id: {"shape": z, "aspect": z, "height": z, "dark": z, "score": z최대, "worst": "3"}}

1. 금액·숫자 칸마다 segment_digits로 숫자를 잘라 0~9 이름을 붙인다 (OCR 글자 기준, 못 자른 칸은 건너뜀)
2. 문서 안 같은 숫자끼리 모은다. 글자 크기가 다른 줄도 있으니 높이로 정규화한 회색 패치(28x20)를 쓴다
3. 숫자 하나마다 '다른 칸'의 같은 숫자들과
     shape  : 1 - NCC(±1칸 이동 중 최대)의 중앙값      -> 모양이 다르면 큼
     aspect : 폭/높이 / height : 같은 칸 나머지 숫자 대비 높이 비 / dark : 잉크 진하기
   를 재고, 같은 숫자 무리 안에서 robust z (|v - 중앙값| / 1.4826·MAD)로 바꾼다 (무리가 4개 이상일 때만)
4. 칸 점수 = 그 칸 숫자들의 z 중 가장 큰 값
5. 같은 숫자 끼리 '경계 선명도'(lap: 글자 경계 라플라시안 분산/대비², 낮을수록 매끈=다시 그린 글자)와
   잉크 노이즈(inkn)도 비교한다. 이 둘은 '낮은 쪽만' 의심 (sharp_z = (중앙값-값)/MAD),
   칸 값은 그 칸 숫자들의 z 중앙값 (칸 전체를 다시 그리므로 숫자 대부분이 같이 매끈해짐)
   v2 24장 첫 측정: 칸 AUC lap 0.72(정상칸 대비)·0.77(같은 문서 안 고친 칸 대비), 모양(shape) 0.57
원본·변조 위치 정보는 쓰지 않는다.
"""
import cv2
import numpy as np

from .cellfilter import money_like
from .segment import segment_digits

PH, PW = 28, 20


def _patch(gray, box):
    x, y, w, h = box
    g = gray[y:y + h, x:x + w].astype(np.float32)
    if g.size == 0:
        return None
    p = 255.0 - cv2.resize(g, (PW, PH), interpolation=cv2.INTER_AREA)
    p -= p.mean()
    n = np.linalg.norm(p)
    return p / n if n > 1e-6 else None


def _sharp(gray, box, mask, P=2):
    x, y, w, h = box
    y0, x0 = max(y - P, 0), max(x - P, 0)
    cr = gray[y0:y + h + P, x0:x + w + P].astype(np.float32)
    m = np.zeros(cr.shape, bool)
    mh, mw = min(mask.shape[0], cr.shape[0] - (y - y0)), min(mask.shape[1], cr.shape[1] - (x - x0))
    m[y - y0:y - y0 + mh, x - x0:x - x0 + mw] = mask[:mh, :mw]
    u = m.astype(np.uint8)
    bg = ~(cv2.dilate(u, np.ones((3, 3))) > 0)
    if bg.sum() < 5 or m.sum() < 5:
        return (np.nan, np.nan)
    con = max(float(np.median(cr[bg]) - np.median(cr[m])), 1.0)
    e = (cv2.dilate(u, np.ones((3, 3))) > 0) & ~(cv2.erode(u, np.ones((3, 3))) > 0)
    lap = cv2.Laplacian(cr, cv2.CV_32F)
    hp = cr - cv2.GaussianBlur(cr, (0, 0), 1.0)
    return (float(np.log(lap[e].var() / con ** 2 + 1e-6)), float(hp[m].std() / con))


def _ncc_best(a, b):
    best = -1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            bb = np.roll(np.roll(b, dy, 0), dx, 1)
            best = max(best, float((a * bb).sum()))
    return best


def _rz(vals, floor=0.03):
    v = np.asarray(vals, float)
    med = np.median(v)
    mad = max(np.median(np.abs(v - med)) * 1.4826, 1e-6, floor * abs(med))
    return np.abs(v - med) / mad


def digit_scores(gray, cells, kinds=None, min_class=5, accept=None, htol=None):
    """accept: 숫자를 모을 칸 고르기 (기본 금액 칸). htol: 경계 선명도를 비교할 때 글자 높이가 ±htol인 같은 숫자끼리만 (11-06, 날짜용)"""
    accept = accept or money_like
    inst = []                      # (cell_id, ch, patch, aspect, hrel, dark, sharp, h)
    for c in cells:
        if not accept(c):
            continue
        segs = segment_digits(gray, c["bbox"], c.get("text", ""))
        if not segs:
            continue
        hmed = np.median([s["box"][3] for s in segs])
        for s in segs:
            p = _patch(gray, s["box"])
            if p is None:
                continue
            x, y, w, h = s["box"]
            g = gray[y:y + h, x:x + w]
            dark = float(np.median(g[s["mask"]])) if s["mask"].any() else np.nan
            inst.append((c["id"], s["ch"], p, w / max(h, 1), h / max(hmed, 1), dark, _sharp(gray, s["box"], s["mask"]), h))
    out = {}
    for ch in "0123456789":
        I = [i for i in inst if i[1] == ch]
        if len(I) < min_class or len({i[0] for i in I}) < 3:
            continue
        shape = []
        for i in I:
            sims = [_ncc_best(i[2], j[2]) for j in I if j[0] != i[0]]
            shape.append(1 - float(np.median(sims)) if sims else np.nan)
        feats = {"shape": shape, "aspect": [i[3] for i in I], "dark": [i[5] for i in I]}
        if ch != "1":
            feats["height"] = [i[4] for i in I]
        zs = {k: _rz(np.nan_to_num(v, nan=np.nanmedian(v)), floor=0.25 if k == "shape" else 0.03) for k, v in feats.items()}
        sz = {}
        for k, idx in (("lap", 0), ("inkn", 1)):
            v = np.array([i[6][idx] for i in I], float)
            if np.isnan(v).all():
                continue
            v = np.nan_to_num(v, nan=np.nanmedian(v))
            if htol:
                zz = []
                for n, i in enumerate(I):     # 같은 숫자 중 글자 높이가 비슷하고 다른 칸에 있는 것끼리
                    o = [m for m, j in enumerate(I) if j[0] != i[0] and abs(j[7] - i[7]) <= htol * i[7]]
                    if len(o) < 3:
                        o = [m for m, j in enumerate(I) if j[0] != i[0]]
                    vv = v[o]; med = np.median(vv); mad = max(np.median(np.abs(vv - med)) * 1.4826, 1e-6, 0.02 * abs(med))
                    zz.append((med - v[n]) / mad)
                sz[k] = np.array(zz)
                continue
            med = np.median(v); mad = max(np.median(np.abs(v - med)) * 1.4826, 1e-6, 0.02 * abs(med))
            sz[k] = (med - v) / mad
        for n, i in enumerate(I):
            d = out.setdefault(i[0], {"shape": 0.0, "aspect": 0.0, "height": 0.0, "dark": 0.0, "score": 0.0, "worst": "",
                                      "_lap": [], "_inkn": []})
            for k, z in zs.items():
                d[k] = max(d[k], float(z[n]))
            for k in sz:
                d["_" + k].append(float(sz[k][n]))
            sc = max(float(zs[k][n]) for k in zs)
            if sc > d["score"]:
                d["score"], d["worst"] = sc, ch
    for d in out.values():
        d["lap"] = float(np.median(d.pop("_lap"))) if d.get("_lap") else 0.0
        d["inkn"] = float(np.median(d.pop("_inkn"))) if d.get("_inkn") else 0.0
        d["smooth"] = d["lap"]
    return out
