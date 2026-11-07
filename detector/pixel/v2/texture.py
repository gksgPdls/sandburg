"""
칸 질감 비교 (2026-11-06): 고친 칸은 다시 그려져서 글자 안 미세 노이즈·경계 거칠기가 주변 칸보다 '적다'.

    from detector.pixel.v2.texture import texture_scores
    texture_scores(gray, cells) -> {cell_id: {"inknoise": z, "lapvar": z, "con": z, "score": z최대}}

같은 문서 금액·숫자 칸끼리 robust z를 '낮은 쪽만' 본다 (z = (중앙값 - 값) / 1.4826·MAD, 클수록 매끈함).
  inknoise : 잉크 픽셀의 고주파(원본 - 가우시안 1px) 표준편차
  lapvar   : 글자 경계 라플라시안 분산 / 대비²
  con      : 배경 - 잉크 밝기 차
금액 모양 칸만, 글자 높이 ±20% 칸 4개 이상과 비교 (cellfilter). 원본·변조 위치는 쓰지 않는다.
"""
import cv2
import numpy as np

from .cellfilter import money_like, peers
from .segment import _ink

KEYS = ("inknoise", "lapvar", "con")


def cell_texture(gray, bbox, pad=2):
    x, y, w, h = [int(v) for v in bbox]
    cr = gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    if cr.size < 64:
        return None
    ink = _ink(cr)
    if ink.sum() < 15:
        return None
    f = cr.astype(np.float32)
    u8 = ink.astype(np.uint8)
    edge = (cv2.dilate(u8, np.ones((3, 3))) > 0) & ~(cv2.erode(u8, np.ones((3, 3))) > 0)
    bg = ~(cv2.dilate(u8, np.ones((5, 5))) > 0)
    if bg.sum() < 10:
        return None
    con = max(float(np.median(f[bg]) - np.median(f[ink])), 1.0)
    hp = f - cv2.GaussianBlur(f, (0, 0), 1.0)
    lap = cv2.Laplacian(f, cv2.CV_32F)
    dt = cv2.distanceTransform(u8, cv2.DIST_L2, 3)
    sk = dt[(dt >= cv2.dilate(dt, np.ones((3, 3)))) & (dt > 0)]
    sw = float(2 * np.median(sk)) if sk.size else 0.0      # 획 굵기(px)
    return {"inknoise": float(hp[ink].std()), "lapvar": float(lap[edge].var()) / con ** 2, "con": con,
            "sw": sw / max(h, 1), "inkmed": float(np.median(f[ink]))}


def texture_scores(gray, cells, min_peers=4, **_):
    """금액 모양 칸만, 글자 높이가 비슷한 다른 칸들(min_peers개 이상)과 비교"""
    C = [c for c in cells if money_like(c)]
    T = {}
    for c in C:
        t = cell_texture(gray, c["bbox"])
        if t:
            T[c["id"]] = t
    C = [c for c in C if c["id"] in T]
    out = {}
    for c in C:
        P = [o for o in peers(C, c) if o["id"] in T]
        if len(P) < min_peers:
            continue
        d = {}
        for k in KEYS:
            v = np.array([T[o["id"]][k] for o in P], float)
            med = np.median(v); mad = max(np.median(np.abs(v - med)) * 1.4826, 1e-6, 0.05 * abs(med))
            d[k] = float((med - T[c["id"]][k]) / mad)
        d["score"] = max(d[k] for k in KEYS)
        out[c["id"]] = d
    return out
