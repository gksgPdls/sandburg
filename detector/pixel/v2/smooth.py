"""
A11 '다시 그린 칸' 종합 점수 (2026-11-06).
생성기 v2는 배경 노이즈·밝기는 주변에 맞추지만(그래서 v1 노이즈 검사를 통과), 글자는
  흐린 벡터 글꼴 + 한 가지 잉크색 + 회색 잡음으로 다시 그린다. 그래서 같은 문서의 진짜 금액 칸보다
  - 글자 속 미세 노이즈가 적고(inknoise)          v2 24장 칸 AUC 0.76
  - 경계가 매끈하고(lapvar, 같은 숫자끼리 lap)    0.77 / 0.72
  - 글자 속 색 흔들림이 적다(inkchroma)            0.72
  (inkcorr: 글자 속 잔결 옆 픽셀 상관, 단독 0.74지만 평균에 넣으면 정상 오탐이 늘어 뺌 — 11-06 정상 188장 시험)
세 값을 같은 문서·비슷한 글자 높이·획 굵기(±25%) 칸끼리 '낮은 쪽' robust z로 바꿔 평균한다 (각 z는 -3~8로 자름).
원본·변조 위치는 쓰지 않는다.
"""
import cv2
import numpy as np

import os

from .cellfilter import clean_text, column_numeric, date_like, money_like, peers
from .digit_shape import digit_scores
from .segment import _ink
from .texture import cell_texture

HTOL, SWTOL = 0.2, 0.25     # 비교 칸: 글자 높이 ±20%, 획 굵기 ±25% (굵은 합계 칸처럼 모양이 다른 칸과 섞지 않음, 11-06 시험: 더 좁히면 비교 칸이 적어 오히려 나빠짐)


def ink_chroma(bgr, bbox, pad=2):
    x, y, w, h = [int(v) for v in bbox]
    cr = bgr[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    if cr.size < 200:
        return None
    ink = _ink(cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY))
    if ink.sum() < 15:
        return None
    ycc = cv2.cvtColor(cr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return float(ycc[..., 1:][ink].std(0).mean())


def ink_corr(gray, bbox, pad=2):
    """글자 속 잔결의 옆 픽셀 상관. 생성기가 더한 잡음은 흐림(0.7px)을 거쳐 이웃끼리 더 닮음 -> 높으면 의심"""
    x, y, w, h = [int(v) for v in bbox]
    g = gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    if g.size < 100:
        return None
    ink = _ink(g)
    if ink.sum() < 30:
        return None
    F = g.astype(np.float32)
    hp = F - cv2.GaussianBlur(F, (0, 0), 1.5)
    m = ink[:, :-1] & ink[:, 1:]
    a, b = hp[:, :-1][m], hp[:, 1:][m]
    if a.size < 20 or a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _geom(gray, c):
    """찌그러짐: 잉크 높이(행 진하기 50% 지점 사이, 소수점)와 글자당 폭(칸 폭 / 글자 수 / 잉크 높이). 숫자를 자르지 않음."""
    x, y, w, h = [int(v) for v in c["bbox"]]
    cr = gray[y:y + h, x:x + w].astype(np.float32)
    if cr.size < 32:
        return {}
    bg = np.percentile(cr, 90)
    rows = np.clip(bg - cr, 0, None).sum(1)
    if rows.max() <= 0:
        return {}
    idx = np.nonzero(rows >= 0.5 * rows.max())[0]
    ih = float(idx[-1] - idx[0] + 1)
    cols = np.clip(bg - cr, 0, None).sum(0)
    cidx = np.nonzero(cols >= 0.15 * cols.max())[0]
    iw = float(cidx[-1] - cidx[0] + 1) if cidx.size else float(w)
    t = clean_text(c.get("text"))
    n = sum(ch.isdigit() for ch in t) + 0.4 * sum(ch in ",." for ch in t)
    if n < 2 or ih < 4:
        return {}
    return {"_ih": ih, "_wpc": iw / n / ih}


def _two_z(v, P, rel=0.03):
    P = np.asarray(P, float)
    med = np.median(P); mad = max(np.median(np.abs(P - med)) * 1.4826, 1e-6, rel * abs(med))
    return float(abs(v - med) / mad)


def _low_z(v_self, v_peers, rel=0.05):
    v = np.asarray(v_peers, float)
    med = np.median(v); mad = max(np.median(np.abs(v - med)) * 1.4826, 1e-6, rel * abs(med))
    return float(np.clip((med - v_self) / mad, -3, 8))


TIERS = ((HTOL, SWTOL), (0.3, None), (10.0, None))   # 비교 칸이 모자라면 조건을 넓힘 (11-06 튜닝셋 139장: 칸 커버 66 -> 128/139)


COLNUM = os.environ.get("A11_COLNUM", "1") == "1"   # 금액 열의 형식 깨진 칸도 검사 (11-06)
SQUISH_W = float(os.environ.get("A11_SQUISH", "0"))  # 찌그러짐(글자당 폭·잉크 높이) 가중. 11-06 시험: 넣으면 정상 칸도 튀어서(OCR 글자 수 오류) 고도화 정탐이 줄어 기본은 끔
SQUISH_G = float(os.environ.get("A11_SQG", "2.0"))
DATE_DIGITS = os.environ.get("A11_DATE_DIGITS", "0") == "1"   # 날짜 칸은 숫자 단위로 같은 높이의 같은 숫자와 비교. 11-06 시험: 날짜 위조 0건, 정상 오탐 10.3->11.7% 라 기본은 끔
DATES = os.environ.get("A11_DATES", "0") == "1"   # 날짜 칸도 검사: 11-06 시험에서 정상 오탐만 늘고(10.5->11.4%) 날짜 위조는 못 잡아 기본은 끔


def _kind(c):
    return "date" if date_like(c) else "money"


def _peers(C, F, c, min_peers):
    """먼저 같은 종류(금액끼리, 날짜끼리)에서 단계적으로 찾고, 모자라면 종류 상관없이"""
    same = [o for o in C if _kind(o) == _kind(c)]
    for pool in (same, C):
        P = _peers_in(pool, F, c, min_peers)
        if P is not None:
            return P
    return None


def _peers_in(C, F, c, min_peers):
    f = F[c["id"]]
    for ht, st in TIERS:
        P = [o for o in peers(C, c, tol=ht) if o["id"] in F
             and (st is None or abs(F[o["id"]]["_sw"] - f["_sw"]) <= st * max(f["_sw"], 1e-3))]
        if len(P) >= min_peers:
            return P
    return None


def smooth_scores(bgr, cells, min_peers=3):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    C = [c for c in cells if money_like(c) or (DATES and date_like(c)) or (COLNUM and column_numeric(c, cells))]
    F = {}
    for c in C:
        t = cell_texture(gray, c["bbox"])
        ch = ink_chroma(bgr, c["bbox"])
        if t and ch is not None:
            F[c["id"]] = {"inknoise": t["inknoise"], "lapvar": np.log(t["lapvar"] + 1e-6), "inkchroma": ch,
                          "_sw": t["sw"], "_inkmed": t["inkmed"], **_geom(gray, c)}
    C = [c for c in C if c["id"] in F]
    dig = digit_scores(gray, C)
    if DATE_DIGITS:
        D = [c for c in cells if date_like(c) and c["id"] not in F]
        if D:
            dd = digit_scores(gray, C + D, accept=lambda c: money_like(c) or date_like(c) or c["id"] in F, htol=0.25)
            date_out = {c["id"]: dd[c["id"]] for c in D if c["id"] in dd}
        else:
            date_out = {}
    out = {}
    for c in C:
        P = _peers(C, F, c, min_peers)
        if P is None:
            continue
        keys = ("inknoise", "lapvar", "inkchroma") if F[c["id"]]["_inkmed"] > 35 else ("lapvar", "inkchroma")  # 새까만 획은 잡음이 잘려 inknoise 못 믿음
        d = {k: _low_z(F[c["id"]][k], [F[o["id"]][k] for o in P]) for k in keys}
        if c["id"] in dig:
            d["digit_lap"] = float(np.clip(dig[c["id"]]["lap"], -3, 8))
        d["score"] = float(np.mean(list(d.values())))
        if SQUISH_W > 0 and "_wpc" in F[c["id"]]:
            Pg = [o for o in P if "_wpc" in F[o["id"]]]
            if len(Pg) >= 3:
                zw = _two_z(F[c["id"]]["_wpc"], [F[o["id"]]["_wpc"] for o in Pg])
                zh = _two_z(F[c["id"]]["_ih"], [F[o["id"]]["_ih"] for o in Pg])
                d["squish"] = float(min(max(max(zw, zh) - SQUISH_G, 0.0), 6.0))   # 같은 비교 칸보다 SQUISH_G 넘게 다를 때만
                d["score"] += SQUISH_W * d["squish"]
        d["n"] = len(P)
        out[c["id"]] = d
    if DATE_DIGITS:
        for cid, v in date_out.items():   # 날짜 칸: 칸 질감은 안 보고 숫자 단위 선명도만
            out[cid] = {"date_digit_lap": float(np.clip(v["lap"], -3, 8)), "score": float(np.clip(v["lap"], -3, 8)), "n": 0}
    return out
