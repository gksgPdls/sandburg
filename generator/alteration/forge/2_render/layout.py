"""C1: 새 값을 원본 글자 자리에 맞춰 놓기, 쉼표·대시·점은 원본 것을 그대로 쓰기.

- 글자 자리: 원래 값의 글자별 위치(OCR 글자 상자 안 실제 획의 가운데)를 잰다.
  새 값의 글자 수와 기호 자리가 같으면 원래 자리 그대로, 다르면 원래 숫자 간격·기호 앞뒤 간격으로
  금액은 오른쪽 끝, 날짜는 왼쪽 끝을 맞춰 놓는다.
- 기호: 원래 값에 있던 기호의 획 모양(밝기)을 그대로 떠서 쓴다. 원래 값에 없으면 같은 문서의 위아래 줄에서 찾는다.
  못 찾으면 글꼴로 그린다(예전 방식).
모든 좌표는 넘겨받은 이미지(원본, 돌린 것, 펼친 것) 기준."""
import numpy as np

SYMBOLS = ",.-"


def _segments(gray, box, ink_mask_fn):
    """칸 안 글자를 획 덩어리로 나눈다 (OCR 글자 상자는 쉼표·대시에서 옆 숫자까지 넓게 잡혀 못 씀, 10/23 시험).
    가로로 많이 겹치는 덩어리는 한 글자로 합친다. 왼쪽부터 [글자 dict]. 각 글자에 기호 추정 종류(kind)."""
    import cv2
    x, y, w, h = box
    m = (ink_mask_fn(gray[y:y + h, x:x + w]) > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 2]
    comps.sort(key=lambda k: st[k, 0])
    groups = []
    for k in comps:
        a0, a1 = st[k, 0], st[k, 0] + st[k, 2]
        if groups:
            g = groups[-1]
            ov = min(a1, g["a1"]) - max(a0, g["a0"])
            if ov > 0.5 * min(a1 - a0, g["a1"] - g["a0"]):
                g["ks"].append(k); g["a0"] = min(g["a0"], a0); g["a1"] = max(g["a1"], a1)
                continue
        groups.append({"ks": [k], "a0": a0, "a1": a1})
    out = []
    for g in groups:
        sel = np.isin(lab, g["ks"])
        ys, xs = np.nonzero(sel)
        out.append({"cx": x + xs.mean(), "x0": x + xs.min(), "x1": x + xs.max() + 1,
                    "y0": y + ys.min(), "y1": y + ys.max() + 1, "sel": sel, "all": m > 0, "box": box})
    if not out:
        return out
    hs = np.array([g["y1"] - g["y0"] for g in out])
    H = float(np.median(hs[hs >= np.percentile(hs, 50)]))
    # 획이 끊겨 숫자 하나가 두 덩어리로 나뉜 것 합치기 ('0', '3'의 가는 부분이 끊김, 10/30 auto00017 '13,100')
    merged = [out[0]]
    for g in out[1:]:
        p = merged[-1]
        tall = (p["y1"] - p["y0"]) >= 0.6 * H and (g["y1"] - g["y0"]) >= 0.6 * H
        if tall and g["x0"] - p["x1"] <= 1 and (g["x1"] - p["x0"]) <= 0.9 * H:
            sel = p["sel"] | g["sel"]
            ys, xs = np.nonzero(sel)
            merged[-1] = {"cx": x + xs.mean(), "x0": x + xs.min(), "x1": x + xs.max() + 1,
                          "y0": y + ys.min(), "y1": y + ys.max() + 1, "sel": sel, "all": p["all"], "box": box}
        else:
            merged.append(g)
    out = merged
    tops = [g["y0"] for g in out if g["y1"] - g["y0"] >= 0.75 * H]
    top = float(np.median(tops)) if tops else float(min(g["y0"] for g in out))
    for g in out:
        gh, gw = g["y1"] - g["y0"], g["x1"] - g["x0"]
        g["kind"] = "digit"
        if gh < 0.6 * H:
            mid = (g["y0"] + g["y1"]) / 2 - top
            if mid > 0.7 * H:
                g["kind"] = "," if gh >= 1.3 * max(gw, 1) or gh > 0.25 * H else "."
            elif gw >= 1.5 * gh:
                g["kind"] = "-"
            else:
                g["kind"] = "?"
    return out


def _fit_value(segs, value, strict=False):
    """획 덩어리를 원래 값 글자에 맞춘다. 기호 덩어리 수·순서가 값과 같아야 하고,
    기호 사이 숫자 덩어리 수가 숫자 개수와 다르면(붙은 숫자, 끊긴 숫자) 그 구간을 숫자 개수로 고르게 나눈다."""
    if not segs:
        return None
    sym_idx = [i for i, g in enumerate(segs) if g["kind"] != "digit"]
    vsym = [i for i, c in enumerate(value) if not c.isdigit()]
    if len(sym_idx) != len(vsym):
        return None
    for i, j in zip(sym_idx, vsym):
        k = segs[i]["kind"]
        if k != "?" and k != value[j] and not ({k, value[j]} <= {",", "."}):
            return None
    out = []
    bounds_s = [-1] + sym_idx + [len(segs)]
    bounds_v = [-1] + vsym + [len(value)]
    for r in range(len(bounds_s) - 1):
        run = segs[bounds_s[r] + 1:bounds_s[r + 1]]
        nd = bounds_v[r + 1] - bounds_v[r] - 1
        if nd > 0 and not run:
            return None
        if len(run) == nd:
            out += [dict(g, ch="0") for g in run]
        elif nd > 0 and (strict or len(run) > nd):   # 덩어리가 더 많으면 옆 글자·번호가 섞인 것 (10/30)
            return None
        elif nd > 0:
            x0, x1 = min(g["x0"] for g in run), max(g["x1"] for g in run)
            y0, y1 = min(g["y0"] for g in run), max(g["y1"] for g in run)
            step = (x1 - x0) / nd
            for i in range(nd):
                out.append({"cx": x0 + (i + 0.5) * step, "x0": x0 + i * step, "x1": x0 + (i + 1) * step,
                            "y0": y0, "y1": y1, "sel": None, "box": run[0]["box"], "ch": "0"})
        if r < len(sym_idx):
            out.append(dict(segs[sym_idx[r]], ch=value[vsym[r]]))
    if len(out) != len(value):
        return None
    for g, c in zip(out, value):
        g["ch"] = c
    return out


def _sym_patch(gray, g, paper, ink):
    """기호 하나의 모양(0~1)과 기준점. 쉼표 꼬리처럼 옅은 끝까지 담도록 둘레를 넉넉히 뜨고,
    옆 숫자 획은 0으로 (10/23: 처음엔 꼬리가 잘려 쉼표가 점처럼 보였음)."""
    import cv2
    bx, by, bw, bh = g["box"]
    H, W = gray.shape[:2]
    y0, y1, x0, x1 = max(int(g["y0"]) - 2, 0), min(int(g["y1"]) + 3, H), max(int(g["x0"]) - 2, 0), min(int(np.ceil(g["x1"])) + 2, W)
    sub = gray[y0:y1, x0:x1].astype(np.float32)
    a = np.clip((paper - sub) / max(paper - ink, 1.0), 0, 1)
    me = np.zeros(gray.shape[:2], np.uint8)
    other = np.zeros(gray.shape[:2], np.uint8)
    if g.get("sel") is not None:
        me[by:by + bh, bx:bx + bw] = g["sel"].astype(np.uint8)
        other[by:by + bh, bx:bx + bw] = g["all"].astype(np.uint8) & (~g["sel"]).astype(np.uint8)
    keep = cv2.dilate(me, np.ones((5, 5), np.uint8)) & (1 - cv2.dilate(other, np.ones((3, 3), np.uint8)))
    keep = cv2.GaussianBlur(keep[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.6)
    return {"alpha": a * keep, "cx_in": g["cx"] - x0, "top": y0}


def _paper(gray, box, ink_mask_fn):
    x, y, w, h = box
    g = gray[y:y + h, x:x + w]
    m = ink_mask_fn(g) == 0
    return float(np.median(g[m])) if m.any() else float(np.median(g))


def plan(gray, box, gm, orig_value, new_value, ink_mask_fn):
    """(글자 자리 [(문자, 오른쪽 끝 기준 가운데 x)], 기호 모양 {문자: (alpha, 숫자 윗선 기준 위 y)}, 메모).
    원래 값을 글자별로 못 읽으면 None."""
    gs = _fit_value(_segments(gray, box, ink_mask_fn), orig_value)
    if gs is None:
        return None
    text = orig_value
    cxs = [g["cx"] for g in gs]
    if any(b <= a for a, b in zip(cxs, cxs[1:])):
        return None
    note = []
    # 새 값 글자 자리
    same_shape = len(new_value) == len(orig_value) and all(
        (a.isdigit() and b.isdigit()) or a == b for a, b in zip(orig_value, new_value))
    if same_shape:
        cx = cxs
        note.append("원래자리")
    else:
        dd = [b - a for (a, ca), (b, cb) in zip(zip(cxs, text), zip(cxs[1:], text[1:])) if ca.isdigit() and cb.isdigit()]
        if not dd:
            return None
        pd = float(np.median(dd))
        before, after = {}, {}
        for i, ch in enumerate(text):
            if ch in SYMBOLS:
                if i > 0 and text[i - 1].isdigit():
                    before.setdefault(ch, cxs[i] - cxs[i - 1])
                if i + 1 < len(text) and text[i + 1].isdigit():
                    after.setdefault(ch, cxs[i + 1] - cxs[i])

        def step(a, b):   # 글자 a 다음 b까지 가운데 간격
            if a.isdigit() and b.isdigit():
                return pd
            if b in SYMBOLS:
                return before.get(b, pd * 0.6)
            return after.get(a, pd * 0.6)
        left = "-" in orig_value      # 날짜는 왼쪽 끝 맞춤, 금액은 오른쪽 끝 맞춤
        if left:
            cx = [cxs[0]]
            for a, b in zip(new_value, new_value[1:]):
                cx.append(cx[-1] + step(a, b))
        else:
            cx = [cxs[-1]]
            for a, b in zip(new_value[::-1][1:], new_value[::-1]):
                cx.append(cx[-1] - step(a, b))
            cx = cx[::-1]
        note.append("원래간격")
    lay = [(ch, float(c - gm["right"])) for ch, c in zip(new_value, cx)]
    # 기호 모양
    syms = {}
    paper = _paper(gray, box, ink_mask_fn)
    digit_top = float(np.median([g["y0"] for g in gs if g["ch"].isdigit()]))
    for g in gs:
        if g["ch"] in SYMBOLS and g["ch"] not in syms:
            p = _sym_patch(gray, g, paper, gm["ink"])
            syms[g["ch"]] = (p["alpha"], p["top"] - digit_top, p["cx_in"])
    need = {c for c in new_value if c in SYMBOLS} - set(syms)
    if need:
        found = _neighbor_symbols(gray, box, need, gm, ink_mask_fn)
        syms.update(found)
        if found:
            note.append("이웃기호:" + "".join(found))
    if syms:
        note.append("원본기호:" + "".join(sorted(syms)))
    return lay, syms, " ".join(note)


def _neighbor_symbols(gray, box, need, gm, ink_mask_fn):
    """같은 문서 위아래 줄에서 기호 찾기. 숫자 높이가 비슷한 줄에서만."""
    x, y, w, h = box
    H, W = gray.shape[:2]
    out = {}
    for k in (1, -1, 2, -2, 3, -3):
        if not need - set(out):
            break
        yy = int(round(y + k * 1.6 * h))
        nb = (max(x - w // 2, 0), max(yy, 0), min(2 * w, W - max(x - w // 2, 0)), h)
        if nb[1] + nb[3] > H or nb[2] < 5 or nb[3] < 5:
            continue
        gs = _segments(gray, nb, ink_mask_fn)
        for g in gs:
            g["ch"] = "0" if g["kind"] == "digit" else g["kind"]
        digits = [g for g in gs if g["ch"].isdigit()]
        if len(digits) < 2:
            continue
        dh = float(np.median([g["y1"] - g["y0"] for g in digits]))
        if abs(dh - gm["height"]) > 0.2 * gm["height"]:   # 글자 크기가 다른 줄은 쓰지 않음
            continue
        top = float(np.median([g["y0"] for g in digits]))
        paper = _paper(gray, nb, ink_mask_fn)
        for g in gs:
            if g["ch"] in need and g["ch"] not in out:
                p = _sym_patch(gray, g, paper, gm["ink"])
                out[g["ch"]] = (p["alpha"], p["top"] - top, p["cx_in"])
    return out


def image_symbols(orig_bgr, orig_gray, box, orig_value, ink_mask_fn):
    """원본 이미지 그대로(돌리거나 펴지 않은)에서 원래 값 기호의 모양. 돌리거나 펼친 그림에서 뜬 기호는
    한 번 더 다시 계산되면서 쉼표 꼬리가 옅어져 점처럼 보여서(10/23), 기호는 원본 픽셀을 그대로 옮긴다.
    {문자: (원본 색/주변 종이 비율 patch(BGR), 남길 범위(0~1), 가운데 x, 가운데 y)}"""
    import cv2
    gs = _fit_value(_segments(orig_gray, box, ink_mask_fn), orig_value)
    if gs is None:
        return {}
    x, y, w, h = box
    sub = orig_bgr[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32)
    paper_m = (ink_mask_fn(orig_gray[y:y + h, x:x + w]) == 0).reshape(-1)
    paper = np.median(sub[paper_m], axis=0) if paper_m.any() else np.median(sub, axis=0)
    out = {}
    H, W = orig_gray.shape[:2]
    for g in gs:
        if g["ch"] not in SYMBOLS or g["ch"] in out or g.get("sel") is None:
            continue
        p = _sym_patch(orig_gray, g, 0, 0)   # 범위(keep)만 쓰려고. alpha는 아래에서 색 비율로 다시
        y0 = int(p["top"]); x0 = max(int(g["x0"]) - 2, 0)
        hh, ww = p["alpha"].shape
        me = np.zeros(orig_gray.shape[:2], np.uint8)
        other = np.zeros(orig_gray.shape[:2], np.uint8)
        bx, by, bw, bh = g["box"]
        me[by:by + bh, bx:bx + bw] = g["sel"].astype(np.uint8)
        other[by:by + bh, bx:bx + bw] = g["all"].astype(np.uint8) & (~g["sel"]).astype(np.uint8)
        keep = cv2.dilate(me, np.ones((5, 5), np.uint8)) & (1 - cv2.dilate(other, np.ones((3, 3), np.uint8)))
        keep = cv2.GaussianBlur(keep[y0:y0 + hh, x0:x0 + ww].astype(np.float32), (0, 0), 0.6)
        ratio = np.clip(orig_bgr[y0:y0 + hh, x0:x0 + ww].astype(np.float32) / np.maximum(paper, 1), 0, 1)
        dark = (1 - ratio.mean(axis=2)) * keep
        if dark.sum() <= 0:
            continue
        yy, xx = np.mgrid[0:hh, 0:ww]
        out[g["ch"]] = (ratio, keep, float((xx * dark).sum() / dark.sum()), float((yy * dark).sum() / dark.sum()))
    return out


def paste_symbols(img, sym_img, points):
    """points: [(문자, 이미지 좌표 가운데 x, y)]. 원본 기호 픽셀의 어두운 정도(종이 대비 비율)를 그 자리에 곱한다."""
    H, W = img.shape[:2]
    done = []
    for ch, px, py in points:
        if ch not in sym_img:
            continue
        ratio, keep, cx, cy = sym_img[ch]
        hh, ww = keep.shape
        x0, y0 = int(round(px - cx)), int(round(py - cy))
        if x0 < 0 or y0 < 0 or x0 + ww > W or y0 + hh > H:
            continue
        reg = img[y0:y0 + hh, x0:x0 + ww].astype(np.float32)
        k = keep[..., None]
        img[y0:y0 + hh, x0:x0 + ww] = np.clip(reg * (1 - k + k * ratio), 0, 255).astype(np.uint8)
        done.append(ch)
    return done
