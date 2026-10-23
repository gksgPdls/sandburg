"""4단계: 원본 숫자와 가장 닮은 폰트를 골라 새 값을 쓴다 (AI 갈래).

1) 원래 값(예: 72,900)을 후보 폰트마다 같은 자리(오른쪽 끝, 숫자 윗선, 숫자 높이)에 그려 본다
2) 자간은 원래 값의 전체 폭에 맞추고, 굵기(획 두께)와 흐림도 몇 단계로 바꿔 본다
3) 글자 획이 원본과 가장 많이 겹치는 폰트와 굵기를 고르고(±1px 위치 오차 허용), 흐림은 밝기 차이로 고른다
4) 그 조합으로 새 값을 쓴다. 진하기는 원래 글자의 가장 어두운 쪽(5%)에 맞춘다
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 4  # 크게 그린 뒤 줄여서 부드러운 가장자리를 만든다

_MAC = "/System/Library/Fonts"
_SUP = "/System/Library/Fonts/Supplemental"
_CANDIDATES = [
    (f"{_MAC}/AppleSDGothicNeo.ttc", range(0, 9)),
    (f"{_SUP}/AppleGothic.ttf", [0]), (f"{_SUP}/AppleMyungjo.ttf", [0]),
    (f"{_SUP}/Arial.ttf", [0]), (f"{_SUP}/Arial Bold.ttf", [0]),
    (f"{_SUP}/Arial Narrow.ttf", [0]), (f"{_SUP}/Arial Narrow Bold.ttf", [0]),
    (f"{_MAC}/Helvetica.ttc", [0, 1]), (f"{_SUP}/DIN Alternate Bold.ttf", [0]),
    (f"{_SUP}/Times New Roman.ttf", [0]), (f"{_SUP}/Times New Roman Bold.ttf", [0]),
    (f"{_SUP}/Courier New.ttf", [0]), (f"{_SUP}/Courier New Bold.ttf", [0]),
    (f"{_MAC}/Menlo.ttc", [0]), (f"{_MAC}/Monaco.ttf", [0]),
    # 리눅스에서 시험할 때 (맥 폰트와 숫자 모양이 비슷한 대체 폰트)
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", [0]),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf", [0]),
    ("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", [0]),
    ("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf", [0]),
    ("/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf", [0]),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", [0]),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", [0]),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", [0]),
]
CANDIDATE_LIST = _CANDIDATES   # 윈도우 스크립트에서 바꿔 끼운다
WEIGHTS = (-1, 0, 1, 2)      # 획 두께 조정 (크게 그린 그림에서 깎거나 늘리는 픽셀 수, SS배 기준)
BLURS = (0.3, 0.6, 0.9)      # 가장자리 흐림 (원래 크기 기준 px)


def candidates():
    out = []
    for path, idxs in CANDIDATE_LIST:
        if not Path(path).exists():
            continue
        for i in idxs:
            try:
                f = ImageFont.truetype(path, 40, index=i)
            except OSError:
                break
            name = " ".join(f.getname())
            if f.getbbox("0")[3] - f.getbbox("0")[1] <= 0:
                continue
            out.append((path, i, name))
    return out


_size_cache = {}


def _font(path, index, digit_h):
    """'0'의 높이가 digit_h px(원래 크기)가 되는 폰트 (SS배로 크게)."""
    key = (path, index, digit_h)
    if key not in _size_cache:
        ref = ImageFont.truetype(path, 100, index=index)
        _, t, _, b = ref.getbbox("0")
        size = max(4, int(round(100 * digit_h * SS / (b - t))))
        _size_cache[key] = ImageFont.truetype(path, size, index=index)
    return _size_cache[key]


def render_alpha(text, path, index, digit_h, tracking, weight, blur):
    """글자 모양(0~1). (alpha, 오른쪽 끝 x, '0' 윗선 y) 원래 크기 기준."""
    f = _font(path, index, digit_h)
    adv = [f.getlength(ch) for ch in text]
    track = tracking * SS
    total = sum(adv) + track * (len(text) - 1)
    pad = 6 * SS
    _, t0, _, b0 = f.getbbox("0")
    H = int(b0 - t0 + digit_h * SS * 1.2) + 2 * pad
    canvas = Image.new("L", (int(total) + 2 * pad, H), 0)
    d = ImageDraw.Draw(canvas)
    x = pad
    for ch, a in zip(text, adv):
        d.text((x, pad - t0), ch, font=f, fill=255)
        x += a + track
    arr = np.asarray(canvas, np.uint8)
    if weight > 0:
        arr = cv2.dilate(arr, np.ones((weight + 1, weight + 1), np.uint8))
    elif weight < 0:
        arr = cv2.erode(arr, np.ones((1 - weight, 1 - weight), np.uint8))
    small = cv2.resize(arr, (arr.shape[1] // SS, arr.shape[0] // SS), interpolation=cv2.INTER_AREA)
    alpha = small.astype(np.float32) / 255.0
    if blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
    cols = np.nonzero(alpha.max(axis=0) > 0.1)[0]
    right = int(cols.max()) if len(cols) else alpha.shape[1] - 1
    left = int(cols.min()) if len(cols) else 0
    return alpha, right, left, pad // SS


def composite(bg, alpha, right_x, top_y, a_right, a_top, ink):
    """bg(BGR)에 alpha를 오른쪽 끝·윗선 맞춰 얹는다. 진하기 ink(0~255 밝기). 얹은 영역 (x,y,w,h)."""
    h, w = alpha.shape
    x0, y0 = right_x - a_right, top_y - a_top
    H, W = bg.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    a = alpha[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)][..., None]
    region = bg[y0c:y1c, x0c:x1c].astype(np.float32)
    if np.ndim(ink) == 0:   # 회색 진하기 하나
        paper = float(np.median(region)) if region.size else 255.0
        k = min(1.0, ink / paper) if paper > 0 else 0.0
    else:                   # 채널별 글자색 (B, G, R)
        paper = np.median(region.reshape(-1, 3), axis=0) if region.size else np.full(3, 255.0)
        k = np.clip(np.asarray(ink, np.float32) / np.maximum(paper, 1), 0, 1)
    bg[y0c:y1c, x0c:x1c] = np.clip(region * (1 - a * (1 - k)), 0, 255).astype(np.uint8)
    return x0c, y0c, x1c - x0c, y1c - y0c


def _ink5(gray, mask):
    return float(np.percentile(gray[mask], 5)) if mask.any() else 255.0


def _shape_score(g, tmask, ink_mask_fn):
    """글자 모양 차이: 1 - IoU(그린 글자 획, 원본 글자 획). 위치 오차를 감안해 ±1px 이동 중 가장 좋은 값."""
    rmask = ink_mask_fn(g.astype(np.uint8)) > 0
    best = 1.0
    H, W = tmask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.zeros_like(rmask)
            ys, ye = max(dy, 0), H + min(dy, 0)
            xs, xe = max(dx, 0), W + min(dx, 0)
            sh[ys:ye, xs:xe] = rmask[ys - dy:ye - dy, xs - dx:xe - dx]
            inter = (sh & tmask).sum()
            union = (sh | tmask).sum()
            if union:
                best = min(best, 1 - inter / union)
    return best


def _try(erased, box, value, gm, cand, tracking, wgt, bl, tmask, ink_mask_fn):
    x, y, w, h = box
    path, idx = cand
    alpha, ar, al, at = render_alpha(value, path, idx, gm["height"], tracking, wgt, bl)
    trial = erased.copy()
    composite(trial, alpha, gm["right"], gm["top"], ar, at, gm["ink"])
    g = cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY)
    got = _ink5(g[y:y + h, x:x + w], tmask | (ink_mask_fn(g[y:y + h, x:x + w]) > 0))
    ink = float(np.clip(gm["ink"] - (got - gm["ink"]), 0, 255))   # 진하기 한 번 보정
    trial = erased.copy()
    composite(trial, alpha, gm["right"], gm["top"], ar, at, ink)
    return cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].astype(np.float32), ink


def _glyph_targets(orig_gray, box, value, ink_mask_fn, glyphs=None):
    """원본 칸의 글자별 획 마스크 [(문자, 마스크)]. OCR 글자 수가 값과 안 맞으면 None.
    glyphs(글자별 [x, y, w, h])를 주면 OCR을 하지 않는다 (윈도우에서 실행할 때)."""
    if glyphs is None:
        import receipt
        text, glyphs = receipt.char_boxes(orig_gray, box, whitelist="0123456789,.-")
    if len(glyphs) != len(value):
        return None
    x, y, w, h = box
    full = ink_mask_fn(orig_gray[y:y + h, x:x + w]) > 0
    out = []
    for ch, (gx, gy, gw, gh) in zip(value, glyphs):
        if not ch.isdigit():
            continue
        sub = full[:, max(gx - x, 0):gx - x + gw]
        ys, xs = np.nonzero(sub)
        if len(xs) < 4:
            continue
        out.append((ch, sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out or None


def _glyph_score(targets, path, idx, height, wgt, ink_threshold=0.45):
    """숫자 글자를 하나씩 그려 원본 글자와 모양 비교 (가운데 맞춤, ±1px). 1 - 평균 IoU. 자간 영향 없음.
    ink_threshold: 원본 획 마스크(종이보다 35 어두운 픽셀)와 같은 기준이 되도록 35 / (종이 - 잉크)."""
    losses = []
    for ch, t in targets:
        alpha, _, _, _ = render_alpha(ch, path, idx, height, 0, wgt, 0.6)
        m = alpha > ink_threshold
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return 1.0
        r = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        H, W = max(r.shape[0], t.shape[0]) + 2, max(r.shape[1], t.shape[1]) + 2
        def place(a, dy=0, dx=0):
            c = np.zeros((H, W), bool)
            oy, ox = (H - a.shape[0]) // 2 + dy, (W - a.shape[1]) // 2 + dx
            c[max(oy, 0):oy + a.shape[0], max(ox, 0):ox + a.shape[1]] = a[max(-oy, 0):, max(-ox, 0):][:H - max(oy, 0), :W - max(ox, 0)]
            return c
        tt = place(t)
        best = 1.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rr = place(r, dy, dx)
                u = (rr | tt).sum()
                if u:
                    best = min(best, 1 - (rr & tt).sum() / u)
        losses.append(best)
    return float(np.mean(losses))


def choose(erased, orig_gray, box, value, gm, ink_mask_fn, glyphs=None):
    """원래 값을 후보마다 그려 원본 칸과 비교한다.
    1차: 폰트와 굵기를 글자 모양(획이 겹치는 정도)으로 고름
    2차: 고른 폰트에서 흐림을 밝기 차이로 고름"""
    x, y, w, h = box
    target = orig_gray[y:y + h, x:x + w].astype(np.float32)
    tmask = ink_mask_fn(orig_gray[y:y + h, x:x + w]) > 0
    target_w = gm["right"] - gm["left"] + 1
    targets = _glyph_targets(orig_gray, box, value, ink_mask_fn, glyphs)
    paper = float(np.median(target[~tmask])) if (~tmask).any() else 255.0
    thr = float(np.clip(35.0 / max(paper - gm["ink"], 1.0), 0.05, 0.95))
    results = []
    for path, idx, name in candidates():
        a0, r0, l0, _ = render_alpha(value, path, idx, gm["height"], 0, 0, 0.6)
        tracking = (target_w - (r0 - l0 + 1)) / max(len(value) - 1, 1)
        if abs(tracking) > 0.35 * gm["height"]:   # 자간을 너무 바꿔야 하면 모양이 다른 폰트
            continue
        for wgt in WEIGHTS:
            g, ink = _try(erased, box, value, gm, (path, idx), tracking, wgt, 0.6, tmask, ink_mask_fn)
            # 글자별 모양 비교가 되면 그걸 쓰고(자간 영향 없음), 안 되면 값 전체 획 겹침
            score = (_glyph_score(targets, path, idx, gm["height"], wgt, thr) if targets
                     else _shape_score(g, tmask, ink_mask_fn))
            results.append({"score": score, "path": path, "index": idx, "by": "글자별" if targets else "전체",
                            "name": name, "tracking": tracking, "weight": wgt, "blur": 0.6, "ink": ink})
    results.sort(key=lambda r: r["score"])
    best = dict(results[0])
    # 2차: 고른 폰트로 굵기, 흐림, 진하기를 함께 맞춘다. 칸 안 밝기 분포(어두운 쪽 분위수)를 원본과 비교한다.
    # 픽셀끼리 비교하면 획 위치가 조금만 어긋나도 옅은 글자가 유리해져서(시험 결과 너무 옅게 나옴),
    # 위치와 상관없는 분포 비교를 쓴다. 어두운 픽셀의 양은 굵기, 밝기는 진하기와 흐림을 반영한다
    qs = (2, 5, 10, 15, 20, 30, 40)
    tq = np.percentile(target, qs)
    best_err = None
    for wgt in WEIGHTS:
        for bl in BLURS:
            alpha, ar, al, at = render_alpha(value, best["path"], best["index"], gm["height"],
                                             best["tracking"], wgt, bl)
            for d_ink in (-40, -20, 0, 20, 40, 60):
                ink = float(np.clip(gm["ink"] + d_ink, 0, 255))
                trial = erased.copy()
                composite(trial, alpha, gm["right"], gm["top"], ar, at, ink)
                g = cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].astype(np.float32)
                err = float(np.abs(np.percentile(g, qs) - tq).mean())
                if best_err is None or err < best_err:
                    best_err = err
                    best.update({"weight": wgt, "blur": bl, "ink": ink, "밝기차": err})
    # 상위 목록은 폰트별 최고 1개씩
    seen, top = set(), []
    for r in results:
        if r["name"] not in seen:
            seen.add(r["name"]); top.append(r)
    return best, top[:5]


def ink_color(orig_bgr, orig_gray, box, ink_mask_fn):
    """원본 글자 획 중 어두운 쪽 30% 픽셀의 평균 색 (B, G, R)과 그 밝기."""
    x, y, w, h = box
    m = ink_mask_fn(orig_gray[y:y + h, x:x + w]) > 0
    if m.sum() < 5:
        return None, None
    g = orig_gray[y:y + h, x:x + w][m].astype(np.float32)
    dark = g <= np.percentile(g, 30)
    col = orig_bgr[y:y + h, x:x + w][m][dark].astype(np.float32).mean(axis=0)
    return col, float(g[dark].mean())


def write_matched(img, erased_for_scoring, orig_gray, box, original_value, new_value, gm, ink_mask_fn,
                  orig_bgr=None, glyphs=None):
    """원래 값으로 폰트를 고르고, 그 조합으로 new_value를 img에 쓴다. (쓴 영역, 고른 조합, 상위 5개).
    orig_bgr를 주면 글자색(파란 볼펜 같은 색)도 원본 획 색에 맞춘다. 진하기는 고른 값을 유지."""
    best, top = choose(erased_for_scoring, orig_gray, box, original_value, gm, ink_mask_fn, glyphs)
    alpha, ar, al, at = render_alpha(new_value, best["path"], best["index"], gm["height"],
                                     best["tracking"], best["weight"], best["blur"])
    ink = best["ink"]
    if orig_bgr is not None:
        col, col_gray = ink_color(orig_bgr, orig_gray, box, ink_mask_fn)
        if col is not None and col_gray > 0:
            ink = np.clip(col * (best["ink"] / col_gray), 0, 255)
            best["글자색"] = "#%02x%02x%02x" % tuple(int(v) for v in col[::-1])
    written = composite(img, alpha, gm["right"], gm["top"], ar, at, ink)
    ys, xs = np.nonzero(alpha > 0.1)
    x0 = gm["right"] - ar
    y0 = gm["top"] - at
    tight = (x0 + int(xs.min()), y0 + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    return tight, best, top
