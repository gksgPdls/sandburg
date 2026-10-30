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
    (f"{_SUP}/Tahoma.ttf", [0]), (f"{_SUP}/Verdana.ttf", [0]),
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
# 맥에서 따로 설치한 한글 글꼴 찾기 (서체 관리자에서 내려받은 글꼴, 사용자 설치 글꼴, 정식 MS Office에 들어 있는 글꼴)
_EXTRA_DIRS = [Path(__file__).resolve().parents[3] / "fonts",            # 저장소 fonts/ (git 제외)
               Path.home() / "Library" / "Fonts", Path("/Library/Fonts"),
               Path("/System/Library/AssetsV2"),                      # 서체 관리자에서 받은 글꼴
               Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts"),
               Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"),
               Path("/Applications/Microsoft Excel.app/Contents/Resources/DFonts")]
_EXTRA_NAMES = ("gulim", "dotum", "batang", "malgun", "nanum", "notosanskr", "notoserifkr", "d2coding")


def _extra_fonts():
    found, seen = [], set()
    for d in _EXTRA_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            n = f.name.lower().replace(" ", "").replace("-", "")
            if f.suffix.lower() in (".ttf", ".ttc", ".otf") and any(k in n for k in _EXTRA_NAMES) and f.name not in seen:
                seen.add(f.name)
                found.append((str(f), range(0, 4) if f.suffix.lower() == ".ttc" else [0]))
    return found


# 글꼴 범위: 기본은 윈도우 글꼴만 (10/23 결정). FORGE_FONTS=all 이면 맥 글꼴도 후보에 넣음
_WIN_LIKE = [(p, i) for p, i in _CANDIDATES
             if Path(p).name.startswith(("Arial", "Times New Roman", "Courier New", "Tahoma", "Verdana"))]
import os as _os
if _os.environ.get("FORGE_FONTS", "windows") == "all":
    CANDIDATE_LIST = _CANDIDATES + _extra_fonts()
else:
    CANDIDATE_LIST = _extra_fonts() + _WIN_LIKE   # 윈도우 스크립트에서 바꿔 끼운다
WEIGHTS = (-1, 0, 1, 2)      # 획 두께 조정 (크게 그린 그림에서 깎거나 늘리는 픽셀 수, SS배 기준)
BLURS = (0.3, 0.6, 0.9)      # 가장자리 흐림 (원래 크기 기준 px)


_HANDWRITING = ("brush", "pen", "script", "hand", "붓", "펜", "손글씨")


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
            if "Gungsuh" in name:   # 궁서는 붓글씨체라 영수증 숫자와 다르고, 작은 크기에서 쉼표가 점처럼 보임 (10/23, auto00962)
                continue
            if any(k in name.lower() for k in _HANDWRITING):   # 손글씨체는 인쇄 문서에 안 맞음 (10/23, auto00466 나눔붓)
                continue
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


def _render_layout(f, lay, syms, weight, pad, t0, H, blur=0.0):
    """C1: 글자를 정해진 가운데 자리(lay, 오른쪽 끝 기준 px)에 하나씩 그리고, 기호는 원본에서 뜬 모양(syms)을 붙인다.
    (캔버스, 기준점(오른쪽 끝) 열). 기준점 열은 SS의 배수가 되게 맞춘다 (줄일 때 반 픽셀 어긋나지 않게)."""
    half = 2 * f.size
    cs = [c for _, c in lay]
    cmin, cmax = min(cs + [0.0]), max(cs + [0.0])
    X = lambda c: pad + half + (c - cmin) * SS
    shift = (-X(0.0)) % SS
    Xs = lambda c: X(c) + shift
    Wc = int(Xs(cmax) + half + pad) + SS
    canvas = Image.new("L", (Wc, H), 0)
    d = ImageDraw.Draw(canvas)
    # 정해진 자리 간격보다 글자가 넓으면 글자끼리 닿으므로 가로만 줄인다 (최소 1px 틈)
    dig = [(ch, c) for ch, c in lay if ch.isdigit()]
    sx = 1.0
    if len(dig) >= 2:
        dmin = min(b - a for (_, a), (_, b) in zip(dig, dig[1:]) if b > a) if any(b > a for (_, a), (_, b) in zip(dig, dig[1:])) else None
        wmax = max(_ink_x(f, ch)[1] - _ink_x(f, ch)[0] for ch, _ in dig)
        # 틈은 0.3px까지만 두고, 가로로 줄이는 건 88%까지만 (10/23: 1px 틈, 60%까지 줄였더니 글자가 홀쭉하다는 의견, auto00094)
        if dmin and wmax > dmin * SS - 0.3 * SS:
            sx = max((dmin * SS - 0.3 * SS) / wmax, 0.88)
    if sx < 1.0 and weight < 0:   # 가로로 줄인 글자를 또 깎으면 너무 가늘고 연해짐
        weight = 0
    pieces = []
    for ch, c in lay:
        if syms and ch in syms:
            continue
        l, r = _ink_x(f, ch)
        if sx == 1.0:
            d.text((Xs(c) - (l + r) / 2, pad - t0), ch, font=f, fill=255)
        else:
            tmp = Image.new("L", (int(r - l) + 2 * f.size, H), 0)
            ImageDraw.Draw(tmp).text((f.size - l, pad - t0), ch, font=f, fill=255)
            t = cv2.resize(np.asarray(tmp, np.uint8), (max(int(round(tmp.width * sx)), 1), H), interpolation=cv2.INTER_AREA)
            pieces.append((t, int(round(Xs(c) - (f.size + (r - l) / 2) * sx))))
    arr = np.asarray(canvas, np.uint8).copy()
    for t, x0 in pieces:
        a0, a1 = max(x0, 0), min(x0 + t.shape[1], arr.shape[1])
        if a1 > a0:
            arr[:, a0:a1] = np.maximum(arr[:, a0:a1], t[:, a0 - x0:a1 - x0])
    if weight > 0:
        arr = cv2.dilate(arr, np.ones((weight + 1, weight + 1), np.uint8))
    elif weight < 0:
        arr = cv2.erode(arr, np.ones((1 - weight, 1 - weight), np.uint8))
    for ch, c in lay:
        if not syms or syms.get(ch) is None:
            continue
        a, top, cx_in = syms[ch]
        if blur > 0:   # 뒤에서 글자 전체에 흐림을 한 번 더 주므로, 원본에서 뜬 기호는 그만큼 미리 또렷하게 (쉼표 꼬리가 사라져 점처럼 보였음)
            a = np.clip(a + 1.0 * (a - cv2.GaussianBlur(a, (0, 0), blur)), 0, 1)
        big = cv2.resize(a, (a.shape[1] * SS, a.shape[0] * SS), interpolation=cv2.INTER_NEAREST)
        # 원본 픽셀 격자에 딱 맞게 놓는다 (반 픽셀 어긋나면 줄일 때 두 칸에 나뉘어 쉼표 꼬리가 옅어짐)
        x0 = int(round((Xs(c) - cx_in * SS) / SS)) * SS
        y0 = int(round(pad + top * SS))
        hh, ww = big.shape
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(x0, 0), max(y0, 0)
        dx1, dy1 = min(x0 + ww, arr.shape[1]), min(y0 + hh, arr.shape[0])
        if dx1 > dx0 and dy1 > dy0:
            piece = np.clip(big[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * 255, 0, 255).astype(np.uint8)
            arr[dy0:dy1, dx0:dx1] = np.maximum(arr[dy0:dy1, dx0:dx1], piece)
    return arr, int(Xs(0.0))


def render_alpha(text, path, index, digit_h, tracking, weight, blur, rotate=0.0, big=False, lay=None, syms=None,
                 sym_blur=None):
    """글자 모양(0~1). (alpha, 오른쪽 끝 x, 왼쪽 끝 x, '0' 윗선 y) 원래 크기 기준.
    rotate(도, getRotationMatrix2D 방향)를 주면 크게 그린 상태에서 '오른쪽 끝, 숫자 윗선' 점을 중심으로 돌린 뒤
    줄인다(작은 그림을 돌릴 때 생기는 계단 현상 방지). 이때 오른쪽 끝·윗선 값은 회전 중심점 위치."""
    f = _font(path, index, digit_h)
    adv = [f.getlength(ch) for ch in text]
    track = tracking * SS
    total = sum(adv) + track * (len(text) - 1)
    pad = 6 * SS + (int(abs(np.sin(np.radians(rotate))) * total) + SS if rotate else 0)
    _, t0, _, b0 = f.getbbox("0")
    H = int(b0 - t0 + digit_h * SS * 1.2) + 2 * pad
    ref = None
    if lay:
        arr, ref = _render_layout(f, lay, syms, weight, pad, t0, H, blur if sym_blur is None else sym_blur)
    else:
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
    if rotate:
        c = np.nonzero(arr.max(axis=0) > 64)[0]
        ax = float(ref) if ref is not None else (float(c.max()) if len(c) else float(arr.shape[1] - 1))
        M = cv2.getRotationMatrix2D((ax, float(pad)), rotate, 1.0)
        arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if big:   # 크게 그린 상태 그대로 (원근 변형을 크게 한 뒤 줄이려고). 오른쪽 끝, '0' 윗선도 큰 좌표
        c = np.nonzero(arr.max(axis=0) > 25)[0]
        right_big = ref if ref is not None else (int(c.max()) if len(c) else arr.shape[1] - 1)
        return arr.astype(np.float32) / 255.0, right_big, (int(c.min()) if len(c) else 0), pad
    small = cv2.resize(arr, (arr.shape[1] // SS, arr.shape[0] // SS), interpolation=cv2.INTER_AREA)
    alpha = small.astype(np.float32) / 255.0
    if blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
    cols = np.nonzero(alpha.max(axis=0) > 0.1)[0]
    right = int(cols.max()) if len(cols) else alpha.shape[1] - 1
    left = int(cols.min()) if len(cols) else 0
    if rotate:
        right = int(round(ax / SS))
    elif ref is not None:
        right = ref // SS
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


FONT_HINT = None    # (path, index): 같은 이미지에서 앞서 고른 글꼴
HINT_MARGIN = 0.08
LAST_CHOICE = None
_ALL = ("*",)       # choose(only=_ALL): 모든 글꼴, 자간 기준 없이
SPACING_W = 1.0   # 글자 간격 차이 가중치


def _spacing_err(glyphs, value, path, idx, height, tracking):
    """글자 사이 간격이 원본과 얼마나 다른지 (글자 중심 위치 차이 평균 / 숫자 높이).
    글자 모양만 보면 고정폭(굴림체·돋움체)과 가변폭(굴림·돋움)을 못 가려서, 쉼표 자리나 자간이 원본과 달라짐 (10/23 auto00466)."""
    if not glyphs or len(glyphs) != len(value) or len(value) < 3:
        return 0.0
    oc = np.array([gx + gw / 2 for gx, gy, gw, gh in glyphs], np.float64)
    f = _font(path, idx, height)
    x, rc = 0.0, []
    for ch in value:
        l, _, r, _ = f.getbbox(ch)
        rc.append(x + (l + r) / 2)
        x += f.getlength(ch) + tracking * SS
    rc = np.array(rc) / SS
    oc, rc = oc - oc.mean(), rc - rc.mean()
    return float(np.abs(oc - rc).mean() / max(height, 1))


def choose(erased, orig_gray, box, value, gm, ink_mask_fn, glyphs=None, only=None):
    """원래 값을 후보마다 그려 원본 칸과 비교한다.
    1차: 폰트와 굵기를 글자 모양(획이 겹치는 정도)으로 고름
    2차: 고른 폰트에서 흐림을 밝기 차이로 고름"""
    x, y, w, h = box
    target = orig_gray[y:y + h, x:x + w].astype(np.float32)
    tmask = ink_mask_fn(orig_gray[y:y + h, x:x + w]) > 0
    target_w = gm["right"] - gm["left"] + 1
    if glyphs is None:
        try:
            import receipt
            _, glyphs = receipt.char_boxes(orig_gray, box, whitelist="0123456789,.-")
        except Exception:
            glyphs = []
    targets = _glyph_targets(orig_gray, box, value, ink_mask_fn, glyphs)
    paper = float(np.median(target[~tmask])) if (~tmask).any() else 255.0
    thr = float(np.clip(35.0 / max(paper - gm["ink"], 1.0), 0.05, 0.95))
    results = []
    for path, idx, name in candidates():
        if only is not None and only is not _ALL and (path, idx) != only:
            continue
        a0, r0, l0, _ = render_alpha(value, path, idx, gm["height"], 0, 0, 0.6)
        tracking = (target_w - (r0 - l0 + 1)) / max(len(value) - 1, 1)
        if abs(tracking) > 0.35 * gm["height"] and only is None:   # 자간을 너무 바꿔야 하면 모양이 다른 폰트
            continue
        for wgt in WEIGHTS:
            g, ink = _try(erased, box, value, gm, (path, idx), tracking, wgt, 0.6, tmask, ink_mask_fn)
            # 글자별 모양 비교가 되면 그걸 쓰고(자간 영향 없음), 안 되면 값 전체 획 겹침
            score = (_glyph_score(targets, path, idx, gm["height"], wgt, thr) if targets
                     else _shape_score(g, tmask, ink_mask_fn))
            sp = _spacing_err(glyphs, value, path, idx, gm["height"], tracking)
            score += SPACING_W * sp
            results.append({"score": score, "간격": round(sp, 3), "path": path, "index": idx, "by": "글자별" if targets else "전체",
                            "name": name, "tracking": tracking, "weight": wgt, "blur": 0.6, "ink": ink})
    if not results and only is None:   # 자간 기준에 다 걸리면(칸 크기를 잘못 잰 경우 등) 기준 없이 다시
        return choose(erased, orig_gray, box, value, gm, ink_mask_fn, glyphs, only=_ALL)
    if not results:
        raise ValueError("글꼴 후보를 하나도 못 그림")
    results.sort(key=lambda r: r["score"])
    best = dict(results[0])
    # 같은 문서에서 먼저 고친 칸의 글꼴이 거의 같은 점수면 그 글꼴로 (합계 맞추기에서 두 칸 글꼴이 달라 보이지 않게)
    if FONT_HINT is not None and only is None or only is _ALL and FONT_HINT is not None:
        same = [r for r in results if (r["path"], r["index"]) == FONT_HINT]
        if same and same[0]["score"] <= best["score"] + HINT_MARGIN:
            best = dict(same[0])
            best["글꼴맞춤"] = "앞칸"
    global LAST_CHOICE
    LAST_CHOICE = (best["path"], best["index"])
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
    lay, syms = _plan(orig_gray, box, gm, original_value, new_value, ink_mask_fn, best)
    _SYM_FRAME.clear(); _SYM_FRAME.update({k: v for k, v in (syms or {}).items() if v is not None})
    syms_r, sym_img = _split_syms(orig_bgr, orig_gray, box, original_value, ink_mask_fn, lay, syms, best)
    alpha, ar, al, at = render_alpha(new_value, best["path"], best["index"], gm["height"],
                                     best["tracking"], best["weight"], best["blur"], lay=lay, syms=syms_r)
    ink = best["ink"]
    if orig_bgr is not None:
        col, col_gray = ink_color(orig_bgr, orig_gray, box, ink_mask_fn)
        if col is not None and col_gray > 0:
            ink = np.clip(col * (best["ink"] / col_gray), 0, 255)
            best["글자색"] = "#%02x%02x%02x" % tuple(int(v) for v in col[::-1])
    written = composite(img, alpha, gm["right"], gm["top"], ar, at, ink)
    if sym_img:
        import layout
        layout.paste_symbols(img, sym_img, _sym_points(lay, syms, sym_img, gm, lambda fx, fy: (fx, fy)))
    ys, xs = np.nonzero(alpha > 0.1)
    x0 = gm["right"] - ar
    y0 = gm["top"] - at
    tight = (x0 + int(xs.min()), y0 + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    return tight, best, top


def write_matched_rotated(img, scoring_bg, orig_gray, orig_bgr, box, original_value, new_value, theta,
                          ink_mask_fn, metrics_fn):
    """B2: 기울어진 문서. 칸 주변을 theta도 만큼 반듯하게 돌려 놓고(측정과 글꼴 고르기는 반듯한 상태에서),
    새 값을 그린 뒤 글자 모양(알파)만 다시 원래 기울기로 돌려 원본 위에 얹는다. 원본 이미지 자체는 돌리지 않는다.
    theta: 글자 밑선 각도(도, 아래 방향이 양수). (쓴 영역, 고른 조합, 상위 5개)"""
    x, y, w, h = box
    H, W = orig_gray.shape
    mx, my = max(w, 3 * h), 2 * h
    X0, Y0, X1, Y1 = max(x - mx, 0), max(y - my, 0), min(x + w + mx, W), min(y + h + my, H)
    cw, ch = X1 - X0, Y1 - Y0
    bx = (x - X0, y - Y0, w, h)
    R = cv2.getRotationMatrix2D((bx[0] + w / 2, bx[1] + h / 2), theta, 1.0)
    Rinv = cv2.invertAffineTransform(R)
    rot = lambda a, interp=cv2.INTER_CUBIC: cv2.warpAffine(a, R, (cw, ch), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    og = rot(orig_gray[Y0:Y1, X0:X1])
    ob = rot(orig_bgr[Y0:Y1, X0:X1])
    bg = rot(scoring_bg[Y0:Y1, X0:X1])
    gm = metrics_fn(og, bx)
    best, top = choose(bg, og, bx, original_value, gm, ink_mask_fn)
    lay, syms = _plan(og, bx, gm, original_value, new_value, ink_mask_fn, best)
    _SYM_FRAME.clear(); _SYM_FRAME.update({k: v for k, v in (syms or {}).items() if v is not None})
    syms_r, sym_img = _split_syms(orig_bgr, orig_gray, box, original_value, ink_mask_fn, lay, syms, best)
    # 반듯한 상태에서 잰 '오른쪽 끝, 숫자 윗선'을 원래(기울어진) 위치로 옮기고, 글자는 크게 그린 상태에서 기울여 얹는다
    anchor = Rinv @ np.array([gm["right"], gm["top"], 1.0])
    alpha, ar, al, at = render_alpha(new_value, best["path"], best["index"], gm["height"],
                                     best["tracking"], best["weight"], best["blur"], rotate=-theta, lay=lay, syms=syms_r)
    ink = best["ink"]
    col, col_gray = ink_color(ob, og, bx, ink_mask_fn)
    if col is not None and col_gray > 0:
        ink = np.clip(col * (best["ink"] / col_gray), 0, 255)
        best["글자색"] = "#%02x%02x%02x" % tuple(int(v) for v in col[::-1])
    sub = img[Y0:Y1, X0:X1]
    composite(sub, alpha, int(round(anchor[0])), int(round(anchor[1])), ar, at, ink)
    img[Y0:Y1, X0:X1] = sub
    if sym_img:
        import layout
        def to_image(fx, fy):
            p = Rinv @ np.array([fx, fy, 1.0])
            return p[0] + X0, p[1] + Y0
        layout.paste_symbols(img, sym_img, _sym_points(lay, syms, sym_img, gm, to_image))
    A = np.zeros((ch, cw), np.float32)
    ax0, ay0 = int(round(anchor[0])) - ar, int(round(anchor[1])) - at
    ah, aw = alpha.shape
    sx0, sy0 = max(0, -ax0), max(0, -ay0)
    dx0, dy0 = max(ax0, 0), max(ay0, 0)
    dx1, dy1 = min(ax0 + aw, cw), min(ay0 + ah, ch)
    A[dy0:dy1, dx0:dx1] = alpha[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    ys, xs = np.nonzero(A > 0.1)
    best["기울기"] = round(float(theta), 2)
    return (X0 + int(xs.min()), Y0 + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1), best, top


def _ink_x(f, ch):
    """글자 잉크의 왼쪽·오른쪽 끝 (글자 시작점 기준, SS배 px). getbbox는 잉크가 아닌 글자 칸을 줘서 직접 그려 잰다."""
    w = int(f.getlength(ch)) + 4 * f.size
    im = Image.new("L", (w, 3 * f.size), 0)
    ImageDraw.Draw(im).text((f.size, f.size), ch, font=f, fill=255)
    c = np.nonzero(np.asarray(im).max(axis=0) > 64)[0]
    return (c.min() - f.size, c.max() + 1 - f.size) if len(c) else (0, 0)


def _min_gap(text, path, index, height):
    """자간 0일 때 이웃한 두 글자 잉크 사이 가장 좁은 틈 (원래 크기 px)."""
    f = _font(path, index, height)
    gaps = []
    for a, b in zip(text, text[1:]):
        _, ra = _ink_x(f, a)
        lb, _ = _ink_x(f, b)
        gaps.append(f.getlength(a) - ra + lb)
    return min(gaps) / SS if gaps else 0.0


USE_LAYOUT = True   # C1: 원본 글자 자리·원본 기호 쓰기 (끄면 예전처럼 값 전체 폭만 맞춤)


def _plan(gray, box, gm, original_value, new_value, ink_mask_fn, best):
    if not USE_LAYOUT:
        return None, None
    try:
        import layout
        r = layout.plan(gray, box, gm, original_value, new_value, ink_mask_fn)
    except Exception as e:   # 배치를 못 하면 예전 방식
        best["배치"] = f"실패({type(e).__name__})"
        return None, None
    if r is None:
        best["배치"] = "못읽음"
        return None, None
    lay, syms, note = r
    best["배치"] = note
    return lay, syms


def _split_syms(orig_bgr, orig_gray, box, original_value, ink_mask_fn, lay, syms, best):
    """기호를 원본 이미지 픽셀로 옮길 수 있으면 그리기에서 빼고(None), 옮길 목록을 돌려준다."""
    if not lay or not syms or orig_bgr is None:
        return syms, {}
    import layout
    try:
        sym_img = layout.image_symbols(orig_bgr, orig_gray, box, original_value, ink_mask_fn)
    except Exception:
        sym_img = {}
    sym_img = {k: v for k, v in sym_img.items() if syms.get(k) is not None}
    if sym_img:
        best["배치"] = best.get("배치", "") + " 기호원본픽셀"
    return {k: (None if k in sym_img else v) for k, v in syms.items()}, sym_img


def _sym_points(lay, syms, sym_img, gm, to_image):
    """그림 좌표의 기호 가운데를 원본 이미지 좌표로."""
    pts = []
    for ch, c in lay:
        if ch not in sym_img:
            continue
        a, top, cx_in = syms_frame = _SYM_FRAME[ch]
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        cy = float((yy * a).sum() / max(a.sum(), 1e-6))
        fx, fy = gm["right"] + c, gm["top"] + top + cy
        ix, iy = to_image(fx, fy)
        pts.append((ch, ix, iy))
    return pts


_SYM_FRAME = {}


def choose_rotated(scoring_bg, orig_gray, box, value, theta, ink_mask_fn, metrics_fn):
    """기울기만 펴고(회전) 글꼴을 고른다. 원근 펴기는 사다리꼴이 조금만 어긋나도 글자 모양이 찌그러져
    글꼴 고르기가 틀어지므로(10/23 auto00466 나눔붓, 00141), 글꼴은 회전으로 편 상태에서도 골라 비교한다."""
    x, y, w, h = box
    H, W = orig_gray.shape
    mx, my = max(w, 3 * h), 2 * h
    X0, Y0, X1, Y1 = max(x - mx, 0), max(y - my, 0), min(x + w + mx, W), min(y + h + my, H)
    bx = (x - X0, y - Y0, w, h)
    R = cv2.getRotationMatrix2D((bx[0] + w / 2, bx[1] + h / 2), theta, 1.0)
    rot = lambda a: cv2.warpAffine(a, R, (X1 - X0, Y1 - Y0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    og, bg = rot(orig_gray[Y0:Y1, X0:X1]), rot(scoring_bg[Y0:Y1, X0:X1])
    return choose(bg, og, bx, value, metrics_fn(og, bx), ink_mask_fn)


def write_matched_perspective(img, scoring_bg, orig_gray, orig_bgr, box, original_value, new_value, quad,
                              ink_mask_fn, metrics_fn, margin=20, theta=0.0):
    """B2-2: 비스듬히 찍은 문서(원근). 칸의 사다리꼴(quad: 좌상, 우상, 우하, 좌하)을 반듯한 직사각형으로 펴고
    (warpPerspective) 펴진 상태에서 측정·글꼴 고르기·새 값 그리기를 한 뒤, 그린 글자 모양(알파)만 원래 사다리꼴로 되돌려
    원본 위에 얹는다. 원본 이미지 자체는 변형하지 않는다. (쓴 영역, 고른 조합, 상위 5개)"""
    H, W = orig_gray.shape
    q = np.asarray(quad, np.float32)
    Wr = int(round((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2))
    Hr = int(round((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2))
    X0, Y0 = max(int(q[:, 0].min()) - margin, 0), max(int(q[:, 1].min()) - margin, 0)
    X1, Y1 = min(int(q[:, 0].max()) + margin, W), min(int(q[:, 1].max()) + margin, H)
    qc = q - np.float32([X0, Y0])
    cw, ch = X1 - X0, Y1 - Y0
    dst = np.float32([[margin, margin], [margin + Wr, margin], [margin + Wr, margin + Hr], [margin, margin + Hr]])
    Hm = cv2.getPerspectiveTransform(qc, dst)
    Hinv = np.linalg.inv(Hm)
    RW, RH = Wr + 2 * margin, Hr + 2 * margin
    flat = lambda a: cv2.warpPerspective(a, Hm, (RW, RH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    og = flat(orig_gray[Y0:Y1, X0:X1])
    ob = flat(orig_bgr[Y0:Y1, X0:X1])
    bg = flat(scoring_bg[Y0:Y1, X0:X1])
    x, y, w, h = box
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) - np.float32([X0, Y0])
    pc = cv2.perspectiveTransform(corners[None], Hm)[0]
    bx0, by0 = max(int(np.floor(pc[:, 0].min())), 0), max(int(np.floor(pc[:, 1].min())), 0)
    bx1, by1 = min(int(np.ceil(pc[:, 0].max())), RW), min(int(np.ceil(pc[:, 1].max())), RH)
    bx = (bx0, by0, bx1 - bx0, by1 - by0)
    gm = metrics_fn(og, bx)
    # 글꼴·굵기·흐림은 기울기만 편 상태에서 고른다 (10/23 사용자 확인: 원근으로 편 원본은 다시 계산되며 흐려지고
    # 모양도 찌그러져 글꼴이 틀어지고 흐림을 크게 고름). 원근은 글자를 그린 뒤 얹을 때만 쓴다.
    rbest, top = choose_rotated(scoring_bg, orig_gray, box, original_value, theta, ink_mask_fn, metrics_fn)
    best = dict(rbest)
    # 자간만 펴진 상태의 칸 폭에 맞춘다 (펴진 칸은 가로세로 비율이 원본과 조금 달라서, 회전 쪽 자간을 쓰면
    # 값 전체 폭이 원본보다 20% 정도 넓어짐. 10/23 시험: auto00466 예상 71px, 회전 자간 86px, 다시 맞춤 70px)
    f0 = _font(best["path"], best["index"], gm["height"])
    w0 = sum(f0.getlength(c) for c in original_value) / SS
    tw = gm["right"] - gm["left"] + 1
    n1 = max(len(original_value) - 1, 1)
    best["tracking"] = (tw - w0) / n1
    # 자간은 글자끼리 닿기 직전까지(최소 0.3px 틈)만 줄이고, 그래도 넓으면 글자 폭을 줄인다
    # (10/23: 처음엔 글자가 붙었고(auto00466 '000'), 높이의 8%로 막았더니 글자가 홀쭉해지고 자간이 넓어 보였음)
    sx = 1.0
    lay, syms = _plan(og, bx, gm, original_value, new_value, ink_mask_fn, best)
    _SYM_FRAME.clear(); _SYM_FRAME.update({k: v for k, v in (syms or {}).items() if v is not None})
    syms_r, sym_img = _split_syms(orig_bgr, orig_gray, box, original_value, ink_mask_fn, lay, syms, best)
    lim = 0.3 - _min_gap(new_value, best["path"], best["index"], gm["height"])
    if lay is None and best["tracking"] < lim:
        sx = (tw - lim * n1) / max(w0, 1)
        best["tracking"] = lim / sx
        best["가로배율"] = round(sx, 3)
    best["글꼴기준"] = "회전"
    # 뿌옇게 되지 않게: 크게(SS배) 그린 글자를 크게 펼친 캔버스에 놓고, 원근 변형도 큰 상태에서 한 뒤 줄이고,
    # 흐림은 마지막에 한 번만 준다 (10/23: 작은 글자를 변형하면 다시 계산하면서 한 번 더 흐려졌음)
    big, br, bl_, bt = render_alpha(new_value, best["path"], best["index"], gm["height"],
                                    best["tracking"], best["weight"], 0, big=True, lay=lay, syms=syms_r,
                                    sym_blur=best["blur"])
    if sx != 1.0:   # 오른쪽 끝을 기준으로 가로만 줄임
        big = cv2.resize(big, (max(int(round(big.shape[1] * sx)), 1), big.shape[0]), interpolation=cv2.INTER_AREA)
        br = int(round(br * sx))
    A = np.zeros((RH * SS, RW * SS), np.float32)
    ax0, ay0 = gm["right"] * SS + SS // 2 - br, gm["top"] * SS - bt
    ah, aw = big.shape
    sx0, sy0 = max(0, -ax0), max(0, -ay0)
    dx0, dy0 = max(ax0, 0), max(ay0, 0)
    dx1, dy1 = min(ax0 + aw, RW * SS), min(ay0 + ah, RH * SS)
    A[dy0:dy1, dx0:dx1] = big[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    S = np.diag([SS, SS, 1.0])
    Hbig = S @ Hinv @ np.linalg.inv(S)
    Abig = cv2.warpPerspective(A, Hbig, (cw * SS, ch * SS), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Ab0 = cv2.resize(Abig, (cw, ch), interpolation=cv2.INTER_AREA)
    ink = best["ink"]
    col, col_gray = ink_color(ob, og, bx, ink_mask_fn)
    if col is not None and col_gray > 0:
        ink = np.clip(col * (best["ink"] / col_gray), 0, 255)
        best["글자색"] = "#%02x%02x%02x" % tuple(int(v) for v in col[::-1])
    ys, xs = np.nonzero(Ab0 > 0.1)
    if len(xs) == 0:
        raise ValueError("원근 변환 후 글자가 비었음")
    sub = img[Y0:Y1, X0:X1].astype(np.float32)
    ring = sub[max(ys.min() - 3, 0):ys.max() + 4, max(xs.min() - 3, 0):xs.max() + 4].reshape(-1, 3)
    paper = np.median(ring, axis=0)
    k = np.clip(np.asarray(ink, np.float32) / np.maximum(paper, 1), 0, 1) if np.ndim(ink) else \
        np.full(3, min(1.0, float(ink) / max(float(np.median(ring)), 1)), np.float32)
    # 흐림은 기울기만 편 상태에서 고른 값을 그대로, 마지막에 한 번만 준다. 원근으로 편 원본에서 고르면
    # 원본이 다시 계산되며 흐려져 있어 흐림을 크게 골라 뿌옇게 됐음 (10/23 사용자 확인, auto00844·00160)
    Ab = cv2.GaussianBlur(Ab0, (0, 0), best["blur"]) if best["blur"] > 0 else Ab0
    img[Y0:Y1, X0:X1] = np.clip(sub * (1 - Ab[..., None] * (1 - k)), 0, 255).astype(np.uint8)
    if sym_img:
        import layout
        def to_image(fx, fy):
            p = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), Hinv)[0, 0]
            return float(p[0]) + X0, float(p[1]) + Y0
        layout.paste_symbols(img, sym_img, _sym_points(lay, syms, sym_img, gm, to_image))
    best["원근"] = "사다리꼴"
    return (X0 + int(xs.min()), Y0 + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1), best, top
