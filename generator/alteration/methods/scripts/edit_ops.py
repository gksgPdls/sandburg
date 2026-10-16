"""칸 안의 글자를 지우고 새 글자를 쓰는 공용 함수. 다른 스크립트가 가져다 쓴다.

- erase: 글자 획을 찾아 주변 종이로 메운다(inpaint). 흰 박스를 쓰지 않고, 종이 결을
  흉내 내도록 잡음을 더한다. 회의에서 정한 "스포이드로 배경색을 따서 덮기"에 해당한다
- write: 원본 글자의 높이·오른쪽 끝·진하기를 재서 새 글자를 맞춘다
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SUPERSAMPLE = 4


def ink_mask(gray_box):
    """칸 안의 글자 획(어두운 픽셀) 마스크."""
    paper = float(np.median(gray_box))
    return (gray_box < paper - 35).astype(np.uint8) * 255


def ink_bbox(gray, box):
    """칸 안 글자의 외곽 (x, y, w, h). 글자가 없으면 None."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w])
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return None
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def number_box(gray, box):
    """칸 안에서 오른쪽 끝에 붙은 글자 덩어리(숫자)의 영역 (x, y, w, h).

    합계 칸에는 숫자 왼쪽에 ①② 같은 표시가 떨어져 있는 경우가 있다. 획이 있는 열을
    이어 보다가 글자 높이의 0.8배보다 넓은 빈 열이 나오면 거기서 끊고, 가장 오른쪽
    덩어리만 숫자로 본다."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w]) > 0
    cols = np.nonzero(m.any(axis=0))[0]
    if len(cols) < 3:
        return None
    rows = np.nonzero(m.any(axis=1))[0]
    gap_limit = max(4, int(0.8 * (rows.max() - rows.min() + 1)))
    start = cols[-1]
    for a, b in zip(cols[::-1][1:], cols[::-1][:-1]):
        if b - a > gap_limit:
            break
        start = a
    return x + int(start), y, int(cols[-1] - start) + 1, h


def digit_extent(gray, box):
    """칸 안 숫자의 윗선과 높이. 쉼표처럼 아래로 처지는 좁은 획은 뺀다."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w]) > 0
    rows = m.sum(axis=1)
    if rows.max() == 0:
        return None
    on = np.nonzero(rows >= 0.25 * rows.max())[0]
    return y + int(on.min()), int(on.max() - on.min()) + 1


def ink_level(gray, box):
    """칸 안 글자의 진하기(어두운 쪽 5% 밝기)와 종이 밝기."""
    x, y, w, h = box
    g = gray[y:y + h, x:x + w]
    m = ink_mask(g) > 0
    paper = float(np.median(g[~m])) if (~m).any() else float(np.median(g))
    ink = float(np.percentile(g[m], 5)) if m.any() else paper
    return ink, paper


def erase(img, box, seed=0):
    """칸 안의 글자를 지운다. img(BGR)를 바꾸고, 지운 영역 외곽을 돌려준다."""
    x, y, w, h = box
    region = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    m = cv2.dilate(ink_mask(gray), np.ones((3, 3), np.uint8))
    if not m.any():
        return None
    filled = cv2.inpaint(region, m, 3, cv2.INPAINT_TELEA)
    # 메운 자리는 매끈해서 티가 난다. 주변 종이의 결(표준편차)만큼 잡음을 준다
    paper_std = float(gray[m == 0].std()) if (m == 0).any() else 2.0
    noise = np.random.default_rng(seed).normal(0, paper_std, filled.shape[:2])[..., None]
    filled = np.where(m[..., None] > 0, np.clip(filled + noise, 0, 255), filled).astype(np.uint8)
    img[y:y + h, x:x + w] = filled
    ys, xs = np.nonzero(m)
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def _font_for_digit_height(height, index=0):
    """숫자 '0' 의 높이가 height px 가 되는 폰트."""
    best, best_err = None, 1e9
    for s in np.arange(6, 60, 0.25):
        f = ImageFont.truetype(FONT, round(s * SUPERSAMPLE), index=index)
        _, t, _, b = f.getbbox("0")
        err = abs((b - t) / SUPERSAMPLE - height)
        if err < best_err:
            best, best_err = f, err
    return best


def write(img, text, right, top, digit_height, ink, blur=0.5, index=0):
    """오른쪽 끝을 right, 숫자 윗선을 top 에 맞춰 text 를 쓴다. 쓴 영역 외곽을 돌려준다."""
    f = _font_for_digit_height(digit_height, index)
    left, t0, r, b = f.getbbox(text)
    _, dt, _, _ = f.getbbox("0")
    pad = 4 * SUPERSAMPLE
    canvas = Image.new("L", (r - left + 2 * pad, b - t0 + 2 * pad), 0)
    ImageDraw.Draw(canvas).text((pad - left, pad - t0), text, font=f, fill=255)
    small = canvas.resize((canvas.width // SUPERSAMPLE, canvas.height // SUPERSAMPLE), Image.LANCZOS)
    alpha = np.asarray(small, np.float32) / 255.0
    if blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)

    ys, xs = np.nonzero(alpha > 0.1)
    # '0' 의 윗선이 top 에 오도록 (쉼표는 아래로 처지므로 숫자 기준으로 맞춘다)
    digit_top = pad // SUPERSAMPLE + (dt - t0) // SUPERSAMPLE
    x0 = right - int(xs.max())
    y0 = top - digit_top
    h, w = alpha.shape
    region = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    paper = float(np.median(region))
    k = min(1.0, ink / paper) if paper > 0 else 0.0
    img[y0:y0 + h, x0:x0 + w] = np.clip(region * (1 - alpha[..., None] * (1 - k)), 0, 255).astype(np.uint8)
    return x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def replace_text(img, box, new_text, seed=0):
    """칸의 글자를 지우고 new_text 를 원본 글자 위치·크기·진하기에 맞춰 쓴다.
    조작 영역(지운 곳 ∪ 새로 쓴 곳) 외곽을 돌려준다. 칸에 글자가 없으면 None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cell = box
    nb = number_box(gray, box)
    if nb is None:
        return None
    # 지우기·크기 측정은 숫자 덩어리 안에서만 한다 (왼쪽의 다른 표시를 지우지 않게)
    box = (max(cell[0], nb[0] - 2), cell[1], min(cell[0] + cell[2], nb[0] + nb[2] + 2) - max(cell[0], nb[0] - 2), cell[3])
    ib = ink_bbox(gray, box)
    if ib is None:
        return None
    ink, _ = ink_level(gray, box)
    top, height = digit_extent(gray, box)
    erased_img = img.copy()
    erased = erase(erased_img, box, seed)

    # 진하기: 번짐 때문에 합성 글자가 원본보다 옅게 나온다. 몇 가지 값으로 써 보고
    # 원본 글자의 진하기(어두운 쪽 5%)에 가장 가까운 것을 고른다
    best = None
    for offset in (0, 20, 40, 60, 80):
        trial = erased_img.copy()
        written = write(trial, new_text, right=ib[0] + ib[2] - 1, top=top,
                        digit_height=height, ink=max(0.0, ink - offset))
        x, y, w, h = cell
        # 새 글자가 칸 안쪽을 벗어나면 괘선을 덮으므로 쓰지 않는다
        if written[0] < x or written[1] < y or written[0] + written[2] > x + w \
                or written[1] + written[3] > y + h:
            return None
        got, _ = ink_level(cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY), box)
        if best is None or abs(got - ink) < abs(best[0] - ink):
            best = (got, trial, written)
    img[:] = best[1]
    return union(erased, best[2])


def union(*boxes):
    boxes = [b for b in boxes if b]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


# ---------- 글자 단위 바꾸기 (날짜의 일 두 자리 등) ----------

def glyph_boxes(gray, box):
    """칸 안 글자를 획 덩어리로 나눈 외곽 목록 (왼쪽부터). x 범위가 겹치는 조각은 한 글자로 합친다."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w])
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    parts = sorted([list(s[:4]) for s in st[1:] if s[4] >= 2], key=lambda s: s[0])
    merged = []
    for px, py, pw, ph in parts:
        if merged:
            mx, my, mw, mh = merged[-1]
            overlap = min(mx + mw, px + pw) - max(mx, px)
            if overlap > 0.5 * min(mw, pw):
                nx, ny = min(mx, px), min(my, py)
                merged[-1] = [nx, ny, max(mx + mw, px + pw) - nx, max(my + mh, py + ph) - ny]
                continue
        merged.append([px, py, pw, ph])
    return [(x + a, y + b, c, d) for a, b, c, d in merged]


def replace_glyphs(img, cell, glyphs, new_text, seed=0):
    """cell 안의 글자들(glyphs: 글자 외곽 목록)을 지우고 그 자리에 new_text 를 쓴다.
    새 글자의 오른쪽 끝·윗선·높이는 지운 글자에 맞춘다. 조작 영역 외곽 또는 None."""
    gx0 = min(g[0] for g in glyphs)
    gx1 = max(g[0] + g[2] for g in glyphs)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = (gx0 - 1, cell[1], gx1 - gx0 + 2, cell[3])
    # 글자 외곽(OCR 이 준 대략값)을 실제 획으로 다시 잰다
    ib = ink_bbox(gray, area)
    if ib is None:
        return None
    gx0, gx1, top, height = ib[0], ib[0] + ib[2], ib[1], ib[3]
    area = (gx0 - 1, cell[1], gx1 - gx0 + 2, cell[3])
    ink, _ = ink_level(gray, area)

    erased_img = img.copy()
    erased = erase(erased_img, area, seed)
    if erased is None:
        return None
    best = None
    for offset in (0, 20, 40, 60, 80):
        trial = erased_img.copy()
        written = write(trial, new_text, right=gx1 - 1, top=top, digit_height=height,
                        ink=max(0.0, ink - offset))
        x, y, w, h = cell
        if written[0] < x or written[1] < y or written[0] + written[2] > x + w \
                or written[1] + written[3] > y + h:
            return None
        got, _ = ink_level(cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY), area)
        if best is None or abs(got - ink) < abs(best[0] - ink):
            best = (got, trial, written)
    img[:] = best[1]
    return union(erased, best[2])


# ---------- 문서 안 글자 블록 옮기기 (숫자 조합, 날짜 복제) ----------

def glyph_ink(gray, cell, glyphs):
    """OCR 이 준 글자 외곽들(glyphs)의 범위 안에서 실제 획 외곽 (x, y, w, h)."""
    gx0 = min(g[0] for g in glyphs)
    gx1 = max(g[0] + g[2] for g in glyphs)
    return ink_bbox(gray, (gx0, cell[1], gx1 - gx0, cell[3]))


def copy_block(img, src, dst, dst_cell, seed=0):
    """src 획 외곽의 픽셀 블록을 dst 획 자리에 붙인다. dst 글자는 먼저 종이로 지운다.
    오른쪽 끝과 윗선을 dst 에 맞춘다. 블록이 dst_cell 밖으로 나가면 None."""
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    m = 1  # 획 가장자리의 번짐까지 같이 옮긴다
    bx, by, bw, bh = sx - m, sy - m, sw + 2 * m, sh + 2 * m
    px = dx + dw - sw - m          # 오른쪽 끝 맞춤
    py = dy - m                    # 윗선 맞춤
    cx, cy, cw, ch = dst_cell
    if px < cx or py < cy or px + bw > cx + cw or py + bh > cy + ch:
        return None
    block = img[by:by + bh, bx:bx + bw].copy()
    erased = erase(img, (dx - 1, cy, dw + 2, ch), seed)
    img[py:py + bh, px:px + bw] = block
    return union(erased, (px, py, bw, bh))
