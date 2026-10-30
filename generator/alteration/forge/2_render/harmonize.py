"""B3~B6: 쓴 글자와 지운 자리를 주변과 어울리게 다듬는 후처리.

- B4 clean_area : 지운 자리에 남은 원래 획 흔적(옅은 그림자)을 없애고, 옆 종이 결을 떠서 입힌다 (글자 쓰기 전)
- B6 follow_curve: 휘어진 종이면 새로 쓴 글자를 표 선 곡선을 따라 구부린다 (글자 쓴 뒤)
- B3 ink_texture : 원본 획의 얼룩·잔결 크기를 재서 새 글자 획에 같은 만큼 입힌다 (글자 쓴 뒤)
- B5 save_like   : 원본이 jpg면 원본과 같은 압축표(qtables)와 색 샘플링으로 저장

모두 원본 이미지에서 잰 값으로 세기를 정한다 (고정 숫자를 최소화).
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import geometry


def _resid(a, sigma=2.5):
    a = a.astype(np.float32)
    base = cv2.GaussianBlur(a, (0, 0), sigma)
    return base, a - base


# ---------- B4 ----------

def _paper_patches(orig, box, ink_fn, rng, size, n=40):
    """칸 주변 원본 종이에서 글자·선이 없는 정사각 조각의 잔결(고주파) n개."""
    x, y, w, h = box
    H, W = orig.shape[:2]
    m = max(2 * h, 2 * size)
    X0, Y0, X1, Y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, W), min(y + h + m, H)
    reg = orig[Y0:Y1, X0:X1]
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    bad = ink_fn(g) > 0
    bad[max(y - Y0, 0):y - Y0 + h, max(x - X0, 0):x - X0 + w] = True     # 칸 안은 쓰지 않음
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    _, r = _resid(reg)
    ok = cv2.erode((~bad).astype(np.uint8), np.ones((size, size), np.uint8)) > 0   # 조각 전체가 종이인 중심
    ys, xs = np.nonzero(ok)
    if len(xs) < 5:
        return None
    out = []
    half = size // 2
    for j in rng.choice(len(xs), size=min(n, len(xs)), replace=len(xs) < n):
        cy, cx = ys[j], xs[j]
        p = r[cy - half:cy - half + size, cx - half:cx - half + size]
        if p.shape[:2] == (size, size):
            out.append(p)
    return out or None


def _synth(shape, patches, rng, size):
    """조각을 겹쳐 붙여(창 함수 가중 합) 잔결 장면을 만든다. 그대로 복사한 픽셀이 없게
    매번 뒤집기·돌리기를 하고 두 조각을 섞는다 (복사 탐지에 걸리지 않게)."""
    H, W = shape
    acc = np.zeros((H + size, W + size, 3), np.float32)
    wsum = np.zeros((H + size, W + size, 1), np.float32)
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)[..., None] + 1e-3
    step = size // 2
    for yy in range(0, H + 1, step):
        for xx in range(0, W + 1, step):
            a, b = rng.integers(len(patches), size=2)
            p = patches[a] * 0.7071 + np.rot90(patches[b], rng.integers(4)) * 0.7071
            if rng.random() < 0.5:
                p = p[:, ::-1]
            if p.shape[:2] != (size, size):
                continue
            acc[yy:yy + size, xx:xx + size] += p * win
            wsum[yy:yy + size, xx:xx + size] += win ** 2
    out = acc / np.sqrt(np.maximum(wsum, 1e-6))
    o = size // 2
    return out[o:o + H, o:o + W]


def clean_area(img, orig, box, ink_fn, stroke_fn, seed=0, margin=6):
    """B4. img: 원래 글자를 지운 이미지(제자리 수정). orig: 원본.
    1) 완만한 밝기(base)에서 원래 획 자리를 다시 메워 옅은 그림자를 없앤다
    2) 그림자가 있던 자리의 잔결(resid)만 주변 원본 종이 결로 바꾼다
    그림자가 없으면 아무것도 안 바꾼다. 표 선과 칸 주변의 다른 글자는 건드리지 않는다. 바꾼 픽셀 수(없으면 None)."""
    rng = np.random.default_rng(seed + 7)
    x, y, w, h = box
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = max(x - margin, 0), max(y - margin, 0), min(x + w + margin, W), min(y + h + margin, H)
    og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    old, lines = stroke_fn(og, box)                     # 칸 좌표
    old_full = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    old_full[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = old
    old_d = cv2.dilate(old_full, np.ones((5, 5), np.uint8))
    # 지키는 것: 표 선, 칸 밖 다른 글자 (원본 기준 잉크 중 원래 획이 아닌 것)
    keep = ink_fn(og[Y0:Y1, X0:X1]) > 0
    keep &= old_d == 0
    lf = np.zeros_like(old_full)
    lf[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = lines
    keep |= lf > 0
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    old_d[keep] = 0

    reg = img[Y0:Y1, X0:X1].astype(np.float32)
    base, resid = _resid(reg)
    if not old_d.any():
        return None
    # 1) 그림자 찾기: 원래 획 자리의 완만한 밝기를 주변으로 메운 값(base_c)보다 지금이 눈에 띄게 어두운 곳만
    #    (10/23 시험: 획 자리 전체를 바꾸면 그림자가 없는 문서까지 결이 바뀌어 탐지기에 더 걸림 4 -> 6/10)
    b8 = np.clip(base, 0, 255).astype(np.uint8)
    base_c = cv2.inpaint(b8, old_d, 5, cv2.INPAINT_TELEA).astype(np.float32)
    grain = float(resid[(old_d == 0) & ~keep].std()) if ((old_d == 0) & ~keep).sum() > 30 else 2.0
    dark = (base_c - base).mean(axis=2)
    ghost = (dark > max(1.0 * grain, 1.5)) & (old_d > 0)   # 10/23: 1.5배에서 낮춤 (옅은 흔적이 남던 경우)
    if ghost.sum() < 3:
        return None
    ghost = cv2.dilate(ghost.astype(np.uint8), np.ones((3, 3), np.uint8))
    ghost[keep] = 0
    wgt = cv2.GaussianBlur(ghost.astype(np.float32), (0, 0), 1.0)
    wgt = np.clip(wgt * 1.5, 0, 1)
    base = base * (1 - wgt[..., None]) + base_c * wgt[..., None]
    # 2) 그림자 자리의 결은 주변 원본 종이 결로 (지운 자리 결이 그림자 모양을 따라가지 않게)
    size = int(np.clip(round(h * 0.8), 8, 24)) // 2 * 2
    patches = _paper_patches(orig, box, ink_fn, rng, size)
    if patches is not None:
        tex = _synth(old_d.shape, patches, rng, size)
        ref = np.concatenate([p.reshape(-1, 3) for p in patches])
        tex *= (ref.std(axis=0) / np.maximum(tex.reshape(-1, 3).std(axis=0), 1e-3))
        # 서로 다른 두 잡음을 섞으면 세기가 줄어든다. 세기가 그대로 되도록 제곱합 1로 섞음
        resid = resid * np.sqrt(1 - wgt ** 2)[..., None] + tex * wgt[..., None]
    out = base + resid
    img[Y0:Y1, X0:X1] = np.clip(out, 0, 255).astype(np.uint8)
    return int(ghost.sum())


# ---------- B6 ----------

MIN_SAG = 0.8   # 칸 폭 절반에서 이만큼(px) 이상 처질 때만 구부림


def follow_curve(img, before, orig_gray, box):
    """B6. 새로 쓴 글자(img - before)만 표 선 곡선을 따라 세로로 옮긴다. 칸 가운데는 그대로.
    (처지는 정도 px, 적용 여부)"""
    a, sag, n = geometry.line_curve(orig_gray, box)
    # 숫자 높이로 곡선을 맞추는 방법도 시험했으나(10/23) 반듯한 문서에서도 0.6~1px 처짐이 잡혀 쓰지 않음.
    # 표 선이 없는 문서는 구부리지 않는다
    if n == 0 or abs(sag) < MIN_SAG:
        return round(sag, 2), False
    x, y, w, h = box
    H, W = img.shape[:2]
    m = max(h, 6)
    X0, Y0, X1, Y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, W), min(y + h + m, H)
    d = img[Y0:Y1, X0:X1].astype(np.float32) - before[Y0:Y1, X0:X1].astype(np.float32)
    xc = x + w / 2 - X0
    gx, gy = np.meshgrid(np.arange(X1 - X0, dtype=np.float32), np.arange(Y1 - Y0, dtype=np.float32))
    dy = a * (gx - xc) ** 2
    d2 = cv2.remap(d, gx, gy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img[Y0:Y1, X0:X1] = np.clip(before[Y0:Y1, X0:X1].astype(np.float32) + d2, 0, 255).astype(np.uint8)
    return round(sag, 2), True


# ---------- B3 ----------

def _core_std(gray, mask, frac=40):
    v = gray[mask].astype(np.float32)
    if v.size < 20:
        return None
    dark = v[v <= np.percentile(v, frac)]
    return float(dark.std())


def ink_texture(img, before, orig_gray, box, stroke_fn, seed=0):
    """B3. 원본 숫자 획의 진한 부분 밝기 흔들림(잉크 얼룩 + 사진 잡음)을 재고, 새 글자 획이 그보다 매끈하면
    모자란 만큼 잡음을 획 위에만 더한다. (원본 흔들림, 새 글자 흔들림(전), 더한 크기)"""
    x, y, w, h = box
    old = stroke_fn(orig_gray[y:y + h, x:x + w]) > 0
    so = _core_std(orig_gray[y:y + h, x:x + w], old)
    if so is None:
        return None
    H, W = img.shape[:2]
    m = max(h, 6)
    X0, Y0, X1, Y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, W), min(y + h + m, H)
    a_g = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_g = cv2.cvtColor(before[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = b_g - a_g
    top = np.percentile(diff[diff > 1], 99) if (diff > 1).sum() > 20 else 0
    if top <= 5:
        return None
    alpha = np.clip(diff / top, 0, 1)
    glyph = alpha > 0.15
    sn = _core_std(a_g, glyph)
    if sn is None:
        return None
    add = float(np.sqrt(max(so ** 2 - sn ** 2, 0.0)))
    if add < 0.5:
        return round(so, 1), round(sn, 1), 0.0
    rng = np.random.default_rng(seed + 13)
    n = rng.normal(0, 1, a_g.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    n *= add / (n.std() + 1e-6)
    # 진한 부분일수록 많이 (가장자리는 원래 흐림에 맡김)
    wgt = alpha ** 1.5
    out = img[Y0:Y1, X0:X1].astype(np.float32) + (n * wgt)[..., None]
    img[Y0:Y1, X0:X1] = np.clip(out, 0, 255).astype(np.uint8)
    return round(so, 1), round(sn, 1), round(add, 1)


# ---------- B5 ----------

def save_like(bgr, orig_path, out_path):
    """원본이 jpg면 같은 압축표·색 샘플링으로 jpg 저장, 아니면 png. 실제 저장 경로."""
    from PIL import JpegImagePlugin
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    src = Image.open(orig_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src.format == "JPEG" and getattr(src, "quantization", None):
        try:
            sub = JpegImagePlugin.get_sampling(src)
        except Exception:
            sub = -1
        p = out_path.with_suffix(".jpg")
        kw = {"qtables": src.quantization}
        if sub in (0, 1, 2):
            kw["subsampling"] = sub
        im.save(p, **kw)
        return p
    p = out_path.with_suffix(".png")
    im.save(p)
    return p


# ---------- C2 ----------

def match_noise(img, box, ink_fn, seed=0, ring=8):
    """C2: 다 만든 뒤 칸 안 종이의 잔결 세기를 칸 바로 바깥 종이와 같게 맞춘다.
    칸 안이 매끈하면 모자란 만큼 잡음을 더하고, 거칠면 잔결을 줄인다. 글자 획과 그 둘레 2px는 건드리지 않는다.
    (안 잔결, 바깥 잔결, 조정 후 안 잔결)"""
    x, y, w, h = box
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = max(x - ring, 0), max(y - ring, 0), min(x + w + ring, W), min(y + h + ring, H)
    reg = img[Y0:Y1, X0:X1].astype(np.float32)
    g = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    ink = cv2.dilate((ink_fn(g) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    inside = np.zeros(g.shape, bool)
    inside[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = True
    base, resid = _resid(reg, 3)
    rg = resid.mean(axis=2)
    pin, pout = inside & ~ink, ~inside & ~ink
    if pin.sum() < 30 or pout.sum() < 30:
        return None
    s_in, s_out = float(rg[pin].std()), float(rg[pout].std())
    wgt = cv2.GaussianBlur((inside & ~ink).astype(np.float32), (0, 0), 1.5)[..., None]
    if s_in < 0.9 * s_out:
        rng = np.random.default_rng(seed + 29)
        n = cv2.GaussianBlur(rng.normal(0, 1, g.shape).astype(np.float32), (0, 0), 0.6)
        n *= np.sqrt(s_out ** 2 - s_in ** 2) / (n.std() + 1e-6)
        reg = reg + n[..., None] * wgt
    elif s_in > 1.15 * s_out:
        f = s_out / s_in
        reg = base + resid * (1 - wgt * (1 - f))
    else:
        return round(s_in, 2), round(s_out, 2), round(s_in, 2)
    img[Y0:Y1, X0:X1] = np.clip(reg, 0, 255).astype(np.uint8)
    g2 = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    r2 = g2 - cv2.GaussianBlur(g2, (0, 0), 3)
    return round(s_in, 2), round(s_out, 2), round(float(r2[pin].std()), 2)


# ---------- C5 ----------

def clean_leftover(img, before_write, orig_gray, box, stroke_fn, seed=0, rounds=2):
    """C5: 새 글자를 다 쓴 뒤 마지막 검사. 원래 획이 있던 자리(둘레 2px)에서 새 글자가 아닌데 종이보다 어두운 픽셀이
    남아 있으면 그 픽셀만 주변 종이로 다시 메운다 (SD가 옅은 획을 다시 그리거나, 옅은 가장자리가 남은 경우).
    표 선과 새 글자는 건드리지 않는다. 다시 메운 픽셀 수."""
    x, y, w, h = box
    old, lines = stroke_fn(orig_gray, box)
    old = cv2.dilate(old, np.ones((5, 5), np.uint8)) > 0
    total = 0
    for r in range(rounds):
        cur = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        bef = cv2.cvtColor(before_write[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        new_glyph = cv2.dilate(((bef - cur) > 4).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        free = ~old | new_glyph
        paper_px = cur[~old & ~new_glyph & (lines == 0)]
        if paper_px.size < 20:
            break
        paper = float(np.median(paper_px))
        res = cur - cv2.GaussianBlur(cur, (0, 0), 3)
        grain = float(res[~old & ~new_glyph & (lines == 0)].std())
        bad = old & ~new_glyph & (lines == 0) & (cur < paper - max(2.0 * grain, 5.0))
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) & (~new_glyph).astype(np.uint8) & (lines == 0).astype(np.uint8)
        n = int(bad.sum())
        if n == 0:
            break
        total += n
        sub = img[y:y + h, x:x + w]
        filled = cv2.inpaint(sub, bad * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
        rng = np.random.default_rng(seed + 41 + r)
        noise = cv2.GaussianBlur(rng.normal(0, 1, bad.shape).astype(np.float32), (0, 0), 0.6)
        noise *= grain / (noise.std() + 1e-6)
        out = np.where(bad[..., None] > 0, np.clip(filled + noise[..., None], 0, 255), sub.astype(np.float32))
        img[y:y + h, x:x + w] = out.astype(np.uint8)
    return total


# ---------- C6 ----------

def cell_boxes(mask, gap=6):
    """마스크의 떨어진 덩어리(고친 칸)마다 상자. 비교 단계(forge_v2_compare._mask_boxes)와 같은 방식."""
    m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((gap, gap), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x + gap // 2), int(y + gap // 2), int(w - gap), int(h - gap)) for x, y, w, h, a in st[1:]]


def _det_noise(bgr, box):
    """탐지기 노이즈분석 값 (칸 안 ÷ 비슷한 주변), 칸 하나만 따로."""
    import sys
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from detector import measure
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    m = measure(rgb, bbox=box, padding=0, methods=["noise"])["noise"]
    return m["score"] if m else None


def match_cell_noise(img, orig, mask, ink_fn, save_fn, seed=0, floor=0.76, rounds=6):
    """C6: 고친 칸마다 따로, 저장된 파일 기준으로 종이 잔결 세기를 원본 같은 자리만큼 맞춘다.
    (10/30 auto00017: 여러 칸을 상자 하나로 재면 칸 하나가 매끈해도 안 보이고, jpg 저장에서 잔결이 깎임
     → 현금 칸 노이즈 0.56, 탐지 기준 0.66 아래)
    목표 = max(min(원본 같은 자리 값 × 0.9, 1), floor). 모자라면 칸 안 종이(글자 획 둘레 제외)에 잡음을 더하고 다시 저장해서 잰다.
    save_fn(img) -> 저장된 파일 경로. [(칸, 원본값, 처음값, 마지막값)]"""
    boxes = cell_boxes(mask)
    path = save_fn(img)
    targets = {}
    for b in boxes:
        t0 = _det_noise(orig, b)
        if t0 is not None:
            targets[b] = (t0, max(min(0.9 * t0, 1.0), floor))   # 원본이 주변보다 거친 칸도 1배까지만 (그 이상은 눈에 띄는 얼룩, 10/30)
    first, last = {}, {}
    rng = np.random.default_rng(seed + 61)
    for r in range(rounds + 1):
        cur = cv2.imread(str(path))
        low = []
        for b, (t0, tgt) in targets.items():
            v = _det_noise(cur, b)
            if v is None:
                continue
            first.setdefault(b, v)
            last[b] = v
            if v < tgt:
                low.append((b, v, tgt))
        if not low or r == rounds:
            break
        for b, v, tgt in low:
            x, y, w, h = b
            reg = img[y:y + h, x:x + w].astype(np.float32)
            g = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
            ink = cv2.dilate((ink_fn(g.astype(np.uint8)) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if (~ink).sum() < 20:
                continue
            wgt = cv2.GaussianBlur((~ink).astype(np.float32), (0, 0), 0.8)[..., None]
            # 모자란 만큼 더함: 잔결 세기 grain을 (목표/지금)배로 키우는 데 필요한 양 (jpg가 깎는 몫은 다음 바퀴에서 다시 잼)
            grain = float((g - cv2.GaussianBlur(g, (0, 0), 1.0))[~ink].std()) + 0.5
            # 한 번에 조금씩 (10/30 auto00053: 1.1배로 한 번에 넣어 목표 1.0을 1.5로 넘기고 점 얼룩이 보였음)
            a = grain * min(np.sqrt(max((tgt / max(v, 0.05)) ** 2 - 1, 0.05)), 1.0) * 0.6
            n = cv2.GaussianBlur(rng.normal(0, 1, g.shape).astype(np.float32), (0, 0), 0.7)
            n *= a / (n.std() + 1e-6)
            img[y:y + h, x:x + w] = np.clip(reg + n[..., None] * wgt, 0, 255).astype(np.uint8)
        path = save_fn(img)
    return [(b, round(targets[b][0], 3), round(first[b], 3), round(last[b], 3)) for b in targets if b in first]
