"""변조 모델 고도화 (브랜치 feat/forgery-upgrade).

같은 원본 30장으로 단계를 하나씩 쌓아가며 비교한다. 결과는 outputs/고도화/.

    python scripts/generate/forge_v2.py select                 원본 30장 고르기 (원본목록.csv)
    python scripts/generate/forge_v2.py baseline               00_기존 (내 방식 복사, 동현님 방식 재현)
    python scripts/generate/forge_v2.py run --branch reuse --steps 1 --out 01_배경
    python scripts/generate/forge_v2.py run --branch ai --steps 1 --out 01_배경    (SD 필요, 맥에서 실행)
    python scripts/generate/forge_v2.py compare --out 01_배경 --prev 00_기존       비교 이미지와 점수

갈래
- ai     AI 인페인팅 (내 방식 개선): 글자 획 자리를 SD가 채우고 숫자는 폰트로 쓴다
- reuse  재사용 위조 (동현님 방식 개선): 글자 획 자리를 주변 종이로 메우고, 문서 안 글자를 재사용한다

단계 (--steps, 누적)
1  배경: 칸 전체를 칠하지 않고 글자 획 자리만 지운다. 표 선은 지우지 않는다
4  [AI] 폰트 맞추기: 원래 값을 후보 폰트로 그려 원본과 가장 닮은 폰트, 굵기, 흐림, 자간을 골라 새 값을 씀 (font_match.py)
6  [AI] SD 블렌딩: 글자 바깥 테두리만 SD로 약하게 다시 그린다 (기존 내 방식과 같은 처리)

    python scripts/generate/forge_v2.py run --branch ai --steps 1,6 --out 06_AI블렌딩 --name AI_블렌딩있음
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
sys.path.insert(0, str(ROOT))
import edit_ops  # noqa: E402  (동현님 함수)

# 맥 기본 폰트가 없는 환경(리눅스)에서는 비슷한 고딕체로 대신한다
_FALLBACK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.environ.get("FORGE_FONT"):
    edit_ops.FONT = os.environ["FORGE_FONT"]
elif not Path(edit_ops.FONT).exists() and Path(_FALLBACK_FONT).exists():
    edit_ops.FONT = _FALLBACK_FONT

ORIG_DIR = ROOT / "dataset" / "original" / "bankbook"
META_DIR = ROOT / "dataset" / "labels" / "meta"
OLD_TAMPERED = ROOT / "dataset" / "tampered" / "ai_inpainting"
OUT_ROOT = ROOT / "outputs" / "고도화"
# 바꿀 목록. 환경변수 FORGE_LIST 로 다른 파일을 지정할 수 있다 (팀원이 자기 데이터로 쓸 때)
LIST_CSV = Path(os.environ["FORGE_LIST"]).expanduser().resolve() if os.environ.get("FORGE_LIST") else OUT_ROOT / "원본목록.csv"
PAD = 2  # 지울 범위: 라벨 bbox에서 이만큼 넓힘 (흐린 획 가장자리 포함)


# ---------- 공통 ----------

def find_original(base):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = ORIG_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def load_list():
    """원본목록.csv. 환경변수 FORGE_LIMIT=10 이면 앞에서 10장만 (SD가 느려서 빠르게 볼 때)."""
    with open(LIST_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    limit = int(os.environ.get("FORGE_LIMIT", "0") or 0)
    return rows[:limit] if limit else rows


def read_bgr(path):
    return cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def save_image(bgr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)


def save_mask(shape, boxes, path):
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in boxes:
        m[max(y, 0):y + h, max(x, 0):x + w] = 255
    Image.fromarray(m).save(path)


def label_box(row, shape):
    x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
    x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
    x1, y1 = min(x + w + PAD, shape[1]), min(y + h + PAD, shape[0])
    return x0, y0, x1 - x0, y1 - y0


# ---------- 1단계: 글자 획만 지우기 ----------

def stroke_mask(gray, box):
    """칸 안 글자 획 마스크. 표 선은 어느 각도든 뺀다 (B1, geometry.py: LSD 선분 검출 + 예전 수평·수직 규칙).
    (획 마스크, 선 마스크) 칸 좌표 기준."""
    import geometry
    return geometry.stroke_mask(gray, box)


def paper_grain(gray, box, ring=10):
    """칸 주변 종이의 잔결 크기. 조명 같은 완만한 밝기 변화는 빼고, 픽셀 단위 흔들림만 잰다.
    (동현님 erase는 종이 전체 표준편차를 써서 잡음이 주변보다 1.5배 정도 세게 들어갔음)"""
    x, y, w, h = box
    y0, y1, x0, x1 = max(y - ring, 0), y + h + ring, max(x - ring, 0), x + w + ring
    g = gray[y0:y1, x0:x1].astype(np.float32)
    paper = edit_ops.ink_mask(gray[y0:y1, x0:x1]) == 0
    paper = cv2.erode(paper.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    resid = g - cv2.GaussianBlur(g, (0, 0), 3)
    return float(resid[paper].std()) if paper.sum() > 20 else 2.0


def stroke_only(gray_box):
    """칸 이미지 하나에서 표 선을 뺀 글자 획 마스크 (font_match에 넘기는 용도)."""
    m, _ = stroke_mask(gray_box, (0, 0, gray_box.shape[1], gray_box.shape[0]))
    return m


def erase_reuse(img, box, seed=0):
    """[재사용] 동현님 erase 방식(글자 획만 주변 종이로 메우고 종이 결 잡음).
    바꾼 점: 표 선 보호, 잡음 크기를 주변 종이 잔결에 맞춤."""
    x, y, w, h = box
    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, lines = stroke_mask(full_gray, box)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m[lines > 0] = 0
    if not m.any():
        return None, m
    region = img[y:y + h, x:x + w]
    filled = cv2.inpaint(region, m, 3, cv2.INPAINT_TELEA).astype(np.float32)
    # inpaint 결과 자체도 약간 흔들림이 있으므로, 최종 잔결이 주변과 같아지게 잡음 크기를 정한다
    target = paper_grain(full_gray, box)
    fg = cv2.cvtColor(filled.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    have = float((fg - cv2.GaussianBlur(fg, (0, 0), 3))[m > 0].std())
    sigma = float(np.sqrt(max(target ** 2 - have ** 2, 0.0)))
    noise = np.random.default_rng(seed).normal(0, 1, filled.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    noise *= sigma / (noise.std() + 1e-6)
    out = np.where(m[..., None] > 0, np.clip(filled + noise[..., None], 0, 255), filled)
    img[y:y + h, x:x + w] = out.astype(np.uint8)
    return box, m


_PIPE = None


AI_ERASE_STRENGTH = 0.35  # SD가 지운 자리를 다시 그리는 세기. 1.0이면 원래 숫자를 통째로, 0.5도 획 조각을 다시 그려냄(10/16 확인)


def erase_ai(img, box, seed=0):
    """[AI] 글자 획 자리만 AI가 채운다.
    SD 인페인팅에 원래 글자가 있는 그대로 마스크를 주면, 주변 맥락을 보고 숫자 획을 다시 그려서
    원래 숫자가 겹쳐 보였다(1차 시도 실패). 그래서 먼저 획 자리를 종이로 메운 뒤(재사용과 같은 지우기),
    SD가 그 자리만 strength 0.5로 다시 그려 종이 질감을 AI가 만든 것으로 바꾼다."""
    global _PIPE
    sys.path.insert(0, str(ROOT / "scripts" / "generate"))
    import ai_inpainting as old
    import torch
    if _PIPE is None:
        _PIPE = old.load_pipeline(old.load_hf_token())
    erased, m = erase_reuse(img, box, seed)
    if erased is None:
        return None, m
    x, y, w, h = box
    H, W = img.shape[:2]
    side = max(w, h) + 2 * old.CROP_CONTEXT_MARGIN
    cx, cy = x + w / 2, y + h / 2
    cx0, cy0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    cx1, cy1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    crop = Image.fromarray(cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
    full_mask = np.zeros((H, W), np.uint8)
    full_mask[y:y + h, x:x + w] = cv2.dilate(m, np.ones((3, 3), np.uint8))
    mcrop = Image.fromarray(full_mask[cy0:cy1, cx0:cx1])
    R = old.MODEL_RESOLUTION
    gen = torch.Generator("cpu").manual_seed(seed)
    out = _PIPE(prompt="clean blank paper texture, scanned document background",
                negative_prompt="text, letters, digits, numbers, ink, lines, strokes",
                image=crop.resize((R, R), Image.LANCZOS),
                mask_image=mcrop.resize((R, R), Image.NEAREST).point(lambda v: 255 if v > 0 else 0),
                strength=AI_ERASE_STRENGTH,
                num_inference_steps=old.NUM_INFERENCE_STEPS, guidance_scale=old.GUIDANCE_SCALE,
                generator=gen).images[0]
    out = cv2.cvtColor(np.asarray(out.resize(crop.size, Image.LANCZOS)), cv2.COLOR_RGB2BGR).astype(np.float32)
    sub = img[cy0:cy1, cx0:cx1]
    mm = full_mask[cy0:cy1, cx0:cx1] > 0
    # SD가 획 조각을 다시 그리면 종이보다 어두워진다. 메운 종이 밝기보다 종이 결 3배 이상 어두운 픽셀은 막는다
    base = cv2.GaussianBlur(sub.astype(np.float32), (0, 0), 3)
    floor = base - 3 * paper_grain(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box)
    out = np.maximum(out, floor)
    sub[mm] = np.clip(out, 0, 255).astype(np.uint8)[mm]  # 획 자리만 바꾸고 나머지 픽셀은 그대로
    return box, m


def ring_blend(img, box, seed=0):
    """[AI] 6단계: 기존 내 방식처럼 글자 영역 바깥 테두리(14px)만 SD로 약하게 다시 그려 경계를 섞는다.
    글자 영역 자체는 SD가 건드리지 않는다."""
    global _PIPE
    sys.path.insert(0, str(ROOT / "scripts" / "generate"))
    import ai_inpainting as old
    import torch
    if _PIPE is None:
        _PIPE = old.load_pipeline(old.load_hf_token())
    x, y, w, h = box
    H, W = img.shape[:2]
    side = max(w, h) + 2 * old.CROP_CONTEXT_MARGIN
    cx, cy = x + w / 2, y + h / 2
    cx0, cy0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    cx1, cy1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    r = old.RING_MARGIN
    ring = np.zeros((H, W), np.uint8)
    ring[max(y - r, 0):y + h + r, max(x - r, 0):x + w + r] = 255
    ring[y:y + h, x:x + w] = 0
    crop = Image.fromarray(cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
    R = old.MODEL_RESOLUTION
    gen = torch.Generator("cpu").manual_seed(seed)
    out = _PIPE(prompt="scanned document paper, printed text, natural texture",
                image=crop.resize((R, R), Image.LANCZOS),
                mask_image=Image.fromarray(ring[cy0:cy1, cx0:cx1]).resize((R, R), Image.NEAREST),
                strength=old.SD_BLEND_STRENGTH, num_inference_steps=old.NUM_INFERENCE_STEPS,
                guidance_scale=old.GUIDANCE_SCALE, generator=gen).images[0]
    out = cv2.cvtColor(np.asarray(out.resize(crop.size, Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    sub = img[cy0:cy1, cx0:cx1]
    mm = ring[cy0:cy1, cx0:cx1] > 0
    sub[mm] = out[mm]
    return (max(x - r, 0), max(y - r, 0), w + 2 * r, h + 2 * r)


# ---------- 새 값 쓰기 (1단계에서는 동현님 write 그대로, 4단계에서 개선) ----------

def free_left(gray, box, limit):
    """box 왼쪽으로 글자나 선이 나오기 전까지 빈 종이 폭."""
    x, y, w, h = box
    x0 = max(x - limit, 0)
    strip = edit_ops.ink_mask(gray[y:y + h, x0:x]) if x > x0 else np.zeros((h, 0), np.uint8)
    cols = np.nonzero(strip.any(axis=0))[0]
    return (x - x0) if len(cols) == 0 else (x - x0 - 1 - int(cols.max()))


def uses_period(gray, box, seen, glyphs, gm):
    """원본이 천 단위 구분에 쉼표 대신 점을 쓰는지. OCR은 작은 쉼표를 점으로 자주 읽어서 OCR 글자는 믿지 않고,
    숫자 아래쪽의 작은 획 덩어리(구분 기호)를 직접 본다. 꼬리가 숫자 아랫선 밑으로 내려오거나 세로로 길면 쉼표.
    (seen, glyphs는 호환용으로 받기만 함)"""
    m, _ = stroke_mask(gray, box)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    top = gm["top"] - box[1]
    bottom = top + gm["height"] - 1
    seps = [(x, y, w, h) for x, y, w, h, _ in st[1:]
            if h <= 0.5 * gm["height"] and y + h / 2 > top + 0.6 * gm["height"] and w <= 0.5 * gm["height"]]
    if not seps:
        return False
    return all(y + h - 1 <= bottom and h <= w + 1 for x, y, w, h in seps)


MIN_TILT = 0.3   # 이보다 작은 기울기(도)는 반듯하게 씀
MIN_PERSP = 0.5  # 칸 사다리꼴이 이만큼(도) 이상 찌그러졌으면 원근 변환(B2-2), 아니면 회전만


def text_angle(geo):
    """새 글자를 기울일 각도. 글자 밑선(점 3개 이상)을 우선 쓰되, 표 선 기울기와 2도 넘게 다르면
    밑선 측정이 틀렸다고 보고 표 선 기울기를 쓴다. 둘 다 없으면 0."""
    b, d = geo.get("baseline_angle"), geo.get("doc_angle")
    if b is not None and geo.get("baseline_pts", 0) >= 3:
        if d is not None and abs(b - d) > 2:
            return d
        return b
    return d if d is not None else 0.0


def glyph_metrics(gray, box):
    """칸 안 글자의 오른쪽 끝, 숫자 윗선, 높이, 진하기. 표 선은 빼고 잰다.
    (동현님 ink_bbox, digit_extent는 칸 안에 표 선이 걸치면 선까지 글자로 봐서 글자가 크게 써졌음)"""
    x, y, w, h = box
    m, _ = stroke_mask(gray, box)
    m = m > 0
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return None
    rows = m.sum(axis=1)
    on = np.nonzero(rows >= 0.25 * rows.max())[0]
    g = gray[y:y + h, x:x + w]
    ink = float(np.percentile(g[m], 5))
    return {"right": x + int(xs.max()), "left": x + int(xs.min()),
            "top": y + int(on.min()), "height": int(on.max() - on.min()) + 1, "ink": ink}


def write_value(img, orig_gray, box, text):
    """원래 글자의 오른쪽 끝, 윗선, 높이, 진하기에 맞춰 text를 쓴다. 쓴 영역 외곽을 돌려준다.
    새 값이 길어 왼쪽 글자나 선을 덮게 되면 글자 높이를 조금씩 줄인다(최대 20%)."""
    gm = glyph_metrics(orig_gray, box)
    ink, top, height, right = gm["ink"], gm["top"], gm["height"], gm["right"]
    room = right - box[0] + 1 + free_left(orig_gray, box, limit=box[2] * 3) - 2
    best = None
    for scale in (1.0, 0.95, 0.9, 0.85, 0.8):
        for offset in (0, 20, 40, 60, 80):
            trial = img.copy()
            hh = max(int(round(height * scale)), 5)
            written = edit_ops.write(trial, text, right=right, top=top + (height - hh) // 2,
                                     digit_height=hh, ink=max(0.0, ink - offset))
            if written[2] > room:
                break
            got, _ = edit_ops.ink_level(cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY), box)
            if best is None or abs(got - ink) < abs(best[0] - ink):
                best = (got, trial, written)
        if best is not None:
            break
    if best is None:  # 공간이 부족해도 쓰기는 한다 (기록에 표시)
        trial = img.copy()
        written = edit_ops.write(trial, text, right=right, top=top, digit_height=height, ink=ink)
        best = (None, trial, written)
    img[:] = best[1]
    return best[2], best[0] is None


# ---------- 명령 ----------

def cmd_select(args):
    metas = sorted(META_DIR.glob("*_ai_inpaint.json"))
    rows = []
    for p in metas:
        meta = json.loads(p.read_text(encoding="utf-8"))
        base = p.stem.replace("_ai_inpaint", "")
        orig = find_original(base)
        reg = meta["forged_regions"][0]
        if orig is None or reg["bbox"]["h"] < 9:
            continue
        rows.append({"id": base, "original": str(orig.relative_to(ROOT)),
                     "original_value": reg["original_value"], "forged_value": reg["forged_value"],
                     "x": reg["bbox"]["x"], "y": reg["bbox"]["y"], "w": reg["bbox"]["w"], "h": reg["bbox"]["h"],
                     "kind": "금액" if "," in reg["original_value"] else "날짜"})
    rng = random.Random(0)
    money = [r for r in rows if r["kind"] == "금액"]
    date = [r for r in rows if r["kind"] == "날짜"]
    pick = rng.sample(money, min(20, len(money))) + rng.sample(date, min(10, len(date)))
    pick.sort(key=lambda r: r["id"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LIST_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pick[0]))
        wr.writeheader()
        wr.writerows(pick)
    print(f"{len(pick)}장 선택 (금액 {sum(r['kind']=='금액' for r in pick)}, 날짜 {sum(r['kind']=='날짜' for r in pick)}) -> {LIST_CSV}")


# 앞의 것이 안 되면 다음 것을 시도 (모두 동현님 코드 그대로)
DH_OPS = {"금액": {"write": ["digit", "sum_break"], "copy": ["digit_combo", "copymove"]},
          "날짜": {"write": ["date_shift", "date_conflict"], "copy": ["date_copy"]}}


def erase_v2_for_dh(img, box, seed=0):
    """동현님 edit_ops.erase와 같은 입출력으로 감싼 1단계 지우기 (실행 중에만 바꿔 끼움, 동현님 파일은 그대로)."""
    _, m = erase_reuse(img, box, seed)
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    return box[0] + int(xs.min()), box[1] + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1


def dh_real(orig, kind, mode, seed, step1=False):
    """기존 팀원 방식: 동현님 auto_receipt.py 조작 함수를 고치지 않고 그대로 돌린 결과.
    칸과 새 값은 동현님 코드가 직접 고른다(금액: digit / digit_combo, 날짜: date_shift / date_copy).
    못 하면 None (동현님 코드도 그 원본은 건너뜀)."""
    import random as _random
    import auto_receipt as AR
    original_erase = edit_ops.erase
    if step1:
        edit_ops.erase = erase_v2_for_dh
    try:
        return _dh_real(AR, _random, orig, kind, mode, seed)
    finally:
        edit_ops.erase = original_erase


def _dh_real(AR, _random, orig, kind, mode, seed):
    oriented = {}
    for op in DH_OPS[kind][mode]:
        dates = op in AR.DATE_OPS
        if dates not in oriented:
            oriented[dates] = AR.orient(orig, dates=dates)
        up, table, k = oriented[dates]
        im = up.copy()
        info = AR.FUNCS[op](im, table, _random.Random(seed))
        if info:
            break
    else:
        return None
    mask = AR.changed_mask(up, im)
    return (np.ascontiguousarray(np.rot90(im, -k)), np.ascontiguousarray(np.rot90(mask, -k)),
            f"{op}: {info['description']}")


def copy_digit(img, orig, orig_gray, box, row, lib, seed):
    """[글자 복사] 라벨 칸의 원래 값에서 한 글자만, 문서 안에 있는 같은 크기의 다른 숫자로 바꾼다.
    금액은 맨 앞자리를 더 큰 숫자로, 날짜는 일(日)의 끝자리를 바꾼다(동현님 digit_combo, date_copy와 같은 규칙).
    문서에 쓸 글자가 없으면 None (건너뜀). 지우기는 1단계 방식(erase_reuse)."""
    import random as _random
    import auto_receipt as AR
    import receipt
    text, glyphs = receipt.char_boxes(orig_gray, box, whitelist="0123456789,-")
    want = row["original_value"]
    if text != want or len(glyphs) != len(text):
        return None, f"OCR 불일치({text})"
    if row["kind"] == "금액":
        pos = next(i for i, ch in enumerate(text) if ch.isdigit())
        ok = lambda d: int(d) > int(text[pos])
    else:
        pos = len(text) - 1
        tens = int(text[-2])
        ok = lambda d: d != text[pos] and 1 <= tens * 10 + int(d) <= 28
    gx = glyphs[pos][0]
    full_m, _ = stroke_mask(orig_gray, box)  # 표 선은 빼고 글자 획만 (칸 전체로 선을 찾은 뒤 그 글자 열만)
    gm_ = full_m[:, max(gx - box[0], 0):gx - box[0] + glyphs[pos][2]]
    ys, xs = np.nonzero(gm_)
    dst = (gx + int(xs.min()), box[1] + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1) if len(xs) else None
    if not dst:
        return None, "글자 위치 못 찾음"
    donors = [g for g in lib if g["ch"].isdigit() and ok(g["ch"]) and AR._similar(g["ink"], dst)]
    if not donors:
        return None, "문서에 쓸 수 있는 같은 크기 숫자 없음"
    rng = _random.Random(seed)
    rng.shuffle(donors)
    g = min(donors, key=lambda g: abs(g["ink"][0] - dst[0]) + abs(g["ink"][1] - dst[1]))
    sx, sy, sw, sh = g["ink"]
    dx, dy, dw, dh = dst
    m = 1
    block = orig[sy - m:sy + sh + m, sx - m:sx + sw + m].copy()
    erase_box = (dx - 1, box[1], dw + 2, box[3])
    erase_reuse(img, erase_box, seed)
    px, py = dx + dw - sw - m, dy - m
    img[py:py + block.shape[0], px:px + block.shape[1]] = block
    new_text = text[:pos] + g["ch"] + text[pos + 1:]
    return edit_ops.union(erase_box, (px, py, block.shape[1], block.shape[0])), f"{text} → {new_text}"


def _library(img, box):
    import glyph_library as GL
    return GL.build(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), exclude=box)


def _write_log(path, log):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        keys = list(dict.fromkeys(k for e in log for k in e))   # 행마다 칸이 달라도 모두 기록
        wr = csv.DictWriter(f, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(log)


def cmd_baseline(args):
    """00_기존: 내방식(기존 결과 복사), 동현님_write, 동현님_copy(동현님 코드 그대로 실행한 결과)."""
    import zlib
    out = OUT_ROOT / "00_기존"
    mine, dw, dc = out / "내방식", out / "동현님_write", out / "동현님_copy"
    for d in (mine, dw, dc):
        d.mkdir(parents=True, exist_ok=True)
    logs = {d: [] for d in (mine, dw, dc)}
    rows = load_list()
    for i, row in enumerate(rows):
        sid = row["id"]
        shutil.copy(OLD_TAMPERED / f"{sid}_ai_inpaint.png", mine / f"{sid}.png")
        orig = read_bgr(ROOT / row["original"])
        box = label_box(row, orig.shape)
        save_mask(orig.shape, [box], mine / f"{sid}_mask.png")
        logs[mine].append({"id": sid, "변경": f'{row["original_value"]} → {row["forged_value"]}'})
        for d, mode in ((dw, "write"), (dc, "copy")):
            res = dh_real(orig, row["kind"], mode, zlib.crc32(sid.encode()))
            if res is None:
                logs[d].append({"id": sid, "변경": "건너뜀(동현님 코드가 고칠 칸을 못 찾음)"})
                continue
            im, mask, desc = res
            save_image(im, d / f"{sid}.jpg")
            Image.fromarray(mask).save(d / f"{sid}_mask.png")
            logs[d].append({"id": sid, "변경": desc})
        print(f"{i + 1}/{len(rows)} {sid}", flush=True)
    for d, log in logs.items():
        _write_log(d / "생성기록.csv", log)
    print("00_기존 완료")


def cmd_run(args):
    steps = {int(s) for s in args.steps.split(",")}
    if args.branch == "ai":
        sub = args.name or "AI"
    else:
        sub = args.name or f"재사용_{args.mode}"
    if 6 in steps and args.branch != "ai":
        raise SystemExit("6단계(SD 블렌딩)는 AI 갈래만 해당")
    if 1 not in steps:
        raise SystemExit("1단계 없이 실행하는 경우는 아직 없음")
    out = OUT_ROOT / args.out / sub
    out.mkdir(parents=True, exist_ok=True)
    log = []
    rows = load_list()
    for i, row in enumerate(rows):
        sid = row["id"]
        ext = ".png" if args.branch == "ai" else ".jpg"  # 5단계 전까지는 기존 갈래 형식 유지
        if args.resume and ((out / f"{sid}{ext}").exists() or (5 in steps and (out / f"{sid}_mask.png").exists())):
            continue
        orig = read_bgr(ROOT / row["original"])
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        img = orig.copy()
        box = label_box(row, img.shape)
        entry = {"id": sid, "steps": args.steps, "font": Path(edit_ops.FONT).name,
                 "변경": f'{row["original_value"]} → {row["forged_value"]}'}
        if args.branch == "ai":
            if args.no_sd:
                erase_reuse(img, box, seed=i)   # SD 없이 빠르게 볼 때 (리눅스 시험용)
            else:
                erase_ai(img, box, seed=i)
            if not args.no_b4:   # B4: 원래 획 그림자 없애고 주변 종이 결 입히기
                import harmonize as HZ
                entry["그림자px"] = HZ.clean_area(img, orig, box, edit_ops.ink_mask, stroke_mask, seed=i) or 0
            before_write = img.copy()
            if 4 in steps:
                import font_match as FM
                if not any(k in n.lower() for _, _, n in FM.candidates() for k in ("gulim", "dotum", "batang", "malgun")):
                    raise SystemExit("윈도우 글꼴(굴림, 돋움, 바탕, 맑은 고딕)을 못 찾음. fonts/README.md 참고해서 fonts/ 에 넣은 뒤 다시 실행")
                scoring_bg = orig.copy()
                erase_reuse(scoring_bg, box, seed=i)  # 폰트 비교용 배경 (원래 글자를 지운 칸)
                gm = glyph_metrics(orig_gray, box)
                orig_val, new_val = row["original_value"], row["forged_value"]
                # 원본이 천 단위에 쉼표 대신 점을 쓰면(예: 140.000) 새 값도 점으로 (라벨은 OCR이라 쉼표로 적힌 경우 있음)
                import receipt
                seen, gl = receipt.char_boxes(orig_gray, box, whitelist="0123456789,.-")
                # 점 구분 판단은 사진 문서에서 작은 쉼표를 점으로 오판해서 끔(10/23, auto00304). 라벨의 쉼표를 따른다
                if False and "," in orig_val and uses_period(orig_gray, box, seen, gl, gm):
                    orig_val, new_val = seen, new_val.replace(",", ".")
                    entry["구분기호"] = "점"
                    entry["변경"] = f"{orig_val} → {new_val}"
                else:
                    seen = seen.replace(".", ",") if "," in orig_val else seen  # OCR이 쉼표를 점으로 읽은 경우
                    same_shape = len(seen) == len(orig_val) and all(a.isdigit() == b.isdigit() for a, b in zip(seen, orig_val))
                    n_diff = sum(a != b for a, b in zip(seen, orig_val)) if same_shape else 99
                    # OCR이 한두 글자만 다르게 읽은 경우만 OCR을 믿는다 (사진 문서에서 OCR이 깨지는 경우 auto00804)
                    if same_shape and 0 < n_diff <= 2:
                        entry["원래값_OCR"] = seen   # 라벨의 원래 값이 OCR 오류인 경우 (예: 140,000을 440,000으로)
                        orig_val = seen
                import geometry
                geo = geometry.measure(orig_gray, box)
                theta = text_angle(geo)
                entry["문서기울기"] = round(geo["doc_angle"], 2) if geo["doc_angle"] is not None else ""
                entry["밑선기울기"] = round(geo["baseline_angle"], 2) if geo["baseline_angle"] is not None else ""
                quad, how = (None, "끔") if args.no_persp else geometry.cell_quad(orig_gray, box)
                persp = geometry.perspective_strength(quad) if quad is not None else 0.0
                entry["원근"] = f"{how} {persp:.2f}도" if quad is not None else how
                if quad is not None and persp >= MIN_PERSP and not args.no_tilt:
                    written, best, top = FM.write_matched_perspective(img, scoring_bg, orig_gray, orig, box, orig_val,
                                                                      new_val, quad, stroke_only, glyph_metrics, theta=theta)
                elif abs(theta) >= MIN_TILT and not args.no_tilt:
                    written, best, top = FM.write_matched_rotated(img, scoring_bg, orig_gray, orig, box, orig_val,
                                                                  new_val, theta, stroke_only, glyph_metrics)
                else:
                    written, best, top = FM.write_matched(img, scoring_bg, orig_gray, box, orig_val,
                                                          new_val, gm, stroke_only, orig_bgr=orig)
                entry["적용기울기"] = best.get("기울기", 0)
                entry["글꼴기준"] = best.get("글꼴기준", "")
                entry["간격차이"] = best.get("간격", "")
                entry["가로배율"] = best.get("가로배율", "")
                entry.update({"폰트": best["name"], "굵기": best["weight"], "흐림": best["blur"],
                              "글자색": best.get("글자색", ""),
                              "자간": round(best["tracking"], 2), "차이": round(best["score"], 2),
                              "후보2_3": " / ".join(f'{t["name"]}({t["score"]:.2f})' for t in top[1:3])})
            else:
                written, overflow = write_value(img, orig_gray, box, row["forged_value"])
                entry["넘침"] = "Y" if overflow else ""
            import harmonize as HZ
            if not args.no_b6:   # B6: 휘어진 종이면 글자를 표 선 곡선에 맞춰 구부림
                sag, bent = HZ.follow_curve(img, before_write, orig_gray, box)
                entry["휘어짐px"] = sag
                entry["구부림"] = "Y" if bent else ""
            if not args.no_b3:   # B3: 새 글자 획에 원본 획만큼 얼룩·잔결
                t = HZ.ink_texture(img, before_write, orig_gray, box, stroke_only, seed=i)
                if t:
                    entry["획흔들림_원본"], entry["획흔들림_새"], entry["획잡음"] = t
        else:
            # 1단계 적용 팀원 방식: 동현님 코드를 그대로 돌리되 지우기만 1단계 방식으로 바꿔 끼움.
            # 난수가 같아서 기존 팀원 방식과 같은 칸, 같은 새 값이 나온다
            import zlib
            res = dh_real(orig, row["kind"], args.mode, zlib.crc32(sid.encode()), step1=True)
            if res is None:
                log.append({**entry, "변경": "건너뜀(동현님 코드가 고칠 칸을 못 찾음)"})
                continue
            im, mask, desc = res
            save_image(im, out / f"{sid}{ext}")
            Image.fromarray(mask).save(out / f"{sid}_mask.png")
            log.append({**entry, "변경": desc})
            print(f"{i + 1}/{len(rows)} {sid}", flush=True)
            continue
        extra = []
        if 6 in steps:
            extra.append(ring_blend(img, edit_ops.union(box, written), seed=i))
        if 5 in steps:   # B5: 원본 jpg 압축표 그대로 저장
            import harmonize as HZ
            for old_f in (out / f"{sid}.png", out / f"{sid}.jpg"):
                if old_f.exists():
                    try:
                        old_f.unlink()   # 예전 형식 결과가 남아 비교에 섞이지 않게
                    except OSError:
                        pass
            entry["저장"] = HZ.save_like(img, ROOT / row["original"], out / sid).suffix
        else:
            save_image(img, out / f"{sid}{ext}")
        save_mask(img.shape, ([box] if args.branch == "ai" or args.mode == "write" else []) + [written] + extra,
                  out / f"{sid}_mask.png")
        log.append(entry)
        print(f"{i + 1}/{len(rows)} {sid}", flush=True)
    if log:
        keys = []
        for e in log:
            keys += [k for k in e if k not in keys]
        log = [{k: e.get(k, "") for k in keys} for e in log]
        _write_log(out / "생성기록.csv", log)


def cmd_export_win(args):
    """4단계 폰트 맞추기를 윈도우 글꼴로 하기 위한 입력 폴더 (outputs/고도화/04_윈도우입력).
    맥에서 SD 지우기, 측정값, 글자 위치(OCR)를 미리 만들어 두고, 윈도우에서는 글꼴 고르기와 쓰기만 한다."""
    import json as _json
    import receipt
    import font_match as FM
    out = OUT_ROOT / "04_윈도우입력"
    for sub in ("original", "erased", "scoring_bg"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    samples = []
    rows = load_list()
    for i, row in enumerate(rows):
        sid = row["id"]
        orig = read_bgr(ROOT / row["original"])
        og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        box = label_box(row, orig.shape)
        gm = glyph_metrics(og, box)
        ov, nv = row["original_value"], row["forged_value"]
        seen, glyphs = receipt.char_boxes(og, box, whitelist="0123456789,.-")
        if "," in ov and uses_period(og, box, seen, glyphs, gm):
            ov, nv = seen, nv.replace(",", ".")
        elif "." in seen and "," in ov:
            seen = seen.replace(".", ",")   # OCR이 쉼표를 점으로 읽은 경우
        if len(seen) == len(ov) and seen != ov and not ("." in seen and "," in ov):
            ov = seen   # 라벨의 원래 값이 OCR 오류인 경우
        img = orig.copy()
        if args.no_sd:
            erase_reuse(img, box, seed=i)
        else:
            erase_ai(img, box, seed=i)
        bg = orig.copy()
        erase_reuse(bg, box, seed=i)
        save_image(orig, out / "original" / f"{sid}.png")
        save_image(img, out / "erased" / f"{sid}.png")
        save_image(bg, out / "scoring_bg" / f"{sid}.png")
        samples.append({"id": sid, "original_value": ov, "forged_value": nv, "box": list(map(int, box)),
                        **{k: (int(gm[k]) if k != "ink" else float(gm[k])) for k in ("right", "left", "top", "height", "ink")},
                        "glyphs": [[ch, *map(int, g)] for ch, g in zip(seen, glyphs)] if len(seen) == len(ov) else []})
        print(f"{i + 1}/{len(rows)} {sid}", flush=True)
    (out / "samples.json").write_text(_json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    src = (ROOT / "scripts" / "generate" / "font_match.py").read_text(encoding="utf-8")
    drv = (ROOT / "scripts" / "generate" / "font_match_win_driver.py").read_text(encoding="utf-8")
    (out / "font_match_win.py").write_text(src + drv, encoding="utf-8")
    (out / "읽어보기.txt").write_text(
        "윈도우 노트북에서 실행\n"
        "1. pip install numpy opencv-python pillow\n"
        "2. 이 폴더에서: python font_match_win.py\n"
        "3. 결과: 옆 폴더 04_윈도우결과/AI_윈도우폰트 -> 맥 outputs/고도화/04_폰트맞추기_윈도우/ 로 복사\n"
        "글꼴은 C:\\Windows\\Fonts 에서 읽기만 함 (복사하지 않음)\n", encoding="utf-8")
    print(f"완료: {out}")


def cmd_blend(args):
    """6단계: 이미 만든 결과(--src 폴더)에 SD 경계 블렌딩만 더한다. 결과는 --out/AI_블렌딩있음.
    블렌딩 없는 쪽은 --src 그대로라 비교는 compare --out <out> --prev <src의 단계 폴더>."""
    src = OUT_ROOT / args.src
    out = OUT_ROOT / args.out / "AI_블렌딩있음"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    rows = load_list()
    for i, row in enumerate(rows):
        sid = row["id"]
        f = next((src / f"{sid}{e}" for e in (".png", ".jpg") if (src / f"{sid}{e}").exists()), None)
        if f is None:
            continue
        img = read_bgr(f)
        m = np.asarray(Image.open(src / f"{sid}_mask.png").convert("L")) > 0
        ys, xs = np.nonzero(m)
        box = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        ring = ring_blend(img, box, seed=i)
        save_image(img, out / f"{sid}.png")
        save_mask(img.shape, [box, ring], out / f"{sid}_mask.png")
        log.append({"id": sid, "변경": f'{row["original_value"]} → {row["forged_value"]}', "블렌딩": "SD 테두리 14px, strength 0.35"})
        print(f"{i + 1}/{len(rows)} {sid}", flush=True)
    _write_log(out / "생성기록.csv", log)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select")
    sp.add_parser("baseline")
    bl = sp.add_parser("blend")
    bl.add_argument("--src", required=True)
    bl.add_argument("--out", required=True)
    e = sp.add_parser("export-win")
    e.add_argument("--no-sd", action="store_true")
    r = sp.add_parser("run")
    r.add_argument("--branch", choices=["ai", "reuse"], required=True)
    r.add_argument("--steps", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=["write", "copy"], default="write",
                   help="재사용 갈래에서 새 값을 만드는 방식: write(폰트로 쓰기), copy(문서 안 글자 복사)")
    r.add_argument("--no-persp", action="store_true", help="4단계에서 원근 맞추기(B2-2)를 끔, 회전만 (비교용)")
    r.add_argument("--no-tilt", action="store_true", help="4단계에서 기울기 맞추기(B2)를 끔 (비교용)")
    r.add_argument("--no-b3", action="store_true", help="B3 글자 질감(획 얼룩·잔결) 끔 (비교용)")
    r.add_argument("--no-b4", action="store_true", help="B4 지운 자리 그림자 없애기·주변 결 입히기 끔 (비교용)")
    r.add_argument("--no-b6", action="store_true", help="B6 휘어진 종이 따라 구부리기 끔 (비교용)")
    r.add_argument("--no-sd", action="store_true", help="AI 갈래에서 SD 대신 종이 메우기만 (SD 없는 환경에서 시험)")
    r.add_argument("--resume", action="store_true", help="이미 만든 결과는 건너뜀")
    r.add_argument("--name", default=None, help="결과 하위 폴더 이름 (기본: AI 또는 재사용)")
    c = sp.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--prev", required=True)
    args = ap.parse_args()
    if args.cmd == "compare":
        from forge_v2_compare import compare
        compare(OUT_ROOT, load_list(), args.out, args.prev)
    else:
        {"select": cmd_select, "baseline": cmd_baseline, "run": cmd_run,
         "export-win": cmd_export_win, "blend": cmd_blend}[args.cmd](args)


if __name__ == "__main__":
    main()
