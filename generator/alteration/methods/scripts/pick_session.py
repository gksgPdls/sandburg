#!/usr/bin/env python3
"""반자동 조작: 사람이 사진마다 위치를 마우스로 찍고, 스크립트가 조작·마스크·CSV를 만든다.

두 단계다.
    pick   사진을 한 장씩 띄운다. 단계마다 영역을 드래그하고 ENTER. 좌표는 한 장마다
           _work/picks/{task}.json 에 바로 저장되므로 중간에 멈춰도(Ctrl+C) 남는다
    apply  저장된 좌표로 조작본·마스크를 만들어 samples/ 에 등록한다.
           조작 방식을 고쳤을 때 다시 찍지 않고 이것만 다시 돌리면 된다

사용:
    pick_session.py <작업> pick 10   모두 10장이 될 때까지 찍기 (끝나면 자동으로 apply)
    pick_session.py <작업> apply     찍어 둔 좌표로 다시 만들기

찍는 법 (창 위쪽 안내 띠에도 나온다):
    드래그 → ENTER(또는 SPACE)    확정
    c                              이 사진 건너뛰기 (첫 단계에서)
    결과 화면: ENTER 저장 / r 다시 찍기 / c 건너뛰기 / q 그만하기
    Ctrl+C (터미널에서)            그만하기. 찍은 것까지는 저장돼 있다
"""

import argparse
import csv
import os
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labels import ROOT, load_all
from register_sample import register

PICKS = ROOT / "_work" / "picks"
POOL = ROOT / "_work" / "pool"
TMP = ROOT / "_work" / "tmp"
SCREEN_H = 820  # 사진 높이. 화면이 작으면 줄인다
BANNER_H = 64   # 창 위쪽 안내 띠 높이
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


# ---------- 공통 ----------

def manifest():
    with open(POOL / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        return {r["file"]: r["pool_id"] for r in csv.DictReader(f)}


def candidates(pools, seed, exclude):
    """아직 안 쓴 원본들 (무작위 순서)."""
    ids = manifest()
    used = {r.get("source", "") for r in load_all()} | exclude
    files = [f for p in pools for f in sorted((POOL / p).iterdir())
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    random.Random(seed).shuffle(files)
    return [(f, ids.get(f.relative_to(POOL).as_posix(), "")) for f in files
            if ids.get(f.relative_to(POOL).as_posix(), "") not in used]


def _fit(img):
    scale = min(1.0, SCREEN_H / img.shape[0])
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    return view, scale


def _with_banner(view, lines):
    """사진 위에 한글 안내 띠를 붙인다. OpenCV 는 한글을 못 그려서 Pillow 로 그린다."""
    w = max(view.shape[1], 560)
    banner = Image.new("RGB", (w, BANNER_H), (40, 40, 40))
    d = ImageDraw.Draw(banner)
    font = ImageFont.truetype(FONT, 20)
    for i, line in enumerate(lines[:2]):
        d.text((12, 6 + i * 28), line, font=font, fill=(255, 230, 120) if i == 0 else (230, 230, 230))
    banner = cv2.cvtColor(np.asarray(banner), cv2.COLOR_RGB2BGR)
    if view.shape[1] < w:
        view = cv2.copyMakeBorder(view, 0, 0, 0, w - view.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return cv2.vconcat([banner, view])


WINDOW = "pick"


def select_box(img, lines, highlight=None):
    """안내 띠(lines)와 함께 img 를 띄우고 드래그한 사각형을 원본 좌표로 돌려준다.
    highlight: 앞 단계에서 고른 사각형(원본 좌표). 초록 네모로 보여준다. 건너뛰면 None."""
    view, scale = _fit(img)
    if highlight:
        x, y, w, h = [int(v * scale) for v in highlight]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
    x, y, w, h = cv2.selectROI(WINDOW, _with_banner(view, lines), showCrosshair=False)
    y -= BANNER_H
    if w == 0 or h == 0 or y < 0:
        return None
    s = 1 / scale
    return [int(x * s), int(y * s), int(w * s), int(h * s)]


def confirm(img, lines, variants=False):
    """결과를 보여주고 키를 받는다. 'save' / 'redo' / 'skip'. q 는 그만하기.
    variants=True 면 t(다른 글), f(표기 형식)도 받는다."""
    view, _ = _fit(img)
    cv2.imshow(WINDOW, _with_banner(view, lines))
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (13, 10, 32):
            return "save"
        if k in (ord("r"), ord("R")):
            return "redo"
        if k in (ord("c"), ord("C"), 27):
            return "skip"
        if k in (ord("q"), ord("Q")):
            raise KeyboardInterrupt
        if variants and k in (ord("t"), ord("T")):
            return "next_text"
        if variants and k in (ord("f"), ord("F")):
            return "next_format"


def changed_mask(before, after):
    """바뀐 픽셀 덩어리마다 외곽 사각형을 칠한 마스크."""
    diff = (np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(cv2.dilate(diff, np.ones((15, 15), np.uint8)))
    mask = np.zeros(diff.shape, np.uint8)
    for i in range(1, n):
        ys, xs = np.nonzero((labels == i) & (diff > 0))
        mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
    return mask


# ---------- 여백에 진단 추가 (글리프) ----------

from edit_ops import ink_level, ink_mask  # noqa: E402
from tamper_add_text import render_alpha  # noqa: E402

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
DIAG_FORMATS = ["({code}) {name}", "{code} {name}", "{name} ({code})", "{name}"]


def kcd_list():
    with open(ROOT / "kcd_add_list.csv", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["code"].strip()]


def diag_text(entry):
    rows = kcd_list()
    r = rows[entry.get("text_i", 0) % len(rows)]
    return DIAG_FORMATS[entry.get("fmt_i", 0) % len(DIAG_FORMATS)].format(code=r["code"], name=r["name"]), r


def measure_line(gray, box):
    """기준 줄의 글자 높이(획이 많은 행), 진하기, 기울기(반시계 +)."""
    x, y, w, h = box
    m = ink_mask(gray[y:y + h, x:x + w]) > 0
    if m.sum() < 20:
        return None
    rows = m.sum(axis=1)
    core = np.nonzero(rows >= 0.25 * rows.max())[0]
    height = int(core.max() - core.min()) + 1
    ys, xs = np.nonzero(m)
    vx, vy = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)[:2].ravel()
    angle = float(np.clip(-np.degrees(np.arctan2(vy, vx)), -3, 3))
    if abs(angle) > 90:
        angle = 0.0
    ink, paper = ink_level(gray, box)
    return {"height": height, "ink": ink, "paper": paper, "angle": angle}


def _size_for_height(height):
    """한글 글자 높이가 height px 이 되는 글자 크기."""
    best = None
    for s in np.arange(6, 60, 0.5):
        a, _ = render_alpha("가", FONT, 0, float(s), 0)
        rows = np.nonzero((a > 0.3).any(axis=1))[0]
        err = abs((rows.max() - rows.min() + 1) - height)
        if best is None or err < best[1]:
            best = (float(s), err)
    return best[0]


def apply_add_diag(img, pick):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = measure_line(gray, pick["ref"])
    if not ref:
        return None
    text, row = diag_text(pick)
    size = _size_for_height(ref["height"])
    alpha, (ox, oy) = render_alpha(text, FONT, 0, size, ref["angle"])
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.4, ref["height"] / 30))
    tx, ty = pick["target"][:2]
    x0, y0 = tx - ox, ty - oy
    h, w = alpha.shape
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        return None

    region = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    paper = float(np.median(region))
    ys, xs = np.nonzero(alpha > 0.1)
    text_box = (x0 + int(xs.min()), y0 + int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    # 진하기: 몇 가지로 써 보고 기준 줄 진하기에 가장 가까운 것.
    # 번짐 때문에 획 가운데도 완전히 덮이지 않아, 아주 진한 문서는 잉크 값만으로는 못 따라간다.
    # 그래서 획 농도(gain)도 키워 본다
    best = None
    for gain in (1.0, 1.4, 1.8):
        a = np.clip(alpha * gain, 0, 1)
        for offset in (0, 20, 40, 60, 80):
            k = min(1.0, max(0.0, ref["ink"] - offset) / paper) if paper > 0 else 0.0
            out = np.clip(region * (1 - a[..., None] * (1 - k)), 0, 255).astype(np.uint8)
            trial = img.copy()
            trial[y0:y0 + h, x0:x0 + w] = out
            got, _ = ink_level(cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY), text_box)
            if best is None or abs(got - ref["ink"]) < abs(best[0] - ref["ink"]) - 1:
                best = (got, out)
    img[y0:y0 + h, x0:x0 + w] = best[1]
    rx, ry, rw, rh = pick["ref"]
    return {
        "description": f"병명 칸에 '{text}' 한 줄 추가 (x={tx}, y={ty}). 지우지 않고 여백에 씀. "
                       f"기준 줄 (x={rx}, y={ry}, {rw}x{rh})에 맞춤: 글자 높이 {ref['height']}px, "
                       f"진하기 {ref['ink']:.0f}→{best[0]:.0f}, 기울기 {ref['angle']:+.1f}도. 폰트는 맥 기본(Apple SD Gothic Neo)",
        "detection": "글리프: 주력. 추가한 줄의 폰트·획 굵기가 기준 줄과 다름. 복사가 아니라 copy-move 는 못 잡음. "
                     f"코드({row['code']})와 병명이 맞는 진단을 골라 콘텐츠 로직은 못 잡음 (KCD 표기 확인 필요). "
                     "치료 내용·진료비와 맞지 않으면 문서 간 대조로 잡을 여지가 있음",
        "notes": f"ref={rx},{ry},{rw},{rh} target={tx},{ty} text_i={pick.get('text_i', 0)} fmt_i={pick.get('fmt_i', 0)}",
    }


# ---------- 칸 글자 바꾸기 공통 (KCD 코드, 금액·합계) ----------

from edit_ops import digit_extent, erase, ink_bbox, write  # noqa: E402
from receipt import char_boxes  # noqa: E402


def read_box(img, box, whitelist):
    """찍은 네모 안 글자를 로컬 OCR 로 읽는다 (화면 안내용, 설명용)."""
    x, y, w, h = box
    cell = (x - 2, y - 2, w + 4, h + 4)
    return char_boxes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cell, whitelist)[0]


def rewrite(img, box, new_text, seed=0, grow=0.6):
    """찍은 네모 안 글자를 지우고 new_text 를 오른쪽 끝·높이·진하기에 맞춰 쓴다.
    새 글이 더 길 수 있어 왼쪽으로 grow 배만큼 넘어가는 것은 허용한다. 실패하면 None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ib = ink_bbox(gray, box)
    ext = digit_extent(gray, box)
    if ib is None or ext is None:
        return None
    ink, _ = ink_level(gray, box)
    x, y, w, h = box
    limit = (x - int(w * grow), y - 3, w + int(w * grow), h + 6)
    erased_img = img.copy()
    if erase(erased_img, box, seed) is None:
        return None
    best = None
    for offset in (0, 20, 40, 60, 80):
        trial = erased_img.copy()
        wb = write(trial, new_text, right=ib[0] + ib[2] - 1, top=ext[0], digit_height=ext[1],
                   ink=max(0.0, ink - offset))
        if wb[0] < limit[0] or wb[1] < limit[1] or wb[0] + wb[2] > limit[0] + limit[2] \
                or wb[1] + wb[3] > limit[1] + limit[3]:
            return None
        got, _ = ink_level(cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY), wb)
        if best is None or abs(got - ink) < abs(best[0] - ink):
            best = (got, trial)
    img[:] = best[1]
    return True


# ---------- KCD 코드 교체 (콘텐츠 로직, 케이스 4) ----------

CODE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."


def kcd_swap_list():
    with open(ROOT / "kcd_swap_list.csv", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["code"].strip()]


def kcd_choice(entry, old):
    """old 와 다른 장(첫 글자)의 코드 중 entry['text_i'] 번째."""
    rows = [r for r in kcd_swap_list() if not old or r["code"][0] != old[0]]
    return rows[entry.get("text_i", 0) % len(rows)]


def kcd_label(img, entry):
    old = read_box(img, entry["code"], CODE_CHARS)
    new = kcd_choice(entry, old)
    return f"코드 {old or '(못 읽음)'} → {new['code']} ({new['meaning']}).  t 로 다른 코드"


def apply_kcd_swap(img, pick):
    old = read_box(img, pick["code"], CODE_CHARS)
    new = kcd_choice(pick, old)
    if not rewrite(img, pick["code"], new["code"], seed=pick.get("text_i", 0)):
        return None
    x, y, w, h = pick["code"]
    return {
        "description": f"진단서 질병 코드 {old or '(OCR 실패)'} → {new['code']} (x={x}, y={y}). 병명은 그대로. "
                       f"코드만 배경으로 지우고 다시 씀. 새 코드는 '{new['meaning']}' ({new['why']})",
        "detection": "콘텐츠 로직: 병명과 KCD 코드가 맞지 않음. 공개 마스터(KCD 코드표) 대조로 결정적 판정 가능 "
                     "(방법론 케이스 4). 글리프: 다시 쓴 코드의 폰트가 다름",
        "notes": f"code={x},{y},{w},{h} text_i={pick.get('text_i', 0)} old_ocr={old}",
    }


# ---------- 금액 + 합계 같이 바꾸기 (글리프, 케이스 2) ----------

def _amount(text):
    t = text.replace(".", ",")
    return int(t.replace(",", "")) if t.replace(",", "").isdigit() and t else None


def sum_fix_values(img, entry):
    item = _amount(read_box(img, entry["item"], "0123456789,."))
    total = _amount(read_box(img, entry["total"], "0123456789,."))
    if not item or not total or total < item:
        return item, total, None, None
    rng = random.Random(entry.get("text_i", 0))
    digits = len(str(item))
    for _ in range(50):
        new = item + rng.choice([-1, 1]) * rng.randint(1, 9) * 10 ** max(digits - 2, 2)
        if new > 0 and new != item and len(str(new)) == digits:
            new_total = total + new - item
            if len(str(new_total)) == len(str(total)):
                return item, total, new, new_total
    return item, total, None, None


def sum_fix_label(img, entry):
    item, total, new, new_total = sum_fix_values(img, entry)
    if new is None:
        return f"읽은 값: 금액 {item}, 합계 {total} — 못 바꾼다. r 로 다시 찍기"
    return f"금액 {item:,} → {new:,},  합계 {total:,} → {new_total:,}.  t 로 다른 금액"


def apply_sum_fix(img, pick):
    item, total, new, new_total = sum_fix_values(img, pick)
    if new is None:
        return None
    s = pick.get("text_i", 0)
    before = img.copy()
    if not rewrite(img, pick["item"], f"{new:,}", seed=s, grow=0.2) or \
            not rewrite(img, pick["total"], f"{new_total:,}", seed=s + 1, grow=0.2):
        img[:] = before
        return None
    (ix, iy, _, _), (tx, ty, _, _) = pick["item"], pick["total"]
    return {
        "description": f"금액 {item:,} → {new:,} (x={ix}, y={iy}), 합계 {total:,} → {new_total:,} (x={tx}, y={ty}). "
                       f"둘 다 배경으로 지우고 다시 씀. 사람이 금액 칸과 그 합계 칸을 지정",
        "detection": "글리프: 다시 쓴 두 곳의 폰트가 주변 숫자와 다름. 이 합계는 맞아서 콘텐츠 로직은 여기서 못 잡음 "
                     "(진료비 총액 등 이 합계를 다시 더하는 칸과는 어긋날 수 있음)",
        "notes": f"item={','.join(map(str, pick['item']))} total={','.join(map(str, pick['total']))} text_i={s}",
    }


# ---------- 작업 목록 ----------

TASKS = {
    # 작업 정의: method(폴더), axis(주 탐지 축), difficulty, pools(원본 풀),
    # steps[(키, [안내 1줄, 안내 2줄])], apply(img, pick) → 설명 dict 또는 None,
    # flow="two_docs" 면 문서 A 에서 고르고 다른 문서 B 에 적용한다
    # 도장 복제·이식은 보험 사기 동기가 없어 쓰지 않는다 (2026-09-30 사용자 판단)
    "add_diag": {
        "method": "add_diag", "axis": "glyph", "difficulty": "중간",
        "pools": ["DOC203"], "variants": True,
        "steps": [
            ("ref", ["1단계: 기존 병명 한 줄을 네모로 딱 맞게 감싸고 Enter",
                     "새 줄의 크기·진하기·기울기를 이 줄에 맞춘다. 진단서가 아니거나 병명 칸에 여백이 없으면 c"]),
            ("target", ["2단계: 새 병명 줄이 시작될 곳을 드래그하고 Enter",
                        "드래그한 네모의 왼쪽 위 = 새 줄의 첫 글자 위치. 보통 기존 줄 바로 아래, 왼쪽 끝 맞춤"]),
        ],
        "apply": apply_add_diag,
        "label": lambda img, e: f"추가: {diag_text(e)[0]}.  t 다른 병명 / f 표기 형식",
    },
    "kcd_swap": {
        "method": "kcd_swap", "axis": "content_logic", "difficulty": "쉬움",
        "pools": ["DOC203"], "variants": True,
        "steps": [
            ("code", ["질병 코드만 딱 맞게 감싸고 Enter (괄호는 빼고, 예: D18.0)",
                      "코드가 없거나 손글씨인 진단서면 c"]),
        ],
        "apply": apply_kcd_swap,
        "label": kcd_label,
    },
    "sum_fix": {
        "method": "amount_sum_fixed", "axis": "glyph", "difficulty": "중간",
        "pools": ["DOC210"], "variants": True,
        "steps": [
            ("item", ["1단계: 바꿀 금액 숫자를 감싸고 Enter (괘선은 빼고 칸 안쪽만)",
                      "합계 줄이 있는 영수증이 아니면 c"]),
            ("total", ["2단계: 그 금액이 들어가는 합계 숫자를 감싸고 Enter",
                       "보통 같은 열 아래쪽 '합계' 줄. 초록 네모 = 1단계 금액"]),
        ],
        "apply": apply_sum_fix,
        "label": sum_fix_label,
    },
}

DOC_TYPE = {"DOC210": "진료비 영수증", "DOC205": "진료비 영수증 (퇴원·세부내역 계열)",
            "DOC203": "진단서·소견서"}


# ---------- 실행 ----------

def load_picks(task):
    path = PICKS / f"{task}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_picks(task, picks):
    PICKS.mkdir(parents=True, exist_ok=True)
    (PICKS / f"{task}.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")


def pick(task, count, seed):
    """찍은 사진이 모두 count 장이 될 때까지 찍는다."""
    spec = TASKS[task]
    picks = load_picks(task)
    need = count - len(picks)
    print(f"\n[{task}] 목표 {count}장, 이미 찍은 것 {len(picks)}장 → {max(need, 0)}장 더 찍는다")
    print("  - 창 위쪽 안내 띠를 따라 하면 된다. 그만하려면 터미널에서 Ctrl+C (찍은 것까지 저장)\n")
    if need <= 0:
        return 0

    got = 0
    done = {p["pool_id"] for p in picks}
    cands = candidates(spec["pools"], seed, done)
    if spec.get("flow") == "two_docs":
        return _pick_two_docs(task, spec, picks, cands, need)
    for f, pool_id in cands:
        if got == need:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        rng = random.Random(f"{seed}-{pool_id}")
        while True:
            entry, prev = {"pool_id": pool_id, "file": f.relative_to(POOL).as_posix()}, None
            if spec.get("variants"):
                entry.update(text_i=rng.randrange(1000), fmt_i=0)
            for key, lines in spec["steps"]:
                b = select_box(img, [f"[{pool_id}]  " + lines[0], lines[1]], highlight=prev)
                if b is None:
                    entry = None
                    break
                entry[key] = prev = b
            if entry is None:
                action = "skip"
                break
            while True:  # t·f 로 글·형식을 바꾸면 다시 그려서 보여준다
                preview = img.copy()
                if spec["apply"](preview, entry) is None:
                    if spec.get("label"):
                        print(f"  {pool_id}: {spec['label'](img, entry)}")
                    action = "bad"
                    break
                if spec.get("variants"):
                    head = "결과 확인: Enter 저장 / t·f 바꾸기 / r 다시 찍기 / c 건너뛰기 / q 그만"
                    sub = f"{spec['label'](img, entry)}    ({got}/{need}장 저장됨)"
                else:
                    head = "결과 확인: Enter 저장  /  r 다시 찍기  /  c 건너뛰기  /  q 그만하기"
                    sub = f"{got}/{need}장 저장됨. 그만해도 찍은 것까지는 저장된다"
                action = confirm(preview, [head, sub], variants=spec.get("variants", False))
                if action == "next_text":
                    entry["text_i"] += 1
                elif action == "next_format":
                    entry["fmt_i"] += 1
                else:
                    break
            if action == "bad":
                print(f"  {pool_id}: 이 위치로는 만들 수 없다 (사진 밖이거나 기준 줄에 글자가 없음). 다시 찍는다")
                continue
            if action != "redo":
                break
        if action == "skip":
            print(f"  {pool_id}: 건너뜀")
            continue
        picks.append(entry)
        save_picks(task, picks)
        got += 1
        print(f"  {pool_id}: 저장 ({got}/{need})")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return got


def _pick_two_docs(task, spec, picks, cands, need):
    """문서 A 에서 고르고 다른 문서 B 에 찍는 흐름. B 는 A 와 다른 사진을 차례로 보여준다."""
    (k1, l1), (k2, l2) = spec["steps"]
    got, used_b = 0, set()
    donors = list(cands)
    receivers = list(reversed(cands))  # A 와 겹치지 않게 반대쪽 끝부터 B 를 꺼낸다
    for fa, id_a in donors:
        if got == need:
            break
        if id_a in used_b:
            continue
        img_a = cv2.imread(str(fa))
        first = select_box(img_a, [f"[{id_a}]  " + l1[0], l1[1]])
        if first is None:
            print(f"  {id_a}: 문서 A 건너뜀")
            continue
        placed = False
        while receivers and not placed:
            fb, id_b = receivers.pop(0)
            if id_b == id_a or id_b in used_b:
                continue
            img_b = cv2.imread(str(fb))
            while True:
                target = select_box(img_b, [f"[{id_b}]  " + l2[0], l2[1]])
                if target is None:
                    action = "next_b"
                    break
                entry = {"pool_id": id_b, "file": fb.relative_to(POOL).as_posix(),
                         "donor_id": id_a, "donor_file": fa.relative_to(POOL).as_posix(),
                         k1: first, k2: target}
                preview = img_b.copy()
                if spec["apply"](preview, entry) is None:
                    print(f"  {id_b}: 이 자리에는 찍을 수 없다 (사진 밖이거나 너무 작음). 다시 드래그")
                    continue
                action = confirm(preview, ["결과 확인: Enter 저장  /  r 다시 찍기  /  c 다른 문서 B  /  q 그만하기",
                                           f"문서 A {id_a} → 이 문서 {id_b}.  {got}/{need}장 저장됨"])
                if action != "redo":
                    break
            if action in ("next_b", "skip"):
                continue
            picks.append(entry)
            save_picks(task, picks)
            used_b.add(id_b)
            got += 1
            placed = True
            print(f"  {id_a} → {id_b}: 저장 ({got}/{need})")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return got


def apply(task, author):
    spec = TASKS[task]
    picks = load_picks(task)
    TMP.mkdir(parents=True, exist_ok=True)
    made = []
    for p in picks:
        if p.get("applied"):
            continue
        f = POOL / p["file"]
        img = cv2.imread(str(f))
        before = img.copy()
        info = spec["apply"](img, p)
        if not info:
            print(f"  {p['pool_id']}: 만들 수 없음 (붙일 위치가 사진 밖 등)")
            continue
        t_path, m_path = TMP / f"t_{os.getpid()}.jpg", TMP / f"m_{os.getpid()}.png"  # 동시 실행 대비
        cv2.imwrite(str(t_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(m_path), changed_mask(before, img))
        sid = register(method=spec["method"], axis=spec["axis"], original=f, tampered=t_path,
                       mask=m_path, description=info["description"], detection=info["detection"],
                       doc_type=DOC_TYPE.get(f.parent.name, f.parent.name),
                       difficulty=spec["difficulty"], external_ai="N", author=author,
                       source=p["pool_id"], notes=f"pick_session.py {task}. {info['notes']}")
        t_path.unlink()
        m_path.unlink()
        p["applied"] = sid
        made.append(sid)
    save_picks(task, picks)
    print(f"[{task}] {len(made)}개 생성 → samples/{spec['method']}/  {' '.join(made)}")


def main():
    ap = argparse.ArgumentParser(description="반자동 조작 (위치를 마우스로 찍기)")
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("mode", choices=["pick", "apply"])
    ap.add_argument("count", type=int, nargs="?", default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--author", default="김동현")
    a = ap.parse_args()
    if a.mode == "pick":
        try:
            pick(a.task, a.count, a.seed)
        except KeyboardInterrupt:
            print("\n중단. 찍은 것까지는 저장돼 있다")
            cv2.destroyAllWindows()
    apply(a.task, a.author)


if __name__ == "__main__":
    sys.exit(main())
