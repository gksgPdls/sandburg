"""같은 문서 안의 숫자·기호 글자 모음 (copy 방식용).

페이지 전체를 OCR(tesseract)로 읽어 숫자가 든 단어를 찾고, 단어마다 글자 단위 위치를 잡는다
(동현님 receipt.char_boxes 사용). 각 글자는 (문자, 실제 획 외곽, 단어의 숫자 윗선 기준 세로 위치, 숫자 높이)로 저장.
"""
import csv
import io
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

import edit_ops
import receipt

WORD_RE = re.compile(r"^[0-9][0-9,.\-]*[0-9]$")


def _page_words(gray):
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, big)
        out = subprocess.run(["tesseract", path, "-", "-l", "eng", "--psm", "11",
                              "-c", "tessedit_char_whitelist=0123456789,.-", "tsv"],
                             capture_output=True).stdout.decode("utf-8", "replace")
    finally:
        os.remove(path)
    words = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        if not WORD_RE.match(t) or sum(c.isdigit() for c in t) < 2 or float(r.get("conf") or 0) < 70:
            continue
        x, y, w, h = (int(r[k]) // 2 for k in ("left", "top", "width", "height"))
        words.append((t, (x - 2, y - 3, w + 4, h + 6)))
    return words


def _overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def build(gray, exclude=None):
    """문서 글자 모음. exclude(바꿀 칸)와 겹치는 단어는 뺀다."""
    H, W = gray.shape
    lib = []
    for text, box in _page_words(gray):
        x, y, w, h = box
        box = (max(x, 0), max(y, 0), min(w, W - max(x, 0)), min(h, H - max(y, 0)))
        if exclude and _overlap(box, exclude):
            continue
        got, glyphs = receipt.char_boxes(gray, box, whitelist="0123456789,.-")
        if got != text or len(glyphs) != len(text):
            continue
        ext = edit_ops.digit_extent(gray, box)
        if ext is None:
            continue
        top, dh = ext
        for ch, g in zip(text, glyphs):
            ink = edit_ops.glyph_ink(gray, box, [g])
            if not ink or ink[2] < 1 or ink[3] < 1:
                continue
            lib.append({"ch": ch, "ink": ink, "dy": ink[1] - top, "digit_h": dh, "word": box})
    return lib


def compose(img, src_img, lib, text, right, top, height, gap, ink, near, use_count, tol=0.12):
    """text를 오른쪽 끝 right, 숫자 윗선 top에 맞춰 배치한다.
    문서에 같은 글자(크기 차이 tol 이내)가 있으면 src_img에서 픽셀 블록째 복사해 붙이고(동현님 copy_block 방식),
    없으면 그 글자만 폰트로 쓴다(edit_ops.write). 붙인 영역 목록과 (복사 수, 폰트 수)를 돌려준다."""
    placed, n_copy, n_font = [], 0, 0
    x_right = right
    for ch in reversed(text):
        cands = [g for g in lib if g["ch"] == ch and abs(g["digit_h"] - height) <= tol * height]
        if cands:
            # 덜 쓴 글자 우선, 그다음 가까운 것 (같은 글자를 한 곳에서만 계속 가져오지 않게)
            g = min(cands, key=lambda g: (use_count.get(id(g), 0),
                                          abs(g["ink"][0] - near[0]) + abs(g["ink"][1] - near[1])))
            use_count[id(g)] = use_count.get(id(g), 0) + 1
            sx, sy, sw, sh = g["ink"]
            m = 1  # 획 가장자리 번짐까지
            block = src_img[sy - m:sy + sh + m, sx - m:sx + sw + m].copy()
            px, py = x_right - sw + 1 - m, top + g["dy"] - m
            img[py:py + block.shape[0], px:px + block.shape[1]] = block
            placed.append((px, py, block.shape[1], block.shape[0]))
            left = px + m
            n_copy += 1
        else:
            b = edit_ops.write(img, ch, right=x_right, top=top, digit_height=height, ink=ink)
            placed.append(b)
            left = b[0]
            n_font += 1
        x_right = left - gap - 1
    return placed, n_copy, n_font


def value_gap(gray, box, height):
    """원래 값의 글자 사이 간격(중앙값). 못 재면 숫자 높이의 12%."""
    text, glyphs = receipt.char_boxes(gray, box, whitelist="0123456789,.-")
    inks = [edit_ops.glyph_ink(gray, box, [g]) for g in glyphs]
    inks = sorted([i for i in inks if i], key=lambda b: b[0])
    gaps = [b[0] - (a[0] + a[2]) for a, b in zip(inks, inks[1:]) if b[0] - (a[0] + a[2]) >= 0]
    return int(np.median(gaps)) if gaps else max(1, int(round(height * 0.12)))
