"""진료비 영수증 표를 로컬에서 읽는다. 다른 스크립트가 가져다 쓴다.

1. 괘선(가로·세로 긴 선)을 찾아 칸으로 나눈다
2. 칸 안쪽을 잘라 세로로 이어 붙이고 tesseract 를 한 번만 돌려 칸별 글자를 얻는다
   (칸마다 따로 돌리면 한 장에 수십 초가 걸린다)

모든 처리는 로컬이다. 결과 글자는 코드 안에서만 쓰고 출력하지 않는다.
"""

import csv
import io
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

AMOUNT = re.compile(r"^\d{1,3}(,\d{3})*$")
LOOSE_AMOUNT = re.compile(r"^\d{1,3}([.,]\d{3})+$")  # 쉼표를 마침표로 읽는 경우가 많다
SCALE = 3      # OCR 전에 키우는 배율. 글자 높이 12px 전후라 그대로는 잘 못 읽는다
GAP = 24       # 이어 붙일 때 칸 사이 여백(px, 확대 후)


def detect_cells(gray, max_w=300):
    """표의 칸 목록 [(x, y, w, h)]. 칸은 괘선 안쪽 영역이다.
    max_w: 이보다 넓은 칸은 뺀다. 금액 칸은 300 이면 되고, 진료기간처럼 넓은 칸은 더 크게 준다."""
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (W // 25, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 60)))
    lines = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(255 - lines, connectivity=4)
    return [tuple(int(v) for v in s[:4]) for s in stats[1:]
            if 12 <= s[3] <= 45 and 25 <= s[2] <= max_w and s[4] > 0.8 * s[2] * s[3]]


def ocr_cells(gray, cells, lang="eng", whitelist="0123456789,.-"):
    """칸마다 (읽은 글자, 최저 신뢰도 0-100) 목록 (cells 와 같은 순서)."""
    if not cells:
        return []
    crops = []
    for x, y, w, h in cells:
        c = gray[y + 1:y + h - 1, x + 1:x + w - 1]
        c = cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        # 회색 종이 배경 그대로는 tesseract 가 줄을 못 찾는다. 칸마다 흑백으로 바꾸고,
        # 대비가 거의 없는 빈 칸은 흰색으로 둔다 (Otsu 가 종이 결을 글자로 만든다)
        if c.std() > 12:
            c = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            c = np.full_like(c, 255)
        crops.append(c)
    width = max(c.shape[1] for c in crops) + 40
    total = sum(c.shape[0] + GAP for c in crops) + GAP
    sheet = np.full((total, width), 255, np.uint8)
    slots, top = [], GAP
    for c in crops:
        sheet[top:top + c.shape[0], 20:20 + c.shape[1]] = c
        slots.append((top, top + c.shape[0]))
        top += c.shape[0] + GAP

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, sheet)
        cmd = ["tesseract", path, "-", "-l", lang, "--psm", "6"]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        out = subprocess.run(cmd + ["tsv"], capture_output=True).stdout.decode("utf-8", "replace")
    finally:
        os.remove(path)

    words = [[] for _ in cells]
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (r.get("text") or "").strip()
        if not text or float(r.get("conf") or -1) < 0:
            continue
        cy = int(r["top"]) + int(r["height"]) / 2
        for i, (a, b) in enumerate(slots):
            if a - GAP / 2 <= cy <= b + GAP / 2:
                words[i].append((int(r["left"]), text, float(r["conf"])))
                break
    return [("".join(t for _, t, _ in sorted(ws)), min((c for _, _, c in ws), default=0.0))
            for ws in words]


def upright(gray):
    """가로로 찍힌 사진이면 세로로 돌린다. (돌린 이미지, 돌린 횟수 k) — np.rot90 기준."""
    if gray.shape[1] > gray.shape[0]:
        return np.rot90(gray, -1).copy(), -1
    return gray, 0


def read_table(gray, max_w=300):
    """칸 목록. 각 칸은 dict(x, y, w, h, text, conf, amount).
    amount 는 금액이면 정수, 아니면 None. conf 는 tesseract 신뢰도(칸 안 단어 중 최저)."""
    cells = detect_cells(gray, max_w)
    texts = ocr_cells(gray, cells)
    table = []
    for (x, y, w, h), (t, conf) in zip(cells, texts):
        if LOOSE_AMOUNT.match(t):
            t = t.replace(".", ",")
        amount = int(t.replace(",", "")) if AMOUNT.match(t) else None
        table.append({"x": x, "y": y, "w": w, "h": h, "text": t, "conf": conf, "amount": amount})
    return table


DATE = re.compile(r"(20\d{2})([.\-]?)(0[1-9]|1[0-2])([.\-]?)(0[1-9]|[12]\d|3[01])")


def find_dates(text):
    """칸 글자에서 날짜 목록. 각 날짜는 dict(y, m, d, day_span) — day_span 은 text 안 일(日) 두 자리 위치."""
    out = []
    for m in DATE.finditer(text):
        a, b = m.span()
        # 구분 기호가 없으면 앞뒤가 숫자가 아닌 정확히 8자리일 때만 날짜로 확신한다
        standalone = (a == 0 or not text[a - 1].isdigit()) and (b == len(text) or not text[b].isdigit())
        out.append({"y": int(m.group(1)), "m": int(m.group(3)), "d": int(m.group(5)),
                    "day_span": m.span(5),
                    "separated": bool(m.group(2) and m.group(4)) or standalone})
    return out


def char_boxes(gray, box, whitelist="0123456789.-"):
    """칸 하나를 한 줄로 읽어 (글자, 글자별 외곽 목록)을 돌려준다. 외곽은 원본 좌표.
    tesseract makebox 를 쓴다. 획 덩어리로 나누면 붙은 숫자가 뭉쳐 글자 수가 안 맞는다."""
    x, y, w, h = box
    c = gray[y + 1:y + h - 1, x + 1:x + w - 1]
    c = cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    c = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    pad = 20
    c = cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, c)
        out = subprocess.run(["tesseract", path, "-", "-l", "eng", "--psm", "7",
                              "-c", f"tessedit_char_whitelist={whitelist}", "makebox"],
                             capture_output=True).stdout.decode("utf-8", "replace")
    finally:
        os.remove(path)
    H = c.shape[0]
    text, boxes = "", []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or len(parts[0]) != 1:
            continue
        ch, x1, y1, x2, y2 = parts[0], *map(int, parts[1:5])
        # makebox 좌표는 왼쪽 아래가 원점이다
        bx0, bx1 = (x1 - pad) / SCALE, (x2 - pad) / SCALE
        by0, by1 = (H - y2 - pad) / SCALE, (H - y1 - pad) / SCALE
        text += ch
        boxes.append((x + 1 + int(bx0), y + 1 + int(by0),
                      max(1, int(round(bx1 - bx0))), max(1, int(round(by1 - by0)))))
    return text, boxes
