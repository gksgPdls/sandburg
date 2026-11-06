"""
문서의 글자 칸 찾기 + 같은 종류 칸끼리 묶기 (A 검사 공통, 2026-10-30).

    from detector.common.cells import read_cells, group_cells
    cells = read_cells("문서.jpg")          # RapidOCR, [{id, text, kind, bbox:(x,y,w,h), conf}]
    groups = group_cells(cells)             # {group_id: [cell, ...]}  같은 열 금액 / 같은 줄 / 같은 모양 날짜

A 검사는 '원래 글씨체를 모르니, 같은 문서의 같은 종류 칸과 비교한다'가 기본이라 묶음이 중요하다.
묶는 기준 (OCR 글자 줄 상자 기준):
  - 금액 열: 오른쪽 끝이 맞고(±max(4px, 0.4×높이)) 높이가 비슷한(0.7~1.4배) 금액 칸들
  - 같은 줄: 세로 가운데가 맞는(±0.5×높이) 숫자 칸들
  - 날짜: 같은 형식(YYYY-MM-DD 등)의 날짜 칸 전부
  한 칸은 여러 묶음에 들어갈 수 있다. 칸이 2개 미만인 묶음은 버린다.
"""
from pathlib import Path

from PIL import Image

from .ocr import rapid_candidates


def read_cells(image):
    img = image if isinstance(image, Image.Image) else Image.open(image)
    cs = rapid_candidates(img.convert("RGB"), only="all")
    return [{"id": i, **c, "bbox": tuple(int(v) for v in c["bbox"])} for i, c in enumerate(cs)]


def _money(c):
    return c["kind"] == "money"


def group_cells(cells, min_size=2):
    groups = {}
    num = [c for c in cells if c["kind"] in ("money", "number")]
    money = [c for c in cells if _money(c)]
    # 금액 열
    used = set()
    for c in sorted(money, key=lambda c: c["bbox"][0] + c["bbox"][2]):
        if c["id"] in used:
            continue
        x, y, w, h = c["bbox"]
        r = x + w
        col = [d for d in money if abs(d["bbox"][0] + d["bbox"][2] - r) <= max(4, 0.4 * h)
               and 0.7 <= d["bbox"][3] / max(h, 1) <= 1.4]
        if len(col) >= min_size:
            groups[f"col_{len(groups)}"] = col
            used.update(d["id"] for d in col)
    # 같은 줄 숫자 칸
    used = set()
    for c in sorted(num, key=lambda c: c["bbox"][1]):
        if c["id"] in used:
            continue
        x, y, w, h = c["bbox"]
        cy = y + h / 2
        row = [d for d in num if abs(d["bbox"][1] + d["bbox"][3] / 2 - cy) <= 0.5 * h
               and 0.7 <= d["bbox"][3] / max(h, 1) <= 1.4]
        if len(row) >= min_size:
            groups[f"row_{len(groups)}"] = row
            used.update(d["id"] for d in row)
    # 날짜
    dates = [c for c in cells if c["kind"] == "date"]
    if len(dates) >= min_size:
        groups["dates"] = dates
    return groups
