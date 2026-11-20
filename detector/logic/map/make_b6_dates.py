"""B6(공휴일 가산 ↔ 진료일) 예시용 변조본: 토요일·일요일·공휴일 진료 영수증의 진료일을 가까운 앞 평일로 옮긴다.
공휴일(토요일 오후 공휴 가산 포함) 표시는 그대로 남으므로 '공휴일 표시인데 평일'이 된다.
날짜 고치기는 팀 생성기 dataset/scripts/auto_receipt.py 의 함수(일 두 자리만 지우고 다시 쓰고, 다시 읽어 확인)를 그대로 쓴다.
회사 실제 문서에서 만든 것이므로 b6_dates/ 안에서만 쓴다 (git·외부 게시 금지).

    python make_b6_dates.py <고른 원본 목록.json> [최대 장수]
"""
import calendar
import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

GEN = Path.home() / "Desktop/SCRYN/3_탐지기/sandburg-scryn-forgery/dataset/scripts"
sys.path.insert(0, str(GEN))
sys.path.insert(0, str(Path.home() / "Desktop/SCRYN/3_탐지기/scryn-combine-web"))
import auto_receipt as AR  # noqa: E402
import _compat  # noqa: E402,F401  (logic_detector 찾기)
from logic_detector.logic import refdata as R  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "b6_dates"
OUT.mkdir(exist_ok=True)
WD = "월화수목금토일"


def target_day(d):
    """d(토·일·공휴일) 바로 앞의 평일(공휴일 아님), 같은 달 안. 없으면 None."""
    x = d - timedelta(days=1)
    while x.month == d.month and (x.weekday() >= 5 or R.is_holiday(x)):
        x -= timedelta(days=1)
    return x if x.month == d.month and x.day >= 1 else None


def forge(path, visit):
    img = cv2.imread(str(path))
    up, table, k = AR.orient(img, dates=True)
    new = target_day(visit)
    if new is None:
        return None, "같은 달 안에 앞 평일 없음"
    cells = [(c, ds, g, t) for c, ds, g, t in AR._date_cells(up, table)
             if all((dt["y"], dt["m"], dt["d"]) == (visit.year, visit.month, visit.day) for dt in ds)]
    if not cells:
        return None, "진료일 칸을 글자 단위로 못 찾음"
    c, ds, glyphs, text = min(cells, key=lambda x: x[0]["y"])          # 위쪽(진료기간) 칸
    before = up.copy()
    if AR._set_days(up, c, glyphs, [(dt, new.day) for dt in ds], 7) is None or \
            not AR._verify(up, c, text, [(new.year, new.month, new.day)] * len(ds)):
        return None, "다시 써서 읽어 보니 새 날짜가 안 나옴"
    mask = AR.changed_mask(before, up)
    out = np.ascontiguousarray(np.rot90(up, -k)); mask = np.ascontiguousarray(np.rot90(mask, -k))
    desc = (f"진료일 {visit} ({WD[visit.weekday()]}) → {new} ({WD[new.weekday()]}). 진료기간 칸의 일(日) 두 자리만 지우고 다시 씀. "
            f"공휴일 가산 표시와 발행일은 그대로")
    return (out, mask, desc), ""


if __name__ == "__main__":
    picks = json.load(open(sys.argv[1], encoding="utf-8"))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rows, n = [], 0
    for i, p in enumerate(picks):
        if n >= limit:
            break
        visit = date.fromisoformat(p["visit"])
        r, why = forge(Path(p["path"]), visit)
        if r is None:
            print("건너뜀", Path(p["path"]).name[:30], why)
            continue
        n += 1
        out, mask, desc = r
        bid = f"B6D_{n:02d}"
        cv2.imwrite(str(OUT / f"{bid}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"{bid}_mask.png"), mask)
        rows.append({"id": bid, "doc_type": "진료비 영수증", "method": "holiday_date_shift", "tampered": f"{bid}.jpg",
                     "mask": f"{bid}_mask.png", "original": p["path"], "description": desc})
        print(bid, desc)
    with open(OUT / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "doc_type", "method", "tampered", "mask", "original", "description"])
        w.writeheader(); w.writerows(rows)
    print(n, "장 →", OUT)


# ── 글자 덩어리 방식 (팀 함수 edit_ops.glyph_boxes / replace_glyphs). tesseract 로 칸을 못 찾을 때
def forge_glyphs(path, visit, sbox):
    import edit_ops
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new = target_day(visit)
    if new is None:
        return None, "같은 달 안에 앞 평일 없음"
    x, y, w, h = sbox
    cell = (max(0, x - 4), max(0, y - 5), w + 8, h + 10)
    g = edit_ops.glyph_boxes(gray, cell)
    if len(g) == 10:
        spans = [(8, 10)]
    elif len(g) == 21:
        spans = [(8, 10), (19, 21)]
    else:
        return None, f"글자 덩어리 수가 날짜와 안 맞음 ({len(g)})"
    before = img.copy()
    for i, (a, b) in enumerate(spans):
        if edit_ops.replace_glyphs(img, cell, g[a:b], f"{new.day:02d}", seed=11 + i) is None:
            return None, "다시 쓰기 실패"
    mask = AR.changed_mask(before, img)
    desc = (f"진료일 {visit} ({WD[visit.weekday()]}, 공휴일 체크 있음) → {new} ({WD[new.weekday()]}). "
            f"진료기간 칸의 일(日) 두 자리만 지우고 다시 씀. 공휴일 체크와 발행일은 그대로")
    return (img, mask, desc), ""


def run_glyphs(picks, fix=None):
    fix = fix or {}
    rows = []
    for n, p in enumerate(picks, 1):
        visit = date.fromisoformat(fix.get(n - 1, p["visit"]))
        r, why = forge_glyphs(Path(p["path"]), visit, p["sbox"])
        if r is None:
            print("건너뜀", n - 1, why)
            continue
        out, mask, desc = r
        bid = f"B6D_{len(rows) + 1:02d}"
        cv2.imwrite(str(OUT / f"{bid}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"{bid}_mask.png"), mask)
        rows.append({"id": bid, "doc_type": "진료비 영수증", "method": "holiday_date_shift", "tampered": f"{bid}.jpg",
                     "mask": f"{bid}_mask.png", "original": p["path"], "description": desc})
        print(bid, desc)
    with open(OUT / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "doc_type", "method", "tampered", "mask", "original", "description"])
        w.writeheader(); w.writerows(rows)
    return rows


# ── 글자 비율 방식: 날짜 칸에서 일(日) 두 자리 자리를 글자 수 비율로 잡아 그 부분만 replace_text
def forge_prop(path, visit, spans):
    """spans: [(칸 상자, 칸 글자 수, 일 첫 글자 위치)] — 오른쪽 끝 기준이면 위치를 음수로."""
    import edit_ops
    img = cv2.imread(str(path))
    new = target_day(visit)
    before = img.copy()
    for i, sp in enumerate(spans):
        (x, y, w, h), n, a = sp[:3]
        pre = sp[3] if len(sp) > 3 else ""
        cw = w / n
        a = n + a if a < 0 else a
        sub = (int(x + (a + 0.3 - (0.9 if pre else 0)) * cw), y + 1, int((1.7 + (0.9 if pre else 0)) * cw) + 3, h - 2)
        if _rewrite(img, sub, f"{pre}{new.day:02d}", 21 + i) is None:
            return None, f"다시 쓰기 실패 {sub}"
    mask = AR.changed_mask(before, img)
    desc = (f"진료일 {visit} ({WD[visit.weekday()]}, 공휴일 체크 있음) → {new} ({WD[new.weekday()]}). "
            f"진료기간 칸의 일(日) 두 자리만 지우고 다시 씀. 공휴일 체크와 발행일은 그대로")
    return (img, mask, desc), ""


def _rewrite(img, sub, text, seed, height=None):
    """sub 안 숫자 획을 지우고 같은 위치·높이·진하기로 text 를 쓴다 (팀 edit_ops 의 erase · write).
    height 를 주면 칸 가운데 그 높이 띠 안에서만 글자를 잰다 (표 선이 글자 높이로 잡히지 않게)."""
    import edit_ops
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if height:
        cy = sub[1] + sub[3] / 2
        sub = (sub[0], int(cy - 0.7 * height), sub[2], int(1.4 * height))
    ib = edit_ops.ink_bbox(gray, sub)
    if ib is None:
        return None
    if height and ib[3] > 1.25 * height:
        ib = (ib[0], int(sub[1] + 0.2 * height), ib[2], int(height))
    area = (ib[0] - 1, ib[1] - 1, ib[2] + 2, ib[3] + 2)
    ink, _ = edit_ops.ink_level(gray, area)
    top, height = ib[1], ib[3]
    work = img.copy()
    if edit_ops.erase(work, area, seed) is None:
        return None
    best = None
    for off in (0, 20, 40, 60):
        t = work.copy()
        edit_ops.write(t, text, right=ib[0] + ib[2] - 1, top=top, digit_height=height, ink=max(0.0, ink - off))
        got, _ = edit_ops.ink_level(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), area)
        if best is None or abs(got - ink) < abs(best[0] - ink):
            best = (got, t)
    img[:] = best[1]
    return True
