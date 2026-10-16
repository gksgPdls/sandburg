#!/usr/bin/env python3
"""진료비 영수증 원본 풀에서 조작본을 자동으로 만들어 samples/ 에 등록한다.

조작 종류 (op):
    copymove   금액 칸을 같은 열의 0 칸·빈칸에 픽셀 그대로 복사       → copymove/ (CM)
    sum_break  항목 금액 하나만 바꿔 열 합계와 어긋나게 함              → amount_sum_break/ (CL)
    digit      항목 금액에 0 을 붙여 자릿수를 키움 (15,000 → 150,000) → amount_digit/ (CL)
    sum_fixed  항목 금액과 열 합계를 같이 바꿔 열 합계는 맞춤          → amount_sum_fixed/ (GL)
    date_shift    날짜의 일(日)을 바꿈. 한 칸에 날짜가 둘이면 같은 만큼 → date_shift/ (GL)
    date_conflict 끝 날짜의 일을 시작 날짜보다 앞으로 (퇴원일 < 입원일)  → date_conflict/ (CL)
    date_copy     문서 안 다른 날짜의 일 두 자리를 픽셀째 복사해 덮음. 그런 짝이 없으면
                  끝 날짜 일의 끝자리를 문서 안 다른 숫자 글자로 덮어 기간을 늘림 → date_copy/ (CM)
    digit_combo   금액 맨 앞자리를 문서 안 다른 금액의 더 큰 숫자로 덮음   → digit_combo/ (CM)

- 표 인식·OCR 은 로컬(receipt.py)에서만 한다. 문서 내용은 출력하지 않는다
- 금액 칸은 OCR 신뢰도가 높은 것(MIN_CONF 이상)만 고른다. 설명에 적는 원래 금액이 틀리지 않게
- sum_break·digit 은 합계 칸을 찾지 않는다. 정상 영수증에서 금액 하나만 바꾸면 어떤 합계와든
  어긋나기 때문이다. 이 사진 품질에서는 열 전체를 정확히 읽어 합계 칸을 찾는 게 40장 중 1장만 된다
- sum_fixed 는 합계 칸을 알아야 해서, 합계 관계가 확인된 열이 있는 영수증에서만 된다 (드묾)
- 마스크는 저장 전 원본과 조작본을 비교해 실제로 바뀐 픽셀의 외곽 사각형으로 만든다
- 이미 labels.csv 에 쓰인 원본은 다시 쓰지 않는다

- 날짜는 일(日) 두 자리만 바꾼다. 날짜 칸을 한 줄로 다시 읽어 tesseract 가 주는 글자별 위치를
  얻고, 그 범위 안의 실제 획으로 위치·높이를 다시 잰다. 구분 기호(- .)가 있는 날짜만 쓰고,
  조작 후 그 칸을 다시 읽어 새 날짜가 나오고 글자 수가 그대로일 때만 채택한다

사용: auto_receipt.py copymove 10 [--pool _work/pool/DOC210 ...] [--seed 0] [--author 김동현]
"""

import argparse
import calendar
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from edit_ops import copy_block, glyph_ink, replace_glyphs, replace_text
from labels import ROOT, load_all
from receipt import char_boxes, find_dates, read_table
from register_sample import register

OPS = {
    "copymove": ("copymove", "copy_move", "쉬움"),
    "sum_break": ("amount_sum_break", "content_logic", "쉬움"),
    "digit": ("amount_digit", "content_logic", "중간"),
    "sum_fixed": ("amount_sum_fixed", "glyph", "중간"),
    "date_shift": ("date_shift", "glyph", "중간"),
    "date_conflict": ("date_conflict", "content_logic", "쉬움"),
    "date_copy": ("date_copy", "copy_move", "중간"),
    "digit_combo": ("digit_combo", "copy_move", "중간"),
}
DATE_OPS = {"date_shift", "date_conflict", "date_copy"}
DOC_TYPE = {"DOC210": "진료비 영수증", "DOC205": "진료비 영수증 (퇴원·세부내역 계열)"}
TMP = ROOT / "_work" / "tmp"
MIN_CONF = 85


# ---------- 표 구조 ----------

def columns(table):
    """같은 x·폭을 가진 칸끼리 묶은 열 목록. 각 열은 y 순서."""
    cols = []
    for c in sorted(table, key=lambda c: (c["x"], c["y"])):
        for col in cols:
            h = col[0]
            if abs(h["x"] - c["x"]) <= 3 and abs(h["w"] - c["w"]) <= 3:
                col.append(c)
                break
        else:
            cols.append([c])
    return [sorted(col, key=lambda c: c["y"]) for col in cols if len(col) >= 3]


def total_of(col):
    """열에서 '맨 아래 금액 = 위 금액들의 합'이 성립하면 (합계 칸, 항목 칸들)."""
    am = [c for c in col if c["amount"] is not None]
    if len(am) < 3:
        return None
    total, items = am[-1], am[:-1]
    nonzero = [c for c in items if c["amount"] > 0]
    if total["amount"] >= 1000 and len(nonzero) >= 2 and sum(c["amount"] for c in items) == total["amount"]:
        return total, items
    return None


def fmt(n):
    return f"{n:,}"


def same_size(a, b):
    return abs(a["w"] - b["w"]) <= 2 and abs(a["h"] - b["h"]) <= 2


def box(c):
    return c["x"], c["y"], c["w"], c["h"]


def inner(c, m=2):
    return c["x"] + m, c["y"] + m, c["w"] - 2 * m, c["h"] - 2 * m


# ---------- 조작 ----------

def op_copymove(img, table, rng):
    for col in rng.sample(columns(table), len(columns(table))):
        srcs = [c for c in col if c["amount"] and c["amount"] >= 1000 and c["conf"] >= MIN_CONF]
        dsts = [c for c in col if (c["amount"] == 0 or (c["text"] == "" and c["amount"] is None))]
        pairs = [(s, d) for s in srcs for d in dsts if same_size(s, d)]
        if not pairs:
            continue
        s, d = rng.choice(pairs)
        x, y, w, h = inner(s)
        dx, dy = d["x"] + 2, d["y"] + 2
        w, h = min(w, d["w"] - 4), min(h, d["h"] - 4)
        img[dy:dy + h, dx:dx + w] = img[y:y + h, x:x + w].copy()
        has_total = total_of(col) is not None
        was = "0" if d["amount"] == 0 else "빈칸"
        return {
            "description": f"금액 칸 {fmt(s['amount'])} (x={x}, y={y})을 같은 열의 {was} 칸 (x={dx}, y={dy})에 "
                           f"픽셀 그대로 복사. 합계는 고치지 않음",
            "detection": "copy-move: 같은 픽셀 블록이 문서 안 두 곳에 있음. 같은 글자를 옮겨서 글리프로는 못 잡음"
                         + (". 열 합계가 어긋나 콘텐츠 로직에도 걸림" if has_total else ""),
            "notes": f"src={x},{y},{w},{h} dst={dx},{dy}",
        }
    return None


def _pick_item(table, rng, min_amount=1000):
    """합계 관계가 확인된 열의 (합계 칸, 항목 칸). sum_fixed 용."""
    cands = []
    for col in columns(table):
        t = total_of(col)
        if t:
            total, items = t
            cands += [(total, it) for it in items if it["amount"] >= min_amount]
    return rng.choice(cands) if cands else (None, None)


def _pick_amount(table, rng, min_amount=1000):
    """OCR 신뢰도가 높은 금액 칸 하나."""
    cands = [c for c in table if c["amount"] and c["amount"] >= min_amount and c["conf"] >= MIN_CONF]
    return rng.choice(cands) if cands else None


def _new_amount(old, rng):
    """자릿수가 같고 그럴듯한 새 금액 (백 원 단위 유지)."""
    digits = len(str(old))
    for _ in range(50):
        new = old + rng.choice([-1, 1]) * rng.randint(1, 9) * 10 ** (digits - 2)
        new = round(new, -2) if old % 100 == 0 else new
        if new > 0 and new != old and len(str(new)) == digits:
            return new
    return None


def op_sum_break(img, table, rng):
    it = _pick_amount(table, rng)
    if not it:
        return None
    new = _new_amount(it["amount"], rng)
    if not new or not replace_text(img, inner(it, 1), fmt(new), seed=rng.randint(0, 9999)):
        return None
    return {
        "description": f"금액 칸 {fmt(it['amount'])} → {fmt(new)} (x={it['x']}, y={it['y']}). "
                       f"배경으로 지우고 다시 씀. 다른 금액·합계 칸은 그대로",
        "detection": f"콘텐츠 로직: 금액 하나만 {fmt(new - it['amount'])}원 바뀌어 이 칸이 들어가는 합계와 어긋남. "
                     f"글리프: 다시 쓴 숫자의 폰트가 다름",
        "notes": "",
    }


def op_digit(img, table, rng):
    it = _pick_amount(table, rng)
    if not it or it["amount"] * 10 >= 10 ** 9:
        return None
    new = it["amount"] * 10
    if not replace_text(img, inner(it, 1), fmt(new), seed=rng.randint(0, 9999)):
        return None
    return {
        "description": f"금액 칸 {fmt(it['amount'])} → {fmt(new)} (x={it['x']}, y={it['y']}). "
                       f"끝에 0을 붙여 자릿수를 키움. 다른 금액·합계 칸은 그대로",
        "detection": "콘텐츠 로직: 이 칸이 들어가는 합계와 어긋나고, 금액이 10배라 이상치. "
                     "글리프: 다시 쓴 숫자의 폰트가 다름. 한 자리 늘어 칸 안 글자 폭이 달라짐",
        "notes": "",
    }


def op_sum_fixed(img, table, rng):
    total, it = _pick_item(table, rng)
    if not it:
        return None
    new = _new_amount(it["amount"], rng)
    if not new:
        return None
    new_total = total["amount"] + new - it["amount"]
    if len(str(new_total)) != len(str(total["amount"])):
        return None
    s = rng.randint(0, 9999)
    if not replace_text(img, inner(it, 1), fmt(new), seed=s) or \
            not replace_text(img, inner(total, 1), fmt(new_total), seed=s + 1):
        return None
    return {
        "description": f"항목 금액 {fmt(it['amount'])} → {fmt(new)} (x={it['x']}, y={it['y']}), "
                       f"열 합계 {fmt(total['amount'])} → {fmt(new_total)} (x={total['x']}, y={total['y']}). "
                       f"둘 다 배경으로 지우고 다시 씀",
        "detection": "글리프: 다시 쓴 두 곳의 폰트가 주변 숫자와 다름. 열 합계는 맞아서 콘텐츠 로직은 이 열에서 못 잡음 "
                     "(단 진료비 총액 등 다른 합계 칸과는 어긋날 수 있음)",
        "notes": "",
    }


def _date_cells(img, table):
    """날짜가 있고 글자 덩어리 수가 OCR 글자 수와 맞는 칸들. [(칸, 날짜들, 글자 외곽들)]"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for c in table:
        if not find_dates(c["text"]):
            continue
        # 칸 하나를 다시 읽어 글자별 위치를 얻는다. 날짜도 이 결과로 다시 찾는다
        text, glyphs = char_boxes(gray, box(c))
        dates = find_dates(text)
        if not dates or len(glyphs) != len(text) or not all(d["separated"] for d in dates):
            continue
        # tesseract 가 준 글자 위치가 옆 글자까지 삼키는 경우가 있다. 일(日) 두 글자의 폭이
        # 이 칸 숫자들의 보통 폭과 크게 다르면 뺀다. 옆 글자를 지운 경우는 _verify 가 잡는다
        digit_w = np.median([g[2] for ch, g in zip(text, glyphs) if ch.isdigit()])
        ok = all(0.5 * digit_w <= glyphs[i][2] <= 2.0 * digit_w
                 for d in dates for i in range(*d["day_span"]))
        if ok:
            out.append((c, dates, glyphs, text))
    return out


def _verify(img, c, text, expect):
    """조작한 칸을 다시 읽어 expect[(y, m, d)] 가 모두 나오고 글자 수가 그대로인지."""
    got_text, _ = char_boxes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box(c))
    got = {(d["y"], d["m"], d["d"]) for d in find_dates(got_text)}
    return len(got_text) == len(text) and all(e in got for e in expect)


def _valid_day(y, m, d):
    return 1 <= d <= calendar.monthrange(y, m)[1]


def _set_days(img, c, glyphs, changes, seed):
    """changes: [(날짜, 새 일)]. 각 날짜의 일 두 자리 글자를 바꾼다. 조작 영역들 또는 None."""
    boxes = []
    for i, (dt, new_d) in enumerate(changes):
        a, b = dt["day_span"]
        r = replace_glyphs(img, box(c), glyphs[a:b], f"{new_d:02d}", seed=seed + i)
        if r is None:
            return None
        boxes.append(r)
    return boxes


def _fmt_date(dt, d=None):
    return f"{dt['y']}-{dt['m']:02d}-{(d or dt['d']):02d}"


def op_date_shift(img, table, rng):
    cands = _date_cells(img, table)
    rng.shuffle(cands)
    for c, dates, glyphs, text in cands:
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        new = [dt["d"] + delta for dt in dates]
        if not all(_valid_day(dt["y"], dt["m"], d) for dt, d in zip(dates, new)):
            continue
        before = img.copy()
        if _set_days(img, c, glyphs, list(zip(dates, new)), rng.randint(0, 9999)) is None or \
                not _verify(img, c, text, [(dt["y"], dt["m"], d) for dt, d in zip(dates, new)]):
            img[:] = before
            continue
        olds = ", ".join(_fmt_date(dt) for dt in dates)
        news = ", ".join(_fmt_date(dt, d) for dt, d in zip(dates, new))
        return {
            "description": f"날짜 칸 (x={c['x']}, y={c['y']})의 일(日)을 {delta:+d}일 바꿈: {olds} → {news}. "
                           f"일 두 자리만 배경으로 지우고 다시 씀" + (". 칸 안 두 날짜를 같은 만큼 바꿔 순서 유지" if len(dates) > 1 else ""),
            "detection": "글리프: 다시 쓴 두 숫자의 폰트가 같은 칸의 나머지 숫자와 다름. 이 칸 안에서는 날짜 순서가 맞아 "
                         "콘텐츠 로직은 약함 (발행일 등 다른 날짜 칸과는 어긋날 수 있음)",
            "notes": "",
        }
    return None


def op_date_conflict(img, table, rng):
    cands = [x for x in _date_cells(img, table) if len(x[1]) >= 2]
    rng.shuffle(cands)
    for c, dates, glyphs, text in cands:
        a, b = dates[0], dates[1]
        if (a["y"], a["m"]) != (b["y"], b["m"]) or a["d"] <= 1 or b["d"] < a["d"]:
            continue
        new_d = a["d"] - rng.randint(1, min(5, a["d"] - 1))
        before = img.copy()
        if _set_days(img, c, glyphs, [(b, new_d)], rng.randint(0, 9999)) is None or \
                not _verify(img, c, text, [(a["y"], a["m"], a["d"]), (b["y"], b["m"], new_d)]):
            img[:] = before
            continue
        return {
            "description": f"날짜 칸 (x={c['x']}, y={c['y']})의 끝 날짜 {_fmt_date(b)} → {_fmt_date(b, new_d)}. "
                           f"시작 날짜 {_fmt_date(a)}보다 앞서게 함. 일 두 자리만 지우고 다시 씀",
            "detection": "콘텐츠 로직: 끝 날짜(퇴원·종료일)가 시작 날짜(입원·시작일)보다 앞섬. 규칙 기반으로 결정적 판정 가능. "
                         "글리프: 다시 쓴 두 숫자의 폰트가 다름",
            "notes": "",
        }
    return None


def _similar(a, b):
    """두 글자 획 외곽의 크기가 붙여도 티 나지 않을 만큼 비슷한지."""
    return abs(a[3] - b[3]) <= 1 and abs(a[2] - b[2]) <= 2


def op_date_copy(img, table, rng):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    days = []  # (칸, 날짜, 칸 글자, 일 두 자리 획 외곽, 날짜 순번)
    for c, dates, glyphs, text in _date_cells(img, table):
        for i, dt in enumerate(dates):
            ink = glyph_ink(gray, box(c), glyphs[slice(*dt["day_span"])])
            if ink:
                days.append((c, dt, text, ink, i, dates))
    pairs = [(s, d) for s in days for d in days
             if s[1]["d"] != d[1]["d"] and _similar(s[3], d[3]) and _valid_day(d[1]["y"], d[1]["m"], s[1]["d"])]
    rng.shuffle(pairs)
    for (sc, sdt, _, sink, _, _), (dc, ddt, dtext, dink, di, ddates) in pairs:
        # 같은 칸의 시작~끝 순서가 뒤집히면 날짜 모순 샘플과 겹치므로 피한다
        new = [(x["y"], x["m"], sdt["d"] if j == di else x["d"]) for j, x in enumerate(ddates)]
        if len(new) >= 2 and new[1] < new[0]:
            continue
        before = img.copy()
        if copy_block(img, sink, dink, box(dc), seed=rng.randint(0, 9999)) is None or \
                not _verify(img, dc, dtext, new):
            img[:] = before
            continue
        same = sc is dc
        return {
            "description": f"{'같은 칸' if same else '다른 칸'}의 날짜 {_fmt_date(sdt)}에서 일(日) 두 자리 "
                           f"(x={sink[0]}, y={sink[1]})를 픽셀째 복사해, 날짜 칸 (x={dc['x']}, y={dc['y']})의 "
                           f"{_fmt_date(ddt)} 일 자리를 덮음 → {_fmt_date(ddt, sdt['d'])}",
            "detection": "copy-move: 같은 두 숫자 블록이 문서 안 두 곳에 있음 (블록이 작아 매칭 난이도 있음). "
                         "원본 글자를 옮겨 글리프로는 못 잡음 (방법론 케이스 8)",
            "notes": f"src={sink[0]},{sink[1]},{sink[2]},{sink[3]} dst={dink[0]},{dink[1]},{dink[2]},{dink[3]}",
        }
    # 외래 영수증은 진료기간이 하루짜리라 일이 다른 날짜 짝이 거의 없다. 끝자리 한 글자로 바꾼다
    return _date_last_digit(img, table, rng)


def _date_last_digit(img, table, rng):
    """끝 날짜 일(日)의 끝자리를 문서 안 다른 숫자 글자로 덮어 날짜를 뒤로 미룬다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dcells = _date_cells(img, table)
    donors = []  # 날짜·금액 칸의 숫자 글자: (숫자, 획 외곽, 칸)
    for c, _, glyphs, text in dcells:
        for ch, g in zip(text, glyphs):
            if ch.isdigit() and (ink := glyph_ink(gray, box(c), [g])):
                donors.append((ch, ink, c))
    for c, text, glyphs in _amount_glyphs(gray, table):
        for ch, g in zip(text, glyphs):
            if ch.isdigit() and (ink := glyph_ink(gray, box(c), [g])):
                donors.append((ch, ink, c))
    rng.shuffle(dcells)
    for c, dates, glyphs, text in dcells:
        dt = dates[-1]  # 칸의 마지막 날짜 (진료기간이면 끝 날짜)
        i = dt["day_span"][1] - 1
        last = glyph_ink(gray, box(c), [glyphs[i]])
        if not last:
            continue
        tens = dt["d"] // 10
        opts = [d for d in donors if tens * 10 + int(d[0]) > dt["d"]
                and _valid_day(dt["y"], dt["m"], tens * 10 + int(d[0])) and _similar(d[1], last)
                and d[1] != last]
        rng.shuffle(opts)
        for ch, ink, dc in opts[:6]:
            new_d = tens * 10 + int(ch)
            expect = [(x["y"], x["m"], new_d if x is dt else x["d"]) for x in dates]
            before = img.copy()
            if copy_block(img, ink, last, box(c), seed=rng.randint(0, 9999)) is None or \
                    not _verify(img, c, text, expect):
                img[:] = before
                continue
            span = f" (같은 칸 시작 날짜 {_fmt_date(dates[0])})" if len(dates) > 1 else ""
            return {
                "description": f"날짜 칸 (x={c['x']}, y={c['y']})의 {_fmt_date(dt)}{span} → {_fmt_date(dt, new_d)}. "
                               f"일의 끝자리를 문서 안 다른 칸 (x={dc['x']}, y={dc['y']})의 숫자 '{ch}'를 픽셀째 "
                               f"오려 붙여 바꿈. 날짜를 뒤로 미뤄 기간을 늘림",
                "detection": "copy-move: 숫자 한 글자 블록이 문서 안 두 곳에 있음 (블록이 아주 작아 매칭이 어려움). "
                             "원본 글자를 옮겨 글리프로는 못 잡음 (방법론 케이스 8). 기간이 늘어 진료비·일수와 "
                             "어긋나면 콘텐츠 로직으로 잡을 여지가 있음",
                "notes": f"src={ink[0]},{ink[1]},{ink[2]},{ink[3]} dst={last[0]},{last[1]},{last[2]},{last[3]}",
            }
    return None


def _amount_glyphs(gray, table):
    """신뢰도 높은 금액 칸마다 (칸, 글자, 글자 외곽들)."""
    out = []
    for c in table:
        if not (c["amount"] and c["amount"] >= 1000 and c["conf"] >= MIN_CONF):
            continue
        text, glyphs = char_boxes(gray, box(c), whitelist="0123456789,")
        if len(text) == len(glyphs) and text.replace(",", "").isdigit() and \
                int(text.replace(",", "")) == c["amount"]:
            out.append((c, text, glyphs))
    return out


def op_digit_combo(img, table, rng):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = _amount_glyphs(gray, table)
    # 문서 안 숫자 글자들: (숫자, 획 외곽, 칸)
    donors = []
    for c, text, glyphs in cells:
        for ch, g in zip(text, glyphs):
            if ch.isdigit():
                ink = glyph_ink(gray, box(c), [g])
                if ink:
                    donors.append((ch, ink, c))
    rng.shuffle(cells)
    for c, text, glyphs in cells:
        lead = glyph_ink(gray, box(c), [glyphs[0]])
        if not lead:
            continue
        cur = int(text[0])
        # 금액을 키우는 쪽으로: 맨 앞자리를 더 큰 숫자로
        opts = [d for d in donors if int(d[0]) > cur and d[2] is not c and _similar(d[1], lead)]
        rng.shuffle(opts)
        for ch, ink, dc in opts[:6]:
            new_text = ch + text[1:]
            new_amount = int(new_text.replace(",", ""))
            before = img.copy()
            if copy_block(img, ink, lead, box(c), seed=rng.randint(0, 9999)) is None:
                img[:] = before
                continue
            got, _ = char_boxes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box(c), whitelist="0123456789,")
            if got != new_text:
                img[:] = before
                continue
            return {
                "description": f"금액 칸 (x={c['x']}, y={c['y']}) {fmt(c['amount'])} → {fmt(new_amount)}. "
                               f"맨 앞자리를 문서 안 다른 금액 칸 (x={dc['x']}, y={dc['y']})의 숫자 '{ch}'를 "
                               f"픽셀째 오려 붙여 바꿈",
                "detection": "copy-move: 숫자 한 글자 블록이 문서 안 두 곳에 있음 (블록이 아주 작아 매칭이 어려움). "
                             "원본 글자를 옮겨 글리프로는 못 잡음 (방법론 케이스 8). 금액만 바뀌어 합계와 어긋나 콘텐츠 로직에 걸림",
                "notes": f"src={ink[0]},{ink[1]},{ink[2]},{ink[3]} dst={lead[0]},{lead[1]},{lead[2]},{lead[3]}",
            }
    return None


FUNCS = {"copymove": op_copymove, "sum_break": op_sum_break, "digit": op_digit, "sum_fixed": op_sum_fixed,
         "date_shift": op_date_shift, "date_conflict": op_date_conflict,
         "date_copy": op_date_copy, "digit_combo": op_digit_combo}


# ---------- 공통 ----------

def orient(img, dates=False):
    """표가 가장 잘 읽히는 방향으로 돌린다. (돌린 이미지, 표, 되돌릴 때 쓸 k).
    dates=True 면 넓은 칸(진료기간 등)까지 읽고 날짜 칸 수로 방향을 고른다."""
    best = None
    ks = (0,) if img.shape[0] >= img.shape[1] else (-1, 1)
    for k in ks:
        r = np.ascontiguousarray(np.rot90(img, k))
        t = read_table(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), max_w=700 if dates else 300)
        score = sum(bool(find_dates(c["text"])) if dates else c["amount"] is not None for c in t)
        if best is None or score > best[3]:
            best = (r, t, k, score)
    return best[0], best[1], best[2]


def changed_mask(before, after, pad=0):
    """저장 전 두 이미지에서 바뀐 픽셀 덩어리마다 외곽 사각형을 칠한 마스크."""
    diff = (np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > 0).astype(np.uint8)
    merged = cv2.dilate(diff, np.ones((15, 15), np.uint8))
    n, labels = cv2.connectedComponents(merged)
    mask = np.zeros(diff.shape, np.uint8)
    for i in range(1, n):
        ys, xs = np.nonzero((labels == i) & (diff > 0))
        mask[ys.min() - pad:ys.max() + 1 + pad, xs.min() - pad:xs.max() + 1 + pad] = 255
    return mask


def used_sources():
    return {r.get("source", "") for r in load_all()}


def pool_sources(pool):
    """pool 파일 → pool_id. CSV 의 source 에는 짧은 pool_id 를 쓴다.
    원래 real_docs 경로는 로컬의 _work/pool/manifest.csv 에만 남긴다."""
    manifest = pool.parent / "manifest.csv"
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        return {r["file"]: r["pool_id"] for r in csv.DictReader(f)}


def tamper_file(op, f, rng):
    """원본 파일 한 장에 op 조작을 한다. (조작본, 마스크, 설명, k) 또는 None. 원래 방향으로 되돌려 준다."""
    img = cv2.imread(str(f))
    if img is None:
        return None
    up, table, k = orient(img, dates=op in DATE_OPS)
    before = up.copy()
    info = FUNCS[op](up, table, rng)
    if not info:
        return None
    mask = changed_mask(before, up)
    return (np.ascontiguousarray(np.rot90(up, -k)), np.ascontiguousarray(np.rot90(mask, -k)), info, k)


def main():
    p = argparse.ArgumentParser(description="영수증 자동 조작")
    p.add_argument("op", choices=list(OPS))
    p.add_argument("count", type=int)
    p.add_argument("--pool", nargs="+", default=[str(ROOT / "_work" / "pool" / "DOC210")],
                   help="원본 풀 폴더 (여러 개 가능)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--author", default="김동현")
    a = p.parse_args()

    method, axis, difficulty = OPS[a.op]
    pools = [Path(x) for x in a.pool]
    sources = pool_sources(pools[0])
    used = used_sources()
    files = sorted(f for pool in pools for f in pool.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng = random.Random(a.seed)
    rng.shuffle(files)
    TMP.mkdir(parents=True, exist_ok=True)

    made, tried = [], 0
    for f in files:
        if len(made) == a.count:
            break
        src = sources.get(f.relative_to(f.parent.parent).as_posix(), "")
        if src in used:
            continue
        tried += 1
        res = tamper_file(a.op, f, rng)
        if not res:
            continue
        tampered, mask, info, k = res
        t_path, m_path = TMP / f"t_{os.getpid()}.jpg", TMP / f"m_{os.getpid()}.png"  # 동시 실행 대비
        cv2.imwrite(str(t_path), tampered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(m_path), mask)
        if k:
            info["notes"] = (info["notes"] + " " if info["notes"] else "") + \
                f"좌표는 세로로 돌린 이미지 기준 (np.rot90 k={k})"
        sid = register(method=method, axis=axis, original=f, tampered=t_path, mask=m_path,
                       description=info["description"], detection=info["detection"],
                       doc_type=DOC_TYPE.get(f.parent.name, f.parent.name), difficulty=difficulty,
                       external_ai="N",
                       author=a.author, source=src,
                       notes=f"auto_receipt.py {a.op} seed={a.seed}. {info['notes']}".strip())
        used.add(src)
        made.append(sid)
        t_path.unlink()
        m_path.unlink()
    print(f"{a.op}: {len(made)}개 생성 (시도한 원본 {tried}장) → samples/{method}/  {' '.join(made)}")
    if len(made) < a.count:
        print(f"  목표 {a.count}개에 못 미침. 원본을 더 뽑는다: sample_originals.py DOC210 N --seed 다른값",
              file=sys.stderr)


if __name__ == "__main__":
    main()
