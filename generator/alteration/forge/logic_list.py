"""C4: 내용 논리가 맞는 위조 목록 (탐지기의 논리 검사에 안 걸리게).
원본마다 표를 읽어(동현님 receipt.read_table, auto_receipt.total_of 그대로) 아래 규칙 중 되는 것 하나를 골라
함께 바꿔야 하는 칸을 모두 같이 바꾸는 목록을 만든다.

규칙 (앞에서부터 시도)
  관계     : 문서의 금액 관계를 전부 찾고(열합계: 열 맨 아래 = 위 칸 합, 행합계: 줄 맨 오른쪽 = 나머지 합 또는 a + b = c,
             곱: 단가 × 수량 = 금액), 재료 칸 하나를 바꾸면 그 칸이 들어간 결과 칸을 끝까지 따라가 같이 바꿈
             (열합계와 행합계가 겹치는 표에서 소계·총계까지). 관계끼리 모순이면 그 후보는 버림
  날짜순서 : 날짜 하나를 바꾸되 문서 안 다른 날짜들과의 앞뒤(이전·같음·이후)가 그대로이고, 오늘보다 뒤가 아니게 (±14일)
그리고 어느 규칙이든, 바꾼 값과 똑같은 값이 문서 다른 칸에 또 있으면 그 칸도 같은 새 값으로 (총액이 두 번 적힌 경우 등)

python scripts/generate/logic_list.py --limit 160   # 여러 번 실행하면 이어서 함
결과: outputs/고도화/원본목록_논리맞춤.csv (forge_v2 run 에 FORGE_LIST 로), 검사 기록 outputs/고도화/_논리스캔.csv"""
import argparse
import csv
import datetime as dt
import random
import re
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "generate"))
OUT = ROOT / "outputs" / "고도화"
SCAN = OUT / "_논리스캔.csv"
LIST = OUT / "원본목록_논리맞춤.csv"
MAXE = 8
FIELDS = ["id", "original", "kind", "규칙", "original_value", "forged_value", "x", "y", "w", "h"] + \
    [f"{c}{k}" for k in range(2, MAXE + 1) for c in ("original_value", "forged_value", "x", "y", "w", "h")]
DATE_FULL = re.compile(r"^(20\d{2})([.\-])(\d{2})\2(\d{2})$")
TODAY = dt.date.today()
DATE_SHIFT = 14      # 날짜를 옮기는 최대 일수 (10/23: 60일이면 5월 영수증이 7월로 가서 부자연스러움)


OCR = "rapid"        # rapid: RapidOCR(문서 전체 글자 줄 찾기+읽기), tesseract: 동현님 표 칸 나누기 + tesseract
FALLBACK = True      # 관계를 못 찾으면 칸 하나만 바꾸는 위조(논리 깨짐)라도 만든다. --strict 로 끔
_RAPID = None
AMOUNT = re.compile(r"^\d{1,3}(,\d{3})+$|^\d{4,}$")


def _rapid():
    global _RAPID
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    return _RAPID


def read_cells(gray):
    """문서의 값 칸 목록. 각 칸 dict(x, y, w, h, text, conf, amount) (동현님 read_table과 같은 모양).
    rapid: 표 선이 흐리거나 비스듬해도 글자 줄을 찾음. 한 줄에 값 하나만 있는 것(금액 하나, 날짜 하나 등)을 칸으로 씀."""
    if OCR == "tesseract":
        from receipt import read_table
        return read_table(gray)
    res, _ = _rapid()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cells = []
    for quad, text, conf in res or []:
        t = text.strip().replace(" ", "").replace("，", ",").replace("．", ".")
        # '⑤13,160'처럼 번호가 붙어 읽힌 금액은 끝의 금액만 (같은 값 칸을 놓치지 않게, 10/23 auto00012)
        tail = re.search(r"(\d{1,3}(?:[.,]\d{3})+)$", t)
        if tail and tail.group(1) != t and not re.search(r"\d$", t[:tail.start()]):
            t = tail.group(1)
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):
            t = t.replace(".", ",")
        elif "-" not in t and re.search(r"\d", t):
            # '6,000②'처럼 동그라미 번호가 붙은 금액: 숫자 덩어리가 하나뿐이면 그 금액만
            toks = re.findall(r"\d{1,3}(?:[.,]\d{3})+|\d+", t)
            if len(toks) == 1 and re.fullmatch(r"\d{1,3}([.,]\d{3})+|\d{1,7}", toks[0]):
                t = toks[0].replace(".", ",")
                if "," in t:
                    t = f"{int(t.replace(',', '')):,}"
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        x, y = int(min(xs)), int(min(ys))
        amount = int(t.replace(",", "")) if re.fullmatch(r"\d{1,3}(,\d{3})*", t) else None
        cells.append({"x": x, "y": y, "w": int(max(xs)) - x, "h": int(max(ys)) - y, "text": t,
                      "conf": float(conf) * 100, "amount": amount, "raw": text})
    return cells


def columns(table):
    """금액 열: 오른쪽 끝이 맞는(±4px) 금액 칸끼리. 표 칸 모양이 아니라 글자 줄 위치로 묶음 (rapid용)."""
    am = sorted([c for c in table if c["amount"] is not None], key=lambda c: c["x"] + c["w"])
    cols = []
    for c in am:
        r = c["x"] + c["w"]
        for col in cols:
            if abs((col[0]["x"] + col[0]["w"]) - r) <= 4:
                col.append(c)
                break
        else:
            cols.append([c])
    return [sorted(col, key=lambda c: c["y"]) for col in cols if len(col) >= 3]


def _ok_box(gray, c, value, ink_fn):
    import layout
    import forge_v2 as F
    from sum_fix_list import value_box
    b = value_box(gray, c, ink_fn, m=0 if OCR == "rapid" else 2, value=value)
    if not b:
        return None
    bx = F.label_box({"x": b[0], "y": b[1], "w": b[2], "h": b[3]}, gray.shape)
    if layout._fit_value(layout._segments(gray, bx, ink_fn), value) is None:
        return None
    return b


def _dupes(table, cell, text):
    """같은 글자가 적힌 다른 칸들."""
    # '⑤'를 숫자로 잘못 읽어 '513,160'처럼 한 글자가 앞에 붙은 것도 같은 값 후보로 (상자는 뒤의 값만 맞추고, 못 맞추면 후보를 버림)
    return [c for c in table if c is not cell and (c["x"], c["y"]) != (cell["x"], cell["y"])
            and (c["text"] == text or (c["text"].endswith(text) and len(c["text"]) - len(text) == 1))]


LOGIC = True        # False면 관계를 찾지 않고 칸 하나만 바꿈 (--no-logic)
USE_FORM = True     # 한국어 OCR로 진료비 영수증 서식 찾기 (캐시나 easyocr가 있을 때)


def _form_layout(path, table):
    if not USE_FORM:
        return None
    try:
        import receipt_form  # noqa: F401  (서식 규칙 파일이 없으면 서식 없이)
    except ImportError:
        return None
    import receipt_form as RF
    # 숫자 칸이 늘어선 모양 + 서식 식으로 찾는다 (한국어 OCR은 사진에서 칸 이름을 자주 틀려 쓰지 않음, 10/23)
    L = RF.find_layout_struct(table)
    if L is not None and "⑦" in L["ok"]:
        L["type"] = "진료비"
        return L
    return RF.find_pharmacy(table)   # 약국 약제비 영수증


_EASY = None


def _has_easyocr():
    global _EASY
    if _EASY is None:
        try:
            import easyocr  # noqa: F401
            _EASY = True
        except Exception:
            _EASY = False
    return _EASY


def _edits_with_dupes(gray, table, pairs, ink_fn, dupes=True):
    """pairs: [(칸, 원래 글자, 새 글자)]. 같은 글자 칸도 추가. 상자가 하나라도 안 맞으면 None."""
    out = []
    seen = set()
    for j, (c, ov, nv) in enumerate(pairs):
        # 같은 값 칸은 합계(둘째 칸)에만. 항목 금액과 같은 값은 다른 항목일 수 있어 바꾸면 그 열 합계가 깨짐 (10/23)
        for cc in [c] + (_dupes(table, c, ov) if dupes and j >= 1 and len(ov) >= 6 else []):
            key = (cc["x"], cc["y"])
            if key in seen:
                continue
            seen.add(key)
            b = _ok_box(gray, cc, ov, ink_fn)
            if not b:
                return None   # 같은 값 칸 하나라도 못 고치면 값이 서로 달라지므로 이 후보는 버림 (10/23)
            out.append((ov, nv, b))
    return out if len(out) <= MAXE else None


# ---------- 관계 전체 따라가기 (10/23) ----------

def _rows(cells):
    """같은 줄(세로 가운데가 글자 높이 절반 안)끼리 묶음, 줄마다 왼쪽부터."""
    rows = []
    for c in sorted(cells, key=lambda c: c["y"] + c["h"] / 2):
        cy = c["y"] + c["h"] / 2
        for r in rows:
            if abs(r[0]["y"] + r[0]["h"] / 2 - cy) < max(r[0]["h"], c["h"]) / 2:
                r.append(c)
                break
        else:
            rows.append([c])
    return [sorted(r, key=lambda c: c["x"]) for r in rows]


def find_relations(table):
    """문서 안 금액 관계 전부. [(종류, 결과 칸 번호, [재료 칸 번호])]. 칸 번호는 table 안 위치.
    - 열합계: 열 맨 아래 = 위 칸 전부의 합
    - 행합계: 줄 맨 오른쪽 = 그 줄 나머지 전부의 합, 또는 줄 안 a + b = c
    - 곱: 줄 안 단가 × 수량 = 금액 (수량은 1~999 정수)"""
    import auto_receipt as AR
    idx = {id(c): i for i, c in enumerate(table)}
    am = [c for c in table if c["amount"] is not None and c["conf"] >= 80]
    rels, seen = [], set()

    def add(kind, total, items):
        key = (idx[id(total)], tuple(sorted(idx[id(i)] for i in items)))
        if key not in seen and idx[id(total)] not in key[1]:
            seen.add(key)
            rels.append((kind, key[0], list(key[1])))
    cols = AR.columns(table) if OCR == "tesseract" else columns(table)
    for col in cols:
        col = [c for c in col if c["amount"] is not None]
        for k in range(2, len(col)):
            items = col[:k]
            if col[k]["amount"] >= 1000 and sum(1 for c in items if c["amount"] > 0) >= 2 \
                    and sum(c["amount"] for c in items) == col[k]["amount"]:
                add("열합계", col[k], items)
    for row in _rows(am):
        if len(row) >= 3:
            last, rest = row[-1], row[:-1]
            if last["amount"] >= 1000 and sum(1 for c in rest if c["amount"] > 0) >= 2 \
                    and sum(c["amount"] for c in rest) == last["amount"]:
                add("행합계", last, rest)
            for i, a in enumerate(row):
                for b in row[i + 1:]:
                    for c in row:
                        if c is not a and c is not b and c["amount"] >= 1000 and a["amount"] > 0 and b["amount"] > 0 \
                                and a["amount"] + b["amount"] == c["amount"]:
                            add("행합계", c, [a, b])
                        if c is not a and c is not b and c["amount"] >= 1000 and a["amount"] > 0 and b["amount"] > 0 \
                                and (1 < b["amount"] < 1000 or 1 < a["amount"] < 1000) \
                                and a["amount"] * b["amount"] == c["amount"]:
                            add("곱", c, [a, b])
    return rels


def propagate(table, rels, start, delta):
    """start 칸을 delta 만큼 바꿀 때 관계를 다 맞추려면 각 칸이 얼마나 바뀌어야 하는지. {칸 번호: 차이}.
    한 칸에 서로 다른 차이가 필요하면(관계끼리 모순) None."""
    d = {start: delta}
    for _ in range(20):
        changed = False
        for kind, t, items in rels:
            if kind == "곱":
                a, b = items
                va, vb = table[a]["amount"], table[b]["amount"]
                qty_a = va < vb   # 작은 쪽을 수량으로
                q, p = (a, b) if qty_a else (b, a)
                if d.get(q, 0):
                    return None   # 수량은 바꾸지 않음
                need = d.get(p, 0) * table[q]["amount"]
            else:
                need = sum(d.get(i, 0) for i in items)
            if t in d:
                if d[t] != need:
                    return None
            elif need:
                if t == start:
                    return None
                d[t] = need
                changed = True
        if not changed:
            return d
    return None


def rule_relations(table, rng):
    """관계에 들어 있는 재료 칸(다른 관계의 결과가 아닌 칸) 하나를 바꾸고, 그 칸이 들어간 결과 칸을 끝까지 따라 바꾼다.
    yield ([(칸, 원래 글자, 새 글자)...], 쓴 관계 요약)"""
    import auto_receipt as AR
    rels = find_relations(table)
    if not rels:
        return
    totals = {t for _, t, _ in rels}
    leaves = sorted({i for kind, _, items in rels for i in items if i not in totals
                     and table[i]["amount"] >= 1000 and table[i]["conf"] >= AR.MIN_CONF})
    rng.shuffle(leaves)
    for leaf in leaves:
        for _ in range(5):
            new = AR._new_amount(table[leaf]["amount"], rng)
            if not new:
                break
            d = propagate(table, rels, leaf, new - table[leaf]["amount"])
            if not d or len(d) < 2:
                continue
            pairs, ok = [], True
            for i, dv in sorted(d.items(), key=lambda kv: kv[0] != leaf):
                old = table[i]["amount"]
                nv = old + dv
                if nv <= 0 or len(str(nv)) != len(str(old)):
                    ok = False
                    break
                pairs.append((table[i], AR.fmt(old), AR.fmt(nv)))
            if not ok:
                continue
            used = sorted({kind for kind, t, items in rels if t in d})
            yield pairs, "+".join(f"{k}{len({t for kk, t, _ in rels if kk == k and t in d})}" for k in used)
            break


def rule_col_sum(table, rng):
    import auto_receipt as AR
    cands = []
    cols = AR.columns(table) if OCR == "tesseract" else columns(table)
    for col in cols:
        t = AR.total_of(col)
        if t:
            total, items = t
            cands += [(total, it) for it in items if it["amount"] >= 1000 and it["conf"] >= AR.MIN_CONF]
    rng.shuffle(cands)
    for total, it in cands:
        new = AR._new_amount(it["amount"], rng)
        if not new:
            continue
        nt = total["amount"] + new - it["amount"]
        if nt > 0 and len(str(nt)) == len(str(total["amount"])):
            yield [(it, AR.fmt(it["amount"]), AR.fmt(new)), (total, AR.fmt(total["amount"]), AR.fmt(nt))]


def rule_row_sum(table, rng):
    import auto_receipt as AR
    am = [c for c in table if c["amount"] and c["amount"] >= 1000 and c["conf"] >= AR.MIN_CONF]
    cands = []
    for c in am:
        row = [d for d in am if d is not c and abs((d["y"] + d["h"] / 2) - (c["y"] + c["h"] / 2)) < c["h"] / 2]
        for i, a in enumerate(row):
            for b in row[i + 1:]:
                if a["amount"] + b["amount"] == c["amount"]:
                    cands += [(a, c), (b, c)]
    rng.shuffle(cands)
    for a, c in cands:
        new = AR._new_amount(a["amount"], rng)
        if not new:
            continue
        nc = c["amount"] + new - a["amount"]
        if nc > 0 and len(str(nc)) == len(str(c["amount"])):
            yield [(a, AR.fmt(a["amount"]), AR.fmt(new)), (c, AR.fmt(c["amount"]), AR.fmt(nc))]


def _parse_dates(table):
    from receipt import find_dates
    out = []
    for c in table:
        for d in find_dates(c["text"]):
            if not d["separated"]:   # '202405-1390' 같은 번호를 날짜로 오인하지 않게 (10/23 auto00083)
                continue
            try:
                out.append((c, dt.date(d["y"], d["m"], d["d"])))
            except ValueError:
                pass
    return out


def _receipt_months(table):
    out = set()
    for c in table:
        m = re.match(r"^(20\d{2})(0[1-9]|1[0-2])-\d{2,}", c.get("raw", c["text"]).replace(" ", ""))
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def rule_date_order(table, rng):
    ds = _parse_dates(table)
    singles = [(c, d) for c, d in ds if DATE_FULL.match(c["text"])]
    rng.shuffle(singles)
    for c, d in singles:
        others = [o for cc, o in ds if cc is not c]
        if not others:
            continue   # 비교할 날짜가 없으면 이 규칙의 의미가 없음
        m = DATE_FULL.match(c["text"])
        sep = m.group(2)
        # 다른 날짜와의 앞뒤가 그대로인 범위 [lo, hi]
        lo, hi = dt.date(2000, 1, 1), TODAY
        for o in others:
            if o < d:
                lo = max(lo, o + dt.timedelta(days=1))
            elif o > d:
                hi = min(hi, o - dt.timedelta(days=1))
            else:
                lo = hi = d   # 같은 날짜가 있으면 그것과 같이 바꿔야 해서 날짜는 그대로 (같은 값 규칙으로 처리 안 함)
        span = [k for k in range(-DATE_SHIFT, DATE_SHIFT + 1) if k and lo <= d + dt.timedelta(days=k) <= hi]
        # 영수증번호(연월-일련번호, 예: 202405-1390)의 연월이 이 날짜와 같으면 달이 바뀌지 않게 (서식 규칙, 10/23)
        if (d.year, d.month) in _receipt_months(table):
            span = [k for k in span if (d + dt.timedelta(days=k)).month == d.month]
        if not span:
            continue
        nd = d + dt.timedelta(days=rng.choice(span))
        yield [(c, c["text"], nd.strftime(f"%Y{sep}%m{sep}%d"))], (lo, hi)


def rule_single(table, rng):
    """관계를 못 찾았을 때: 금액 하나(또는 날짜 하나)만 바꿈. 내용 검사에 걸릴 수 있는 위조(논리 깨짐)."""
    import auto_receipt as AR
    am = [c for c in table if c["amount"] and c["amount"] >= 1000 and c["conf"] >= AR.MIN_CONF]
    rng.shuffle(am)
    for c in am:
        new = AR._new_amount(c["amount"], rng)
        if new:
            yield [(c, AR.fmt(c["amount"]), AR.fmt(new))]
    ds = [(c, d) for c, d in _parse_dates(table) if DATE_FULL.match(c["text"])]
    rng.shuffle(ds)
    for c, d in ds:
        sep = DATE_FULL.match(c["text"]).group(2)
        k = rng.choice([k for k in range(-DATE_SHIFT, DATE_SHIFT + 1) if k])
        nd = d + dt.timedelta(days=k)
        if nd <= TODAY:
            yield [(c, c["text"], nd.strftime(f"%Y{sep}%m{sep}%d"))]


def scan_one(path, ink_fn):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    table = read_cells(gray)
    rng = random.Random("logic" + path.stem)
    rules = [("관계", rule_relations), ("날짜순서", rule_date_order)]
    form = _form_layout(path, table) if LOGIC else None
    if form is not None:
        # 진료비 영수증 서식이면 서식 규칙으로. 숫자만 보는 관계 찾기는 급여 합을 비급여 칸으로 오인해서 쓰지 않음 (10/23)
        import auto_receipt as AR
        import receipt_form as RF
        gen = RF.plans_struct if form["type"] == "진료비" else RF.plans_pharmacy
        rules = [("진료비서식", lambda t, r: gen(form, r, AR._new_amount, AR.fmt)), ("날짜순서", rule_date_order)]
    if FALLBACK:
        rules.append(("단일(논리깨짐)", rule_single))
    if not LOGIC:   # 논리 맞추기 끔: 칸 하나만 바꿈
        rules = [("단일(논리깨짐)", rule_single)]
    for name, gen in rules:
        for item in gen(table, rng):
            if name in ("관계", "날짜순서", "진료비서식"):
                pairs, detail = item
            else:
                pairs, detail = item, ""
            ed = _edits_with_dupes(gray, table, pairs, ink_fn, dupes=(name != "진료비서식"))
            if not ed:
                continue
            label = (detail if name in ("관계", "진료비서식") else name)
            r = {"id": f"logic_{path.stem}", "original": _rel(path),
                 "kind": "날짜" if "-" in pairs[0][1] or "." in pairs[0][1] and "," not in pairs[0][1] else "금액",
                 "규칙": label + (f" 같은값{len(ed) - len(pairs)}" if len(ed) > len(pairs) else "")}
            for k, (ov, nv, b) in enumerate(ed, 1):
                sfx = "" if k == 1 else str(k)
                r.update({f"original_value{sfx}": ov, f"forged_value{sfx}": nv,
                          f"x{sfx}": b[0], f"y{sfx}": b[1], f"w{sfx}": b[2], f"h{sfx}": b[3]})
            return r
    return None


def _rel(path):
    """저장소 안이면 상대 경로, 밖(처음 보는 사진 폴더)이면 절대 경로."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def rescan():
    """이미 찾은 문서만 다시 (규칙을 고쳤을 때)."""
    import forge_v2 as F
    old = list(csv.DictReader(open(LIST, encoding="utf-8-sig")))
    rows = [r for r in (scan_one(ROOT / o["original"], F.stroke_only) for o in old) if r]
    with open(LIST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval=""); w.writeheader(); w.writerows(rows)
    print("다시 검사", len(old), "->", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=0)
    ap.add_argument("--src", default="dataset/original/bankbook", help="원본 사진 폴더 (저장소 밖 절대 경로도 됨)")
    ap.add_argument("--name", default=None, help="결과 목록 이름 (다른 폴더를 검사할 때 섞이지 않게)")
    ap.add_argument("--rescan", action="store_true", help="이미 찾은 문서만 다시 검사")
    ap.add_argument("--ocr", choices=["rapid", "tesseract"], default="rapid")
    ap.add_argument("--want", type=int, default=0, help="목록이 이만큼 모이면 멈춤 (0이면 폴더 전부)")
    ap.add_argument("--no-logic", action="store_true", help="논리 맞추기 끔: 사진마다 금액이나 날짜 하나만 바꿈")
    ap.add_argument("--no-form", action="store_true", help="진료비 영수증 서식 규칙(한국어 OCR) 끔")
    ap.add_argument("--strict", action="store_true", help="관계를 못 찾은 문서는 건너뜀 (칸 하나만 바꾸는 위조를 안 만듦)")
    a = ap.parse_args()
    global OCR, FALLBACK, USE_FORM, LOGIC
    OCR, FALLBACK, USE_FORM, LOGIC = a.ocr, not a.strict, not a.no_form, not a.no_logic
    global SCAN, LIST
    if a.name:
        SCAN, LIST = OUT / f"_스캔_{a.name}.csv", OUT / f"원본목록_{a.name}.csv"
    if a.rescan:
        return rescan()
    import forge_v2 as F
    done = {r["file"] for r in csv.DictReader(open(SCAN, encoding="utf-8-sig"))} if SCAN.exists() else set()
    files = sorted(p for p in (ROOT / a.src).expanduser().iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    t0 = time.time()
    new_scan, new_list = not SCAN.exists(), not LIST.exists()
    n = 0
    have = sum(1 for _ in csv.DictReader(open(LIST, encoding="utf-8-sig"))) if LIST.exists() else 0
    with open(SCAN, "a", encoding="utf-8-sig", newline="") as fs, open(LIST, "a", encoding="utf-8-sig", newline="") as fl:
        ws = csv.DictWriter(fs, fieldnames=["file", "규칙"])
        wl = csv.DictWriter(fl, fieldnames=FIELDS, restval="")
        if new_scan:
            ws.writeheader()
        if new_list:
            wl.writeheader()
        for p in files:
            if p.name in done:
                continue
            if a.limit and time.time() - t0 > a.limit:
                break
            if a.want and have >= a.want:
                break
            try:
                r = scan_one(p, F.stroke_only)
            except Exception:
                r = None
            if r:
                wl.writerow(r)
                have += 1
            ws.writerow({"file": p.name, "규칙": r["규칙"] if r else ""})
            fs.flush(); fl.flush()
            n += 1
    rows = list(csv.DictReader(open(LIST, encoding="utf-8-sig")))
    from collections import Counter
    print(f"검사 {len(done) + n}/{len(files)}, 만든 목록 {len(rows)}:", dict(Counter(r["규칙"].split()[0] for r in rows)))


if __name__ == "__main__":
    main()
