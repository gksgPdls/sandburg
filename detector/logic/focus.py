"""
주목 규칙: 보험금이 걸린 칸을 짚고(주목), 같은 서류 안의 다른 칸과 모순이면 판정한다. 표: data/focus_rules.yaml

    focus(reading)  → [{id, name, what, fields, boxes, why, sources, checks}]   보험금이 걸린 칸 (판정 아님, '여기를 보라')
    check(reading)  → Finding 목록 (rules.check와 같은 형식, rule = "F:<검사 id>")

주목은 why_source(약관·표준약관·KCD)가 있는 항목만, 판정은 checks.status 가 '구현'이고 source 가 있는 검사만 한다.
진단서 소견(opinion)은 판독기가 '치료 내용 및 향후 치료에 대한 소견' 칸에서 읽은 글.
"""
import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import ocr as O
from .rules import GRADES, Checker

DATA = Path(__file__).resolve().parent / "data" / "focus_rules.yaml"

DEPTH_DIGIT = {"1": 1, "2": 2, "3": 3, "5": 1, "6": 2, "7": 3}   # 화상 T20~T25 4번째 자리 (.5~.7 = 부식)
DEEP_2ND = re.compile(r"심재성|심부\s*2\s*도|깊은\s*2\s*도")
SPRAIN = re.compile(r"염좌|긴장|타박|골절\s*(소견|은|이)?\s*(없|아님)")
NO_FX = re.compile(r"골절\s*(소견|은|이)?\s*(없|아님)")
CIS = re.compile(r"제자리|상피내|경계성")
INJURY_WORDS = re.compile(r"넘어져|넘어지며|다쳐|사고|열탕|추락|부딪")
PERIOD = re.compile(r"향후[^.]*?(\d+)\s*(주|일|개월)")


@lru_cache(maxsize=1)
def table():
    return yaml.safe_load(DATA.read_text(encoding="utf-8"))


def _item(item_id):
    return next(i for i in table()["items"] if i["id"] == item_id)


def _check_spec(item_id, check_id):
    return next(c for c in _item(item_id)["checks"] if c["id"] == check_id)


def _code(code):
    return (code or "").replace(".", "").upper()


def kcd_match(code, specs):
    """'T23.24'가 ['T20-T25', 'S09.9', ...] 중 하나에 드는지. 범위는 앞 3자리로 비교, 나머지는 앞자리 일치."""
    c = _code(code)
    for s in specs:
        if "-" in s:
            lo, hi = (_code(x) for x in s.split("-"))
            if len(c) >= 3 and lo <= c[:3] <= hi:
                return True
        elif c.startswith(_code(s)):
            return True
    return False


def _diagnoses(r):
    return r.get("diagnoses") or []


def _main_code(r):
    dx = _diagnoses(r)
    main = next((d for d in dx if d.get("main")), dx[0] if dx else None)
    return _code(main["code"]) if main else ""


def _nonpay_fields(r):
    return [k for k, f in r.fields.items() if k.startswith("rows.") and k.endswith(".nonpay") and (f["value"] or 0) > 0]


def triggered(item, r):
    if item["doc"] != r.doc_type:
        return False
    t = item["trigger"]
    if t.get("kcd") and any(kcd_match(d.get("code"), t["kcd"]) for d in _diagnoses(r)):
        return True
    if t.get("words"):
        text = " ".join([r.get("opinion") or ""] + [d.get("name") or "" for d in _diagnoses(r)])
        if any(w in text for w in t["words"]):
            return True
    for f in t.get("fields", []):
        if (f == "rows.*.nonpay" and _nonpay_fields(r)) or (f != "rows.*.nonpay" and r.has(f)):
            return True
    return False


def _short(src):
    return {k: src[k] for k in ("type", "name", "clause", "quote", "url", "verified") if k in src}


def focus(r):
    out = []
    for item in table()["items"]:
        if not triggered(item, r):
            continue
        fields = []
        for f in item["focus"]:
            fields += _nonpay_fields(r) if f["field"] == "rows.*.nonpay" else [f["field"]]
        out.append({
            "id": item["id"], "name": item["name"], "what": [f["what"] for f in item["focus"]],
            "fields": [f for f in fields if r.has(f)], "boxes": [b for b in (r.box(f) for f in fields) if b],
            "why": item["why"], "sources": [_short(s) for s in item["why_source"]],
            "checks": [{"id": c["id"], "status": c["status"], "rule": c["rule"]} for c in item["checks"]],
        })
    return out


def _add(c, item_id, check_id, fields, title, grade=None, expected=None, actual=None):
    spec = _check_spec(item_id, check_id)
    src = spec["source"]
    c.add(f"F:{check_id}", grade or spec["grade"], fields, title,
          detail=spec["rule"], expected=expected, actual=actual,
          source=f"{src['name']} — “{src['quote']}” ({src['url']})", caveat=spec.get("caveat", ""))


# ── 검사 (status: 구현) ──────────────────────────────────────────────────────

def burn(c):
    r, code, op = c.r, _main_code(c.r), c.r.get("opinion") or ""
    if not (op and kcd_match(code, ["T20-T25"]) and len(code) >= 4 and code[3] in DEPTH_DIGIT):
        return
    depth = DEPTH_DIGIT[code[3]]
    said = {int(d) for d in re.findall(r"([123])\s*도", op)} | ({2} if DEEP_2ND.search(op) or "표재성" in op else set())
    if said and depth not in said:
        _add(c, "BURN", "BURN-DEPTH", ["diagnoses", "opinion"], "화상 깊이: 소견과 질병분류기호가 다름",
             expected=f"소견 {'/'.join(f'{d}도' for d in sorted(said))}", actual=f"{code[:3]}.{code[3:]} = {depth}도")
    m = PERIOD.search(op)
    if DEEP_2ND.search(op) and m:
        n, unit = int(m.group(1)), m.group(2)
        weeks = n / 7 if unit == "일" else n * 4.3 if unit == "개월" else n
        if weeks <= 2:
            _add(c, "BURN", "BURN-DEEP-SHORT", ["opinion"], "심재성 2도인데 향후 치료기간이 짧음",
                 expected="3주 이상 (질병관리청)", actual=m.group(0))


def fracture(c):
    code, op = _main_code(c.r), c.r.get("opinion") or ""
    if not op:
        return
    fx_codes = _item("FRACTURE")["trigger"]["kcd"]
    if kcd_match(code, fx_codes) and SPRAIN.search(op):
        _add(c, "FRACTURE", "FX-KIND", ["diagnoses", "opinion"], "골절 코드인데 소견은 골절이 아님",
             expected="소견도 골절", actual=f"{code} / 소견 '{SPRAIN.search(op).group(0)}'")
    elif re.fullmatch(r"S\d3\w*", code) and "골절" in op and not NO_FX.search(op):
        _add(c, "FRACTURE", "FX-KIND", ["diagnoses", "opinion"], "염좌·탈구 코드인데 소견은 골절",
             expected="소견도 염좌·탈구", actual=f"{code} / 소견 '골절'")


def cancer(c):
    r, code, op = c.r, _main_code(c.r), c.r.get("opinion") or ""
    if op and code.startswith("C") and CIS.search(op):
        _add(c, "CANCER", "CA-CIS", ["diagnoses", "opinion"], "악성 신생물(C) 코드인데 소견의 조직검사 결과는 제자리암·경계성",
             expected="D00~D09 / D37~D48", actual=f"{code} / 소견 '{CIS.search(op).group(0)}'")
    diag = r.date("dates.diagnosis")
    report = next((m for m in O.DATE_RE.finditer(op) if re.match(r"\s*일?\s*(조직검사\s*)?결과\s*보고", op[m.end():])), None)
    if diag and report and kcd_match(code, ["C00-C97", "D00-D09", "D37-D48"]):
        rep = O.find_dates(report.group(0))[0]
        if diag != rep:
            _add(c, "CANCER", "CA-DATE", ["dates.diagnosis", "opinion"], "진단 연월일이 조직검사 결과보고일과 다름",
                 grade="참고", expected=f"결과보고일 {rep}", actual=f"진단 연월일 {diag}")


def admission(c):
    r, op = c.r, c.r.get("opinion") or ""
    if "입원" not in op:
        return
    ds = O.find_dates(op)
    if len(ds) < 2 or not ("퇴원" in op or "까지" in op):
        return
    for key, said in (("dates.admit", ds[0]), ("dates.discharge", ds[-1])):
        got = r.date(key)
        if said and got and said != got:
            _add(c, "ADMISSION", "ADM-OPINION", [key, "opinion"],
                 f"{'입원일' if key == 'dates.admit' else '퇴원일'} 칸과 소견의 날짜가 다름",
                 expected=f"소견 {said}", actual=f"칸 {got}")


def injury(c):
    code, op = _main_code(c.r), c.r.get("opinion") or ""
    m = INJURY_WORDS.search(op)
    if m and code and code[0] not in "ST":
        _add(c, "INJURY", "INJ-CODE", ["diagnoses", "opinion"], "소견은 사고(상해) 경위인데 질병 코드",
             expected="S00~T98 (손상)", actual=f"{code} / 소견 '{m.group(0)}'")


DIAGNOSIS_CHECKS = [burn, fracture, cancer, admission, injury]


def check(r):
    if r.doc_type != "diagnosis":
        return []
    c = Checker(r)
    for fn in DIAGNOSIS_CHECKS:
        fn(c)
    c.found.sort(key=lambda f: GRADES.index(f["grade"]))
    return c.found
