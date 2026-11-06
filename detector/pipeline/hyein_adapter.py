"""AI 판독값(ai_judge 형식) → 혜인 님 Reading → 혜인 님 규칙(check · focus) 판정.

혜인 님 규칙은 이미지를 보지 않고 Reading(칸 이름 → 값)만 본다 (logic_detector/logic/reading.py).
그래서 판독기를 AI(OpenRouter Qwen)로 바꿔도 규칙은 그대로 쓴다. 혜인 님 코드는 고치지 않는다.
칸 이름 대응: 행 이름은 혜인 님 이름표(read_receipt.ROW_ORDER)를 그대로 쓰고, 나머지는 아래 표.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from logic_detector.logic import Reading, check, focus, verdict  # noqa: E402
from logic_detector.logic.read_receipt import ROW_ORDER  # noqa: E402
from logic_detector.logic.rules import GRADES  # noqa: E402

COLS = {"copay": "self", "nhis": "nhis", "full": "full", "uninsured": "nonpay"}
TOTALS = {"t1": "totals.t1", "t2": "totals.t2", "t3": "totals.t3", "t4": "totals.t4", "t5": "totals.cap",
          "t6": "grand_total", "t7": "nhis_total", "t8": "patient_total", "t9": "already_paid", "t10": "to_pay"}
PAID = {"card": "paid.card", "cash_receipt": "paid.cash_receipt", "cash": "paid.cash", "paid_sum": "paid.total",
        "unpaid": "unpaid"}
INPATIENT_ROWS = {"room", "room_1", "room_23", "room_4", "meal"}
VERDICT = {"위변조 의심": "의심", "확인 필요": "확인필요", "정상": "정상"}
AMBIGUOUS = {"행위료", "행위", "약품비", "약품", "기타", "합계"}      # 앞말 없이는 어느 행인지 모르는 이름


def norm(s):
    return re.sub(r"[\s\-·ㆍ()（）]", "", str(s))


ROW_OF = {}
for _key, _labels in ROW_ORDER:
    for _lab in _labels:
        if norm(_lab) not in AMBIGUOUS:
            ROW_OF.setdefault(norm(_lab), _key)


def row_key(name):
    """판독값 행 이름(공식 표기, 예: '주사료-행위료') → 혜인 님 행 이름(inj_act). 모르면 None."""
    n = norm(name)
    if n in ROW_OF:
        return ROW_OF[n]
    if n.startswith("입원료"):                                   # '입원료-1인실' 처럼 앞에 입원료가 붙은 이름
        return ROW_OF.get(n[len("입원료"):], "room")
    return None


def to_reading(v, doc):
    """판독값(readings_check.load 형식) → Reading. 돌려줌: (Reading, 이름을 모르는 행 목록)."""
    inst, f, unknown = v.get("institution") or {}, {}, []
    f.update({"institution.type": inst.get("class"), "institution.name": inst.get("name"),
              "institution.bizno": inst.get("biz_no"), "institution.address": inst.get("address"),
              "patient.rrn": (v.get("patient") or {}).get("reg_no")})
    if doc == "영수증":
        rc = v.get("receipt") or {}
        f.update({"visit.start": rc.get("visit_start"), "visit.end": rc.get("visit_end"),
                  "issue_date": rc.get("issue_date"), "receipt_no": rc.get("receipt_no")})
        for name, cols in (rc.get("rows") or {}).items():
            key = row_key(name)
            if key is None:
                unknown.append(name)
                continue
            for c, hc in COLS.items():
                if isinstance((cols or {}).get(c), int):
                    f[f"rows.{key}.{hc}"] = (f.get(f"rows.{key}.{hc}") or 0) + cols[c]
        for k, path in TOTALS.items():
            f[path] = (rc.get("totals") or {}).get(k)
        for k, path in PAID.items():
            f[path] = (rc.get("payment") or {}).get(k)
        f["rows_complete"] = any(k.startswith("rows.") for k in f) and not unknown
        f["visit.inpatient"] = any(f.get(f"rows.{r}.{c}") for r in INPATIENT_ROWS for c in COLS.values())
        return Reading("receipt", f), unknown
    dx = v.get("diagnosis") or {}
    codes, names = dx.get("codes") or [], dx.get("names") or {}
    f["diagnoses"] = [{"code": c, "name": names.get(c, ""), "main": i == 0} for i, c in enumerate(codes)] or None
    f.update({"dates.onset": dx.get("onset_date"), "dates.diagnosis": dx.get("diag_date"),
              "dates.issue": dx.get("issue_date")})
    return Reading("diagnosis", f), unknown


def judge(v, doc):
    """판독값 → 혜인 님 규칙 판정 {판정(의심/확인필요/정상), findings, 모르는_행}."""
    r, unknown = to_reading(v, doc)
    found = sorted(check(r) + focus.check(r), key=lambda x: GRADES.index(x["grade"]))
    return {"판정": VERDICT[verdict(found)], "findings": found, "모르는_행": unknown}


def reason(found, n=3):
    """혜인 님 findings → 근거 한 줄 (등급 · 규칙 · 제목)."""
    return " / ".join(f"[{x['grade']}] {x['rule']} {x.get('title', '')}".strip() for x in found[:n])
