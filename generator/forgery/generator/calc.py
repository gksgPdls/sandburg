"""진료비 계산서·영수증(요양급여규칙 별지 제6호서식, 2024.7.18 개정) 금액 계산.

열(column) — 공식 서식 기준 4열:
  self    급여 · 일부 본인부담 · 본인부담금  → 합계 ①
  nhis    급여 · 일부 본인부담 · 공단부담금  → 합계 ②
  full    급여 · 전액 본인부담             → 합계 ③  (현재 생성하지 않음)
  nonpay  비급여                           → 합계 ④
  ⑤ 상한액 초과금 (현재 항상 0)

금액산정내용 (서식에 인쇄된 산식 그대로):
  ⑥ 진료비 총액   = ① + ② + ③ + ④
  ⑦ 공단부담 총액 = ② + ⑤
  ⑧ 환자부담 총액 = (① − ⑤) + ③ + ④
  ⑨ 이미 납부한 금액
  ⑩ 납부할 금액   = ⑧ − ⑨
  ⑪ 납부한 금액   = 카드 + 현금영수증 + 현금

절사 규칙
  [법령] ① 본인부담금 100원 미만 제외 — 외래만
         국민건강보험법 시행령 [별표 2] 제1호: "…금액(100원 미만은 제외한다)을 부담한다.
         다만, 입원진료의 경우에는 100원 미만의 금액도 부담한다."
         → 외래: ① 을 100원 미만 절사, 절사분은 공단이 부담하므로 ② 에 더함
           예) 본인부담 3,948+2,133=6,081 → ① 6,000, ② 14,189+81=14,270 (실제 영수증과 일치)
         → 입원: 절사 없음
"""
from decimal import Decimal, ROUND_HALF_UP

COLS = ["self", "nhis", "full", "nonpay"]


def round_won(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def floor_to(x: int, unit: int) -> int:
    return x // unit * unit


def split_covered(total: int, rate: float) -> tuple:
    """급여 금액을 본인부담금 / 공단부담금으로 나눔 (본인부담금은 원 단위 반올림)."""
    self_amt = round_won(total * rate)
    return self_amt, total - self_amt


def compute(lines: list, outpatient: bool, prepaid: bool) -> dict:
    """lines: [{row, cover, amount, rate}] → 행별 금액표 + 합계 + 금액산정내용.

    급여(part)는 같은 행·같은 부담률끼리 먼저 합산한 뒤 본인/공단으로 나눔
    (실제 영수증에서 행마다 본인부담금 = 행 급여합 × 부담률 로 맞아떨어짐).
    """
    rows: dict = {}
    covered: dict = {}
    for ln in lines:
        r = rows.setdefault(ln["row"], {c: 0 for c in COLS})
        if ln["cover"] == "part":
            key = (ln["row"], ln["rate"])
            covered[key] = covered.get(key, 0) + ln["amount"]
        elif ln["cover"] == "full":
            r["full"] += ln["amount"]
        else:  # nonpay
            r["nonpay"] += ln["amount"]
    for (row, rate), total in covered.items():
        s, n = split_covered(total, rate)
        rows[row]["self"] += s
        rows[row]["nhis"] += n

    t1, t2, t3, t4 = (sum(r[c] for r in rows.values()) for c in COLS)
    t5 = 0  # 상한액 초과금

    if outpatient:  # [법령] 외래 본인부담금 100원 미만 제외 → 공단 부담
        cut = t1 - floor_to(t1, 100)
        t1, t2 = t1 - cut, t2 + cut
    patient_total = (t1 - t5) + t3 + t4

    already_paid = patient_total if prepaid else 0
    return {
        "rows": rows,
        "totals": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
        "grand_total": t1 + t2 + t3 + t4,        # ⑥
        "nhis_total": t2 + t5,                   # ⑦
        "patient_total": patient_total,          # ⑧
        "already_paid": already_paid,            # ⑨
        "to_pay": patient_total - already_paid,  # ⑩
        "paid_total": patient_total,             # ⑪ 합계
        "lines_sum": sum(ln["amount"] for ln in lines),
        "outpatient": outpatient,
    }


def verify(r: dict) -> list:
    """생성된 금액이 규칙대로인지 다시 계산. 문제 목록을 반환 (빈 리스트면 통과)."""
    errs = []
    t = r["totals"]
    rows = list(r["rows"].values())
    raw1 = sum(x["self"] for x in rows)
    raw2 = sum(x["nhis"] for x in rows)
    if r["outpatient"]:
        if t["t1"] != floor_to(raw1, 100) or t["t1"] + t["t2"] != raw1 + raw2:
            errs.append("①② 외래 100원 미만 절사/이월 불일치")
    elif t["t1"] != raw1 or t["t2"] != raw2:
        errs.append("①② 합계 불일치")
    if t["t3"] != sum(x["full"] for x in rows) or t["t4"] != sum(x["nonpay"] for x in rows):
        errs.append("③④ 합계 불일치")
    if r["grand_total"] != t["t1"] + t["t2"] + t["t3"] + t["t4"]:
        errs.append("⑥ ≠ ①+②+③+④")
    if r["grand_total"] != r["lines_sum"]:
        errs.append("⑥ ≠ 세부내역 금액 합")
    if r["nhis_total"] != t["t2"] + t["t5"]:
        errs.append("⑦ ≠ ②+⑤")
    if r["patient_total"] != (t["t1"] - t["t5"]) + t["t3"] + t["t4"]:
        errs.append("⑧ ≠ (①-⑤)+③+④")
    if r["to_pay"] != r["patient_total"] - r["already_paid"]:
        errs.append("⑩ ≠ ⑧-⑨")
    pay = r.get("payment")
    if pay and pay["card"] + pay["cash_receipt"] + pay["cash"] != r["paid_total"]:
        errs.append("⑪ 결제수단 합 불일치")
    return errs
