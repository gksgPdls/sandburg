"""[역할] 계산서 금액 계산: 본인/공단 분할, 외래 100원 미만 절사, ①~⑪ 산식, 검산
[쓰는 곳] generate.py, tools/verify_cases.py
[입력 → 출력] 처방 줄 목록 → 계산서 금액표

진료비 계산서·영수증(요양급여규칙 별지 제6호서식, 2024.7.18 개정) 금액 계산 (외래·입원).

열(column) — 공식 서식 기준 4열:
  self    급여 · 일부 본인부담 · 본인부담금  → 합계 ①
  nhis    급여 · 일부 본인부담 · 공단부담금  → 합계 ②
  full    급여 · 전액 본인부담             → 합계 ③  (생성하지 않음, 항상 0)
  nonpay  비급여                           → 합계 ④  (제증명수수료만)
  ⑤ 상한액 초과금 (생성하지 않음, 항상 0)

금액산정내용 (서식에 인쇄된 산식 그대로):
  ⑥ 진료비 총액   = ① + ② + ③ + ④
  ⑦ 공단부담 총액 = ② + ⑤
  ⑧ 환자부담 총액 = (① − ⑤) + ③ + ④
  ⑨ 이미 납부한 금액 (중간 정산 없이 한 번에 수납하므로 0)
  ⑩ 납부할 금액   = ⑧ − ⑨
  ⑪ 납부한 금액   = 카드 + 현금영수증 + 현금 = ⑩

본인부담금
  계산서 행·본인부담률별로 급여 금액을 합한 뒤 부담률을 곱하고 원 단위 반올림, 나머지는 공단부담금
  (실제 영수증에서 행마다 본인부담금 = 행 급여합 × 부담률 로 맞아떨어짐)
  [법령] 외래 본인부담금 100원 미만 제외 — 국민건강보험법 시행령 [별표 2] 제1호
         "…금액(100원 미만은 제외한다)을 부담한다. 다만, 입원진료의 경우에는 100원 미만의 금액도 부담한다."
         → 외래: ① 합계를 100원 미만 절사, 절사분은 ② 공단부담에 더함
           예) 본인부담 3,948+2,133=6,081 → ① 6,000, ② 14,189+81=14,270 (실제 영수증과 일치)
         → 입원: 절사 없음
"""
from decimal import ROUND_HALF_UP, Decimal

COLS = ["self", "nhis", "full", "nonpay"]


def round_won(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def floor_to(x: int, unit: int) -> int:
    return x // unit * unit


def compute(lines: list, outpatient: bool) -> dict:
    """처방 줄 [{row, amount, rate, cover}] → 행별 금액표 + 합계 + 금액산정내용.
    cover: "part" 급여 일부본인부담 (기본) / "nonpay" 비급여 (전액 환자 부담, ④)."""
    covered: dict = {}
    rows: dict = {}
    for ln in lines:
        if ln.get("cover") == "nonpay":
            rows.setdefault(ln["row"], {c: 0 for c in COLS})["nonpay"] += ln["amount"]
            continue
        key = (ln["row"], ln["rate"])
        covered[key] = covered.get(key, 0) + ln["amount"]
    for (row, rate), total in covered.items():
        r = rows.setdefault(row, {c: 0 for c in COLS})
        self_amt = round_won(total * rate)
        r["self"] += self_amt
        r["nhis"] += total - self_amt

    raw1 = sum(r["self"] for r in rows.values())
    raw2 = sum(r["nhis"] for r in rows.values())
    t1 = floor_to(raw1, 100) if outpatient else raw1   # [법령] 외래 본인부담금 100원 미만 제외
    t2 = raw2 + (raw1 - t1)                            # 절사분은 공단 부담
    t3 = t5 = 0
    t4 = sum(r["nonpay"] for r in rows.values())
    patient_total = (t1 - t5) + t3 + t4
    return {
        "rows": rows,
        "totals": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
        "grand_total": t1 + t2 + t3 + t4,        # ⑥
        "nhis_total": t2 + t5,                   # ⑦
        "patient_total": patient_total,          # ⑧
        "already_paid": 0,                       # ⑨
        "to_pay": patient_total,                 # ⑩
        "paid_total": patient_total,             # ⑪ 합계
        "outpatient": outpatient,
    }


def verify(r: dict, lines: list) -> list:
    """생성된 금액을 산식대로 다시 계산해 대조. 문제 목록을 반환 (빈 리스트면 통과)."""
    errs = []
    t = r["totals"]
    rows = list(r["rows"].values())
    raw1 = sum(x["self"] for x in rows)
    raw2 = sum(x["nhis"] for x in rows)
    want1 = floor_to(raw1, 100) if r["outpatient"] else raw1
    if t["t1"] != want1 or t["t1"] + t["t2"] != raw1 + raw2:
        errs.append("①② 합계·외래 100원 미만 절사/이월 불일치")
    if t["t4"] != sum(x["nonpay"] for x in rows):
        errs.append("④ 비급여 합계 불일치")
    if r["grand_total"] != t["t1"] + t["t2"] + t["t3"] + t["t4"]:
        errs.append("⑥ ≠ ①+②+③+④")
    if r["grand_total"] != sum(ln["amount"] for ln in lines):
        errs.append("⑥ ≠ 처방 줄 금액 합")
    if any(ln["amount"] != ln["unit_price"] * ln["qty"] * ln["days"] for ln in lines):
        errs.append("처방 줄 금액 ≠ 단가 × 수량 × 일수")
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
