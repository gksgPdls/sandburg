"""
로직 규칙: Reading(이미지에서 읽은 칸 값) → Finding 목록. 원형은 이전 합성 서류 생성기/tools/verify_cases.py (생성기 검증기).

확실도 등급
  확정  서식에 인쇄된 산식·코드 체계가 깨짐 (위변조 의심)
  강함  공식 단가·부담률·상병마스터 규칙과 안 맞음 (위변조 의심)
  참고  정상 서류도 어긋날 수 있음 (확인 필요). caveat에 왜 위변조 의심까지 안 가는지 적는다
보수 원칙 (2026-11-06): 실제 병원 서류가 정상적으로 어길 수 있는 규칙, 공식 데이터가 다 갖춰지지 않은 규칙은
'위변조 의심'으로 판정하지 않고 '참고'로만 알린다. 무엇을 적용·제외했는지는 logic_detector/logic/규칙_적용기준.md.
규칙이 쓰는 칸 중 하나라도 판독 불확실(uncertain)이면 그 규칙 결과는 '참고'로 낮춘다 (잘못 읽은 값이 오탐이 되지 않게).

규칙 ID
  A1 서식 산식  A2 열 합계·외래 100원 미만 절사  A3 납부 금액
  B1 행별 본인부담률  B2 진찰료 공식 단가  B4 입원료  B6 야간·공휴·토요 가산과 진료일
  C1 상병코드 실존  C2 코드-병명  C3 성별·나이 제한  C4 주상병 사용·주부 중복
  D1 날짜 순서  D2 영수증번호 연월  D3 사업자등록번호  D4 주민등록번호
"""
import re
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache

from . import refdata as R

GRADES = ("확정", "강함", "참고")
RATE_TOL = 10      # 행 본인부담금 반올림 오차 허용 (원). 생성기는 행·부담률별 원 단위 반올림, 실제 EMR은 줄마다 반올림해 몇 원 차이 남

# 외래 본인부담률 — 국민건강보험법 시행령 [별표 2] 제1호. 의원·병원·종합병원 동 지역 값은 rules_data.yaml,
# 읍·면 지역(병원 35%, 종합병원 45%)과 상급종합병원(60%)은 아직 데이터셋으로 확인 전이라 여기 둔다.
EXTRA_OUT_RATES = {"hospital": [0.35], "general": [0.45], "tertiary": [0.6]}
# 부담률이 행마다 다를 수 있어 B1에서 빼는 행 (병실·식대는 따로, 비급여·정액·포괄 행)
# 환자 구분에 이 말이 있으면 건강보험 일반 부담률이 아닐 수 있음 (B1 결과를 '참고'로)
SPECIAL_PATIENT = ("의료급여", "급여1종", "급여2종", "산재", "자보", "자동차", "특례", "차상위", "보훈")
B1_SKIP_ROWS = {"room", "room_diff", "room_1", "room_23", "room_4", "meal", "cert", "selective", "senior", "fixed_ltc", "fixed_hospice",
                "drg", "etc", "prosth"}


# '참고'로만 알리는 이유 (화면·문서에 그대로 나감)
CAVEAT = {
    "A1_to_pay": "실제 영수증에는 감면·할인·미수금 줄이 따로 있어 ⑩이 ⑧-⑨와 다를 수 있음",
    "A1_paid": "실제 영수증에는 계좌이체·포인트·미수금 등 다른 수납 방법 줄이 있을 수 있음",
    "A2_floor": "병원마다 반올림·10원 절사 등 처리 방식이 다르고 65세 이상 정액제 등 예외가 있음",
    "A3": "선수금·예치금·과오납 환불이 있으면 정상 서류도 납부한 금액이 더 클 수 있음",
    "B1": "선별급여(50·80·90%), 65세 이상 정액제, 6세 미만 경감, 산정특례 부담률은 아직 규칙 데이터에 없음",
    "B2": "6세 미만 소아 가산, 상급종합 진찰료, 진찰료 칸에 다른 기본진료료를 합쳐 적는 양식은 아직 규칙 데이터에 없음",
    "B4": "장기입원 체감, 간호간병통합서비스, 소아 가산, 외박 일수는 아직 반영하지 않음",
    "B6": "임시공휴일·가산 표시 방식(야간·공휴 칸 공용 등)이 병원마다 다름",
    "D1_inpatient": "입원 중간 정산 영수증은 발행일이 진료기간 끝보다 이를 수 있음",
    "D2": "영수증번호 체계는 EMR마다 달라 앞자리가 날짜가 아닐 수 있고, 재발행·후일 수납이면 날짜가 다름",
    "D3_tax": "병원 이전·세무서 신설·분할로 사업자번호 앞 3자리가 지금 주소 관할과 다를 수 있음",
    "D4_check": "주민번호 변경·외국인등록번호는 2020년 이전 출생이어도 검증번호가 안 맞을 수 있음",
    "C1_old": "2021년 이전 진단은 그때의 KCD 버전 코드를 썼을 수 있음 (지금 마스터는 2025.9)",
    "C1_U": "U코드(한의·특수목적)는 기관별 사용이 달라 마스터에 없을 수 있음",
    "C4": "주상병 사용구분은 청구 심사 기준이라 진단서에선 안 지켜지기도 하고, 좌우 등으로 같은 코드를 두 번 쓰기도 함",
}


class Checker:
    """규칙 하나 실행 중의 결과 모음. add()가 판독 불확실 칸이 있으면 등급을 '참고'로 낮춘다."""

    def __init__(self, reading):
        self.r = reading
        self.found = []

    def add(self, rule, grade, fields, title, detail="", expected=None, actual=None, source="", caveat=""):
        unsure = [f for f in fields if self.r.uncertain(f)]
        self.found.append({
            "rule": rule, "grade": "참고" if unsure else grade, "title": title, "detail": detail,
            "fields": list(fields), "boxes": [b for b in (self.r.box(f) for f in fields) if b],
            "expected": expected, "actual": actual, "source": source,
            **({"caveat": caveat} if caveat else {}),
            **({"uncertain_fields": unsure, "original_grade": grade} if unsure else {}),
        })


# ── 진료비 계산서·영수증 ─────────────────────────────────────────────────────

def _formulas(form):
    """(규칙 설명, 결과 칸, [(부호, 칸)]) — 서식에 인쇄된 금액산정내용 산식."""
    if form == "old":
        return [("⑦ 진료비 총액 = ①+②+③+④+⑤", "grand_total",
                 [(1, "totals.t1"), (1, "totals.t2"), (1, "totals.t3"), (1, "totals.t4"), (1, "totals.t5")]),
                ("⑧ 환자부담 총액 = ①-⑥+③+④+⑤", "patient_total",
                 [(1, "totals.t1"), (-1, "totals.cap"), (1, "totals.t3"), (1, "totals.t4"), (1, "totals.t5")]),
                ("⑩ 납부할 금액 = ⑧-⑨", "to_pay", [(1, "patient_total"), (-1, "already_paid")])]
    return [("⑥ 진료비 총액 = ①+②+③+④", "grand_total",
             [(1, "totals.t1"), (1, "totals.t2"), (1, "totals.t3"), (1, "totals.t4")]),
            ("⑦ 공단부담 총액 = ②+⑤", "nhis_total", [(1, "totals.t2"), (1, "totals.cap")]),
            ("⑧ 환자부담 총액 = (①-⑤)+③+④", "patient_total",
             [(1, "totals.t1"), (-1, "totals.cap"), (1, "totals.t3"), (1, "totals.t4")]),
            ("⑩ 납부할 금액 = ⑧-⑨", "to_pay", [(1, "patient_total"), (-1, "already_paid")])]


def a1_formulas(c):
    r = c.r
    for title, out, terms in _formulas(r.get("form", "2024")):
        fields = [f for _, f in terms] + [out]
        if not r.has(*fields):
            continue
        want = sum(s * r.get(f) for s, f in terms)
        if r.get(out) != want:
            pay_line = out == "to_pay"
            c.add("A1", "참고" if pay_line else "확정", fields, f"산식 불일치: {title}", expected=want, actual=r.get(out),
                  source="요양급여규칙 별지 제6호서식 금액산정내용", caveat=CAVEAT["A1_to_pay"] if pay_line else "")
    pay = ["paid.card", "paid.cash_receipt", "paid.cash", "paid.total"]
    if r.has(*pay):
        want = r.get("paid.card") + r.get("paid.cash_receipt") + r.get("paid.cash")
        if r.get("paid.total") != want:
            c.add("A1", "참고", pay, "⑪ 납부한 금액 합계 ≠ 카드+현금영수증+현금", expected=want, actual=r.get("paid.total"),
                  source="요양급여규칙 별지 제6호서식", caveat=CAVEAT["A1_paid"])


def a2_column_sums(c):
    r = c.r
    if not r.get("rows_complete"):
        return
    rows = r.rows()
    # 열: 개정 서식 본인부담금·공단부담금·전액본인부담·비급여 / 예전 서식은 비급여 대신 선택진료료·선택진료료 이외
    cols = ["self", "nhis", "full", "select", "nonselect"] if r.get("form") == "old" else ["self", "nhis", "full", "nonpay"]
    col_sum = {col: sum(v.get(col, 0) for v in rows.values()) for col in cols}
    row_fields = lambda col: [f"rows.{k}.{col}" for k, v in rows.items() if col in v]   # noqa: E731
    for i, col in enumerate(cols[2:], start=3):
        tf = f"totals.t{i}"
        if r.has(tf) and r.get(tf) != col_sum[col]:
            c.add("A2", "확정", row_fields(col) + [tf], f"열 합계 {i}번이 행 합과 다름 ({col})", expected=col_sum[col], actual=r.get(tf),
                  source="요양급여규칙 별지 제6호서식")
    if not r.has("totals.t1", "totals.t2"):
        return
    t1, t2, raw1, raw2 = r.get("totals.t1"), r.get("totals.t2"), col_sum["self"], col_sum["nhis"]
    fields = row_fields("self") + row_fields("nhis") + ["totals.t1", "totals.t2"]
    if t1 + t2 != raw1 + raw2:
        c.add("A2", "확정", fields, "급여 합계(①+②) ≠ 행별 본인부담금+공단부담금 합", expected=raw1 + raw2, actual=t1 + t2,
              source="요양급여규칙 별지 제6호서식")
    allowed = {raw1, raw1 // 10 * 10, round(raw1, -1)}
    if not r.get("visit.inpatient"):
        allowed |= {raw1 // 100 * 100, round(raw1, -2)}
    if t1 not in allowed:
        c.add("A2", "참고", row_fields("self") + ["totals.t1"], "① 본인부담금 합계가 행 합(또는 외래 100원 미만 절사)과 다름",
              detail=f"행 합 {raw1:,}", expected=sorted(allowed), actual=t1,
              source="국민건강보험법 시행령 [별표 2] 제1호 (외래 100원 미만 제외, 입원은 절사 없음)", caveat=CAVEAT["A2_floor"])


def a3_paid(c):
    r = c.r
    if r.has("paid.total", "to_pay") and r.get("paid.total") > r.get("to_pay"):
        c.add("A3", "참고", ["paid.total", "to_pay"], "⑪ 납부한 금액이 ⑩ 납부할 금액보다 큼",
              expected=f"≤ {r.get('to_pay'):,}", actual=r.get("paid.total"), caveat=CAVEAT["A3"])


def _out_rates(inst_type):
    types = [inst_type] if inst_type else ["clinic", "hospital", "general", "tertiary"]
    rd = R.rules_data()["institution_types"]
    return sorted({x for t in types for x in ([rd[t]["outpatient_rate"]] if t in rd else []) + EXTRA_OUT_RATES.get(t, [])})


def _fits(self_amt, covered, rate):
    return abs(self_amt - round(covered * rate)) <= RATE_TOL


def b1_row_rates(c):
    """행마다 본인부담금 ÷ (본인부담금+공단부담금) = 부담률. 항목 칸 숫자 하나만 바꾸면 합계를 맞춰도 그 행 비율이 깨진다."""
    r = c.r
    inpatient = r.get("visit.inpatient")
    rows = {k: v for k, v in r.rows().items()
            if "self" in v and "nhis" in v and v["self"] + v["nhis"] > 0 and k not in B1_SKIP_ROWS}
    special = any(w in str(r.get("patient.type") or "") for w in SPECIAL_PATIENT)
    src = "국민건강보험법 시행령 [별표 2] (외래 종별 부담률, 입원 20%)"
    if inpatient:
        _b1_inpatient_special(c, r)
    if not rows:
        return
    if inpatient:
        rates = [R.rules_data()["inpatient_rate"]]
    else:
        # 요양기관 종류 체크 칸은 사진에서 잘못 읽는 경우가 많아(실사진 원본 32%) 쓰지 않고, 모든 종별 부담률 중
        # 행들이 가장 많이 맞는 것을 기준으로 삼는다 (체크 칸만 바꾸는 위조는 드묾)
        rates = _out_rates(None)
    best = max(rates, key=lambda q: (sum(_fits(v["self"], v["self"] + v["nhis"], q) for v in rows.values()), -q))
    fit = {k for k, v in rows.items() if _fits(v["self"], v["self"] + v["nhis"], best)}
    if not fit:
        # 공식 부담률이 하나도 안 맞음 → 산정특례·의료급여 등 다른 부담률 환자일 수 있음. 행끼리라도 같은 비율인지 본다
        big = max(rows, key=lambda k: rows[k]["self"] + rows[k]["nhis"])
        best = rows[big]["self"] / (rows[big]["self"] + rows[big]["nhis"])
        fit = {k for k, v in rows.items() if _fits(v["self"], v["self"] + v["nhis"], best)}
        if len(fit) < 2:
            if not special:
                ks = list(rows)
                c.add("B1", "참고", [f"rows.{k}.{x}" for k in ks for x in ("self", "nhis")],
                      "본인부담률이 공식 부담률과 다름 (특례·의료급여가 아니면 조작 의심)",
                      detail=", ".join(f"{k} {rows[k]['self'] / (rows[k]['self'] + rows[k]['nhis']):.3f}" for k in ks),
                      expected=rates, source=src, caveat=CAVEAT["B1"])
            return
    for k, v in rows.items():
        if k in fit:
            continue
        cov = v["self"] + v["nhis"]
        c.add("B1", "참고", [f"rows.{k}.self", f"rows.{k}.nhis"],
              f"'{k}' 행 본인부담률이 다른 행·공식 부담률과 다름",
              detail=f"이 행 {v['self'] / cov:.3f}, 나머지 행 {best:.3f}", expected=round(cov * best), actual=v["self"],
              source=src, caveat=CAVEAT["B1"])


def _b1_inpatient_special(c, r):
    rd = R.rules_data()
    checks = [("meal", [rd["meal_rate"]]), ("room_4", [0.2]), ("room_23", [0.3, 0.4])]
    for row, rates in checks:
        s, n = r.get(f"rows.{row}.self"), r.get(f"rows.{row}.nhis")
        if s is None or n is None or s + n == 0:
            continue
        if not any(_fits(s, s + n, q) for q in rates):
            c.add("B1", "참고", [f"rows.{row}.self", f"rows.{row}.nhis"], f"'{row}' 행 본인부담률이 규정과 다름",
                  detail=f"{s / (s + n):.3f}", expected=rates, actual=s,
                  source="시행령 [별표 2] 식대 50% / 별지 제6호서식 항목별 설명 (2인실 40%·3인실 30%)")


def _consult_options(day, inst_types):
    """진료일에 나올 수 있는 진찰료 행 금액 → {금액: [(종별, 초재진, 구분)]}.  구분: base / sat(토요 가산) / night_holiday."""
    opts = {}
    sat_from = R.rules_data()["sat_morning_from"]
    for t in inst_types:
        for kind in ("FIRST", "REVISIT"):
            base = R.price(day, f"CONSULT_{kind}_{t}", t)
            if base is None:
                continue
            opts.setdefault(base, []).append((t, kind, "base"))
            nh = R.price(day, f"CONSULT_{kind}_{t}_NIGHT", t)
            if nh:
                opts.setdefault(nh, []).append((t, kind, "night_holiday"))
            sat = R.price(day, f"CONSULT_{kind}_{t}_SAT", t)
            if sat and day >= date.fromisoformat(str(sat_from.get(t, "2100-01-01"))):
                opts.setdefault(base + sat, []).append((t, kind, "sat"))
    # 외래 의약품관리료(원내 투약 시, 기본진료료)를 진찰료 칸에 합쳐 적는 양식도 있음 → 같은 종별 의약품관리료를 더한 값도 허용
    # (예: 2026 의원 재진 13,370 + 220 = 13,590). 투약 행위료 칸에 적는 양식(우리 생성기)은 위 값 그대로
    for v, labels in list(opts.items()):
        for t, kind, cls in labels:
            mk = R.price(day, f"MEDKEEP_OUT_{t}", t)
            if mk:
                opts.setdefault(v + mk, []).append((t, kind, cls))
    return opts


def b2_b6_consult(c):
    """진찰료 행 금액이 진료일 연도·종별 공식 진찰료로 나올 수 있는 값인지, 그 값의 시간 구분(토요·야간/공휴)이 진료일과 맞는지."""
    r = c.r
    day, end = r.date("visit.start"), r.date("visit.end")
    s, n = r.get("rows.consult.self"), r.get("rows.consult.nhis")
    if r.get("visit.inpatient") or not day or (end and end != day) or s is None or n is None or s + n == 0:
        return
    if R.price_table(day) is None:
        return
    # 종별 체크 칸은 믿지 않고(B1과 같은 이유) 세 종별 진찰료를 모두 후보로. 상급종합 진찰료는 단가표에 없음
    types = ["clinic", "hospital", "general"]
    total = s + n
    opts = _consult_options(day, types)
    fields = ["rows.consult.self", "rows.consult.nhis", "visit.start"]
    src = R.price_source(day, f"CONSULT_FIRST_{types[0]}") or "심평원 요양급여비용"
    if total not in opts:
        other = [y for y in (2024, 2025, 2026) if y != day.year and total in _consult_options(date(y, 6, 1), types)]
        c.add("B2", "참고", fields, "진찰료가 진료일 연도·종별 공식 진찰료로 나올 수 없는 금액",
              detail=f"진찰료 {total:,}원" + (f" — {other}년 단가와 같음 (날짜 조작 의심)" if other else ""),
              expected=sorted(opts), actual=total, source=src, caveat=CAVEAT["B2"])
        return
    kinds = {k for _, _, k in opts[total]}
    hol = R.is_holiday(day)
    if kinds == {"sat"} and (day.weekday() != 5 or hol):
        c.add("B6", "참고", fields, "토요일 진찰료 가산 금액인데 진료일이 토요일이 아님",
              detail=f"{day} ({'월화수목금토일'[day.weekday()]})", actual=total, source="요양급여비용 진찰료 토요 가산",
              caveat=CAVEAT["B2"])
    if kinds == {"base"} and hol:
        c.add("B6", "참고", fields, "진료일이 일요일·공휴일인데 진찰료에 공휴 가산이 없음", detail=str(day), actual=total,
              source="요양급여비용 진찰료 공휴일 가산 / 관공서의 공휴일에 관한 규정")
    if kinds == {"night_holiday"} and hol is False and not r.get("visit.night") and r.has("visit.night"):
        c.add("B6", "참고", fields + ["visit.night"], "진찰료가 야간·공휴 가산 금액인데 평일이고 야간 표시 없음",
              detail=str(day), actual=total)


def b6_flags(c):
    r = c.r
    day = r.date("visit.start")
    # 토요일 오후(13시~)는 공휴 가산을 적용해 공휴 칸에 표시하기도 함 → 토요일은 보지 않음
    if day and day.weekday() != 5 and r.get("visit.holiday") and R.is_holiday(day) is False:
        c.add("B6", "참고", ["visit.holiday", "visit.start"], "공휴일 진료 표시인데 진료일이 평일",
              detail=f"{day} ({'월화수목금토일'[day.weekday()]})", source="관공서의 공휴일에 관한 규정", caveat=CAVEAT["B6"])


def b4_inpatient(c):
    """입원료 = 입원료 일수 × 간호등급별 입원료 (등급·소재지는 몰라도 나올 수 있는 값 집합으로 검사)."""
    r = c.r
    t, start, end = r.get("institution.type"), r.date("visit.start"), r.date("visit.end")
    if not r.get("visit.inpatient") or t not in ("hospital", "general") or not start or not end or not R.price_table(start):
        return
    rd = R.rules_data()
    nights = (end - start).days
    room_keys = {"room_4": ["ROOM_4"], "room_23": ["ROOM_3", "ROOM_2"]}
    for row, keys in room_keys.items():
        s, n = r.get(f"rows.{row}.self"), r.get(f"rows.{row}.nhis")
        if s is None or n is None or s + n == 0:
            continue
        unit = set()
        for k in keys:
            for g in rd["nurse_grades"][t]:
                for metro in ("", "_metro"):
                    p = R.price(start, f"{k}_{t}_G{g}{metro}", t)
                    if p:
                        unit.add(p)
        ok = {u * d for u in unit for d in range(max(1, nights - 1), nights + 2)}
        if unit and s + n not in ok:
            c.add("B4", "참고", [f"rows.{row}.self", f"rows.{row}.nhis", "visit.start", "visit.end"],
                  f"입원료({row})가 입원 일수 × 공식 입원료로 나올 수 없는 금액",
                  detail=f"{s + n:,}원, 입원 {nights}박", actual=s + n, source="심평원 요양급여비용 입원료 (간호등급별)",
                  caveat=CAVEAT["B4"])


def d1_receipt_dates(c):
    r = c.r
    start, end, issue = r.date("visit.start"), r.date("visit.end"), r.date("issue_date")
    today = date.today()
    for f in ("visit.start", "visit.end", "issue_date"):
        d = r.date(f)
        if d and d > today:
            c.add("D1", "확정", [f], "미래 날짜", actual=str(d))
    if start and end and start > end:
        c.add("D1", "확정", ["visit.start", "visit.end"], "진료기간 시작이 끝보다 늦음", actual=f"{start} ~ {end}")
    if end and issue and issue < end:
        inp = bool(r.get("visit.inpatient"))
        c.add("D1", "참고" if inp else "강함", ["visit.end", "issue_date"], "발행일이 진료기간 끝보다 이름",
              actual=f"끝 {end}, 발행 {issue}", caveat=CAVEAT["D1_inpatient"] if inp else "")
    if start and end and r.get("visit.inpatient") is False and start != end:
        c.add("D1", "참고", ["visit.start", "visit.end"], "외래인데 진료기간이 하루가 아님", actual=f"{start} ~ {end}")


def d2_receipt_no(c):
    """영수증번호가 연월(일)로 시작하면 그 날짜가 진료기간·발행일 중 하나와 맞아야 함."""
    r = c.r
    no = r.get("receipt_no")
    if not no:
        return
    no = str(no)
    days = [d for d in (r.date("visit.start"), r.date("visit.end"), r.date("issue_date")) if d]
    if not days:
        return
    digits = re.sub(r"\D", "", no)
    fields = ["receipt_no", "visit.start", "visit.end", "issue_date"]
    if len(digits) >= 12:
        try:
            d = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError:
            d = None
        if d and d not in days:
            c.add("D2", "참고", fields, "영수증번호 앞 날짜가 진료기간·발행일과 다름",
                  expected=[str(x) for x in days], actual=str(d), caveat=CAVEAT["D2"])
        return
    m = re.match(r"^(\d{4})(\d{2})-", no)
    if m and 1 <= int(m.group(2)) <= 12 and 2000 <= int(m.group(1)) <= 2099:
        ym = (int(m.group(1)), int(m.group(2)))
        if ym not in {(d.year, d.month) for d in days}:
            c.add("D2", "참고", fields, "영수증번호 연월이 진료기간·발행일 연월과 다름",
                  expected=sorted({f"{d.year}-{d.month:02d}" for d in days}), actual=f"{ym[0]}-{ym[1]:02d}", caveat=CAVEAT["D2"])


BIZ_WEIGHTS = [1, 3, 7, 1, 3, 7, 1, 3, 5]


def biz_check_digit(first9):
    d = [int(x) for x in first9]
    s = sum(a * b for a, b in zip(d, BIZ_WEIGHTS)) + (d[8] * 5) // 10
    return (10 - s % 10) % 10


def d3_bizno(c):
    r = c.r
    no = r.get("institution.bizno")
    if not no:
        return
    digits = re.sub(r"\D", "", str(no))
    if len(digits) != 10:
        return
    src = "국세청 사업자등록번호 검증번호 산식"
    if biz_check_digit(digits[:9]) != int(digits[9]):
        c.add("D3", "확정", ["institution.bizno"], "사업자등록번호 검증번호가 맞지 않음", actual=no, source=src)
    addr = r.get("institution.address")
    code = addr and R.region_tax_code(addr)
    if code and digits[:3] != code:
        c.add("D3", "참고", ["institution.bizno", "institution.address"], "사업자등록번호 앞 3자리가 주소 관할 세무서 코드와 다름",
              expected=code, actual=digits[:3], source="국세청 세무서별 코드번호·관할구역", caveat=CAVEAT["D3_tax"])


# ── 주민번호 (두 서류 공통) ───────────────────────────────────────────────

RRN_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]


def parse_rrn(rrn):
    """'YYMMDD-G…' → (생년월일, 성별 'M'/'F', 13자리 숫자 또는 None). 앞자리가 날짜가 아니면 생년월일 None."""
    digits = re.sub(r"\D", "", str(rrn))
    if len(digits) < 7:
        return None, None, None
    g = int(digits[6])
    century = {1: 1900, 2: 1900, 5: 1900, 6: 1900, 3: 2000, 4: 2000, 7: 2000, 8: 2000, 9: 1800, 0: 1800}[g]
    try:
        birth = date(century + int(digits[:2]), int(digits[2:4]), int(digits[4:6]))
    except ValueError:
        birth = None
    return birth, ("M" if g % 2 == 1 else "F"), (digits if len(digits) == 13 else None)


def d4_rrn(c):
    r = c.r
    rrn = r.get("patient.rrn")
    if not rrn:
        return
    birth, _, full = parse_rrn(rrn)
    if len(re.sub(r"\D", "", str(rrn))) >= 7 and birth is None:
        c.add("D4", "확정", ["patient.rrn"], "주민등록번호 앞자리가 있을 수 없는 생년월일", actual=rrn)
        return
    ref = r.date("visit.start") or r.date("dates.diagnosis") or r.date("dates.issue")
    if birth and ref and birth > ref:
        c.add("D4", "확정", ["patient.rrn"], "생년월일이 진료일보다 늦음", actual=str(birth))
    if full and birth and birth.year < 2020:
        s = sum(int(d) * w for d, w in zip(full[:12], RRN_WEIGHTS))
        if (11 - s % 11) % 10 != int(full[12]):
            c.add("D4", "참고", ["patient.rrn"], "주민등록번호 검증번호가 맞지 않음 (2020.10 이전 부여 번호)", source="주민등록번호 검증번호 산식",
                  caveat=CAVEAT["D4_check"])


# ── 진단서 ───────────────────────────────────────────────────────────────

def _norm_name(s):
    s = re.sub(r"\([^)]*\)", "", str(s))
    return re.sub(r"[\s·,\-†*'\"]", "", s)


def _name_score(name, entry):
    n = _norm_name(name)
    return max((SequenceMatcher(None, n, _norm_name(x)).ratio() for x in entry.names), default=0.0)


@lru_cache(maxsize=None)
def _all_names():
    """[(정규화 병명, 상병기호)] — 병명 → 코드 역검색용 (한 번만 만듦)."""
    return [(_norm_name(x), code) for code, e in R.kcd_master().items() for x in e.names]


def _best_other_code(name, exclude):
    """병명이 가장 닮은 다른 상병기호. quick_ratio로 먼저 거르고 남은 것만 정밀 비교."""
    n = _norm_name(name)
    best, best_code = 0.0, None
    sm = SequenceMatcher(None, "", n)
    for nx, code in _all_names():
        if code == exclude or abs(len(nx) - len(n)) > max(3, len(n) // 2):
            continue
        sm.set_seq1(nx)
        if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
            continue
        s = sm.ratio()
        if s > best:
            best, best_code = s, code
    return best, best_code


def c_diagnoses(c):
    r = c.r
    dxs = r.get("diagnoses") or []
    if not dxs:
        return
    master = R.kcd_master()
    birth, sex, _ = parse_rrn(r.get("patient.rrn") or "")
    ref = r.date("dates.diagnosis") or r.date("dates.issue")
    age = None if not (birth and ref) else ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    src = "심평원 상병마스터 2025.9"
    seen = {}
    for i, dx in enumerate(dxs):
        code = R.normalize_code(dx.get("code", ""))
        e = master.get(code)
        fld = ["diagnoses"]
        if not code:
            continue
        if e is None:
            if code.startswith("U"):
                grade, cav = "참고", CAVEAT["C1_U"]
            elif ref is None or ref < date(2021, 1, 1):
                grade, cav = "참고", CAVEAT["C1_old"]
            else:
                grade, cav = "강함", ""
            c.add("C1", grade, fld, f"상병마스터에 없는 질병분류기호 {dx.get('code')}", actual=dx.get("code"), source=src, caveat=cav)
            continue
        if not e.complete:
            c.add("C1", "참고", fld, f"{dx.get('code')}는 불완전코드 (하위 세분류가 있음)", source=src)
        if dx.get("name"):
            own = _name_score(dx["name"], e)
            if own < 0.8:
                other, other_code = _best_other_code(dx["name"], code)
                if other >= 0.9 and other - own >= 0.2:
                    c.add("C2", "강함", fld, f"병명 '{dx['name']}'은 {other_code[:3] + ('.' + other_code[3:] if len(other_code) > 3 else '')}의 병명 "
                          f"— 적힌 코드 {dx.get('code')}와 다름",
                          expected=master[other_code].names[0], actual=e.names[0], source=src)
                elif own < 0.5:
                    c.add("C2", "참고", fld, f"병명 '{dx['name']}'이 코드 {dx.get('code')}의 공식 병명과 다름",
                          expected=e.names[0], actual=dx["name"], source=src)
        if sex and e.sex and e.sex != sex:
            c.add("C3", "강함", fld + ["patient.rrn"], f"{dx.get('code')}는 {'여성' if e.sex == 'F' else '남성'}만 쓰는 코드인데 환자는 "
                  f"{'남성' if sex == 'M' else '여성'}", source=src + " 성별구분")
        if age is not None and ((e.age_min is not None and age < e.age_min) or (e.age_max is not None and age > e.age_max)):
            # 연령 제한은 심사 경고 성격(예: 노안 40세 이상)이라 참고. 성별 제한은 불가능한 조합이라 강함
            c.add("C3", "참고", fld + ["patient.rrn"], f"{dx.get('code')}의 연령 제한({e.age_min or 0}~{e.age_max or ''}세)에 안 맞음",
                  actual=f"{age}세", source=src + " 상·하한연령")
        if dx.get("main") and not e.main_ok:
            c.add("C4", "참고", fld, f"{dx.get('code')}는 주상병으로 쓸 수 없는 코드", source=src + " 주상병사용구분", caveat=CAVEAT["C4"])
        if code in seen:
            c.add("C4", "참고", fld, f"주·부 진단에 같은 코드 {dx.get('code')}가 두 번", source="의료법 시행규칙 별지 제5호의2서식",
                  caveat=CAVEAT["C4"])
        seen[code] = i


def d1_diagnosis_dates(c):
    r = c.r
    pairs = [("dates.onset", "dates.diagnosis", "발병일이 진단일보다 늦음"),
             ("dates.diagnosis", "dates.issue", "진단일이 발급일보다 늦음"),
             ("dates.admit", "dates.discharge", "입원일이 퇴원일보다 늦음"),
             ("dates.discharge", "dates.issue", "퇴원일이 발급일보다 늦음"),
             ("dates.onset", "dates.issue", "발병일이 발급일보다 늦음")]
    for a, b, title in pairs:
        da, db = r.date(a), r.date(b)
        if da and db and da > db:
            c.add("D1", "확정", [a, b], title, actual=f"{da} > {db}")
    birth, _, _ = parse_rrn(r.get("patient.rrn") or "")
    on = r.date("dates.onset")
    if birth and on and on < birth:
        c.add("D1", "확정", ["dates.onset", "patient.rrn"], "발병일이 생년월일보다 이름", actual=f"{on} < {birth}")


RECEIPT_RULES = [a1_formulas, a2_column_sums, a3_paid, b1_row_rates, b2_b6_consult, b6_flags, b4_inpatient,
                 d1_receipt_dates, d2_receipt_no, d3_bizno, d4_rrn]
DIAGNOSIS_RULES = [c_diagnoses, d1_diagnosis_dates, d4_rrn, d3_bizno]


def check(reading):
    """Reading → Finding 목록 (등급 순)."""
    c = Checker(reading)
    for rule in RECEIPT_RULES if reading.doc_type == "receipt" else DIAGNOSIS_RULES:
        rule(c)
    c.found.sort(key=lambda f: GRADES.index(f["grade"]))
    return c.found


def verdict(findings):
    """확정·강함이 하나라도 있으면 위변조 의심, 참고만 있으면 확인 필요."""
    if any(f["grade"] in ("확정", "강함") for f in findings):
        return "위변조 의심"
    return "확인 필요" if findings else "정상"
