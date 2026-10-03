"""사건(case) 1건 생성: 고정 환자정보 + 규칙 기반 가상 병원·진료·처방·금액 → 진단서/계산서 데이터.

생성 순서 (보고서와 같은 번호):
  1. 진료과 → 2. 기관 종류 → 3. 입원 여부 → 4. 진단명 → 5. 날짜
  6. 처방(세부내역) → 7. 금액 계산 → 8. 결제 → 9. 진단서 문구 → 10. 기관 정보
"""
import random
from datetime import date, timedelta

from . import calc, ids, pricing


def pick(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[float(weights[k]) for k in keys])[0]


def pick_variant(rng: random.Random, spec: dict) -> dict:
    out = {}
    for k, w in spec.items():
        v = pick(rng, {str(x): y for x, y in w.items()})
        out[k] = {"true": True, "false": False}.get(str(v).lower(), v)
    return out


def round10(x: float) -> int:
    return int(round(x / 10.0)) * 10


def rand_workday(rng: random.Random, lo: date, hi: date) -> date:
    """lo~hi 사이 무작위 날짜, 일요일 제외 (야간·공휴일 가산은 생성하지 않음)."""
    while True:
        d = lo + timedelta(days=rng.randint(0, (hi - lo).days))
        if d.weekday() != 6:
            return d


def person_name(rng: random.Random, cb: dict) -> str:
    return rng.choice(cb["surnames"]) + "".join(rng.sample(cb["given_syllables"], 2))


# ── 10. 기관 정보 ──────────────────────────────────────────────
def make_hospital(rng: random.Random, cb: dict, dept: str, inst_type: str) -> dict:
    prefix = rng.choice(cb["hospital_prefixes"])
    if inst_type == "clinic":
        name = f"{prefix}{dept}의원"
    elif inst_type == "hospital":
        name = rng.choice([f"{prefix}{dept}병원", f"{prefix}병원"])
    else:
        name = f"{prefix}{rng.choice(cb['general_suffixes'])}"
    region, roads, area = rng.choice(cb["regions"])
    addr = f"{region} {rng.choice(roads)} {rng.randint(1, 450)}"
    if rng.random() < 0.6:
        addr += f", {rng.randint(2, 7)}층"
    doctor = person_name(rng, cb)
    return {
        "name": name, "type": inst_type,
        "type_label": cb["institution_types"][inst_type]["label"],
        "address": addr, "phone": ids.make_phone(rng, area),
        "bizno": ids.make_bizno(rng, inst_type),
        "ceo": doctor if inst_type == "clinic" else person_name(rng, cb),
        "doctor": doctor, "license_no": ids.make_license_no(rng),
    }


# ── 6. 처방(세부내역) ─────────────────────────────────────────
def line(code: str, name: str, row: str, cover: str, price: tuple, qty: int, days: int, rate: float,
         order_set: str) -> dict:
    """처방 1줄. 금액 = 단가 × 수량 × 일수. price = (단가, 출처) — 항상 pricing.unit_price 의 결과."""
    unit, src = price
    return {"code": code, "name": name, "row": row, "cover": cover, "unit_price": unit, "qty": qty, "days": days,
            "amount": unit * qty * days, "rate": rate if cover == "part" else 0.0, "price_source": src,
            "order_set": order_set}


def build_orders(rng, cb, dcfg, dx, inst_type, admitted, revisit, nights, visit: date) -> list:
    rate = cb["inpatient_rate"] if admitted else cb["institution_types"][inst_type]["outpatient_rate"]
    price = lambda key: pricing.unit_price(key, inst_type, visit, rng)   # noqa: E731
    lines = []

    # 진찰료: 종별 × 초진/재진 수가코드
    key = f"CONSULT_{'REVISIT' if revisit else 'FIRST'}_{inst_type}"
    lines.append(line("CONSULT", "진찰료(재진)" if revisit else "진찰료(초진)", "consult", "part",
                      price(key), 1, 1, rate, "CONSULT"))

    # 입원료·식대: 1일 단가 × 재원일수 / 1식 단가 × 3식 × 재원일수 (종별 수가코드)
    #   입원료 본인부담률은 병실 종류별(4인실 20%, 3인실 30%, 2인실 40%)
    if admitted:
        room_key = pick(rng, {k: r["weight"] for k, r in cb["rooms"].items()})
        room = cb["rooms"][room_key]
        lines.append(line(room_key.upper(), f"입원료({room['label']})", room["row"], "part",
                          price(f"{room['price_key']}_{inst_type}"), 1, nights, room["rate"], "ROOM"))
        lines.append(line("MEAL", "식대", "meal", "part", price(f"MEAL_{inst_type}"),
                          cb["meals_per_day"], nights, cb["meal_rate"], "MEAL"))

    # 처방 묶음: 진단코드 조건 + 초진/재진(또는 입원) 확률
    #   이 종별에 공식 단가가 없는 항목(의원의 비급여)은 줄을 만들지 않음
    sets = dcfg["inpatient_orders"] if admitted else dcfg["orders"]
    for s in sets:
        if "dx" in s and dx["code"] not in s["dx"]:
            continue
        p = s["p"] if admitted else (s["p_revisit"] if revisit else s["p_first"])
        if rng.random() >= p:
            continue
        days = nights if (admitted and s.get("per_day")) else 1
        for code, lo, hi in s["items"]:
            if code.startswith("$"):           # 진단별 촬영 부위 등 (예: $xray → dx.xray)
                code = dx.get(code[1:])
                if not code:
                    continue
            qty = rng.randint(lo, hi)
            if qty and pricing.available(code, inst_type):
                fi = cb["fee_items"][code]
                lines.append(line(code, fi["name"], fi["row"], fi["cover"], price(code), qty, days, rate, s["id"]))
    return lines


def make_case(rng: random.Random, case_id: str, patient: dict, cb: dict, var: dict) -> dict:
    # 1. 진료과
    dept = pick(rng, {k: v["weight"] for k, v in cb["departments"].items()})
    dcfg = cb["departments"][dept]
    # 2. 기관 종류
    inst_type = pick(rng, {k: v["weight"] for k, v in cb["institution_types"].items()})
    # 3. 진단명 (부상병: 같은 과의 다른 진단, sub_diagnosis_prob)
    dx = rng.choice(dcfg["diagnoses"])
    sub_dx = None
    if rng.random() < cb["sub_diagnosis_prob"]:
        sub_dx = rng.choice([d for d in dcfg["diagnoses"] if d["code"] != dx["code"]])
    # 4. 입원 여부: 입원 가능 진단(admit_dx) × 병원급 이상 × admission_prob
    admitted = (dx["code"] in dcfg.get("admit_dx", [])
                and cb["institution_types"][inst_type]["can_admit"]
                and rng.random() < dcfg.get("admission_prob", 0))

    # 5. 날짜: 발병일 ≤ 진단일(첫 진료) ≤ 진료일 ≤ 발행일
    d_from, d_to = (date.fromisoformat(x) for x in cb["date_range"])
    first_visit = rand_workday(rng, d_from, d_to - timedelta(days=45))
    onset = first_visit - timedelta(days=rng.randint(*dx["onset_days"]))
    nights = 0
    if admitted:
        revisit = False
        nights = rng.randint(*dcfg["los_days"])
        start, end = first_visit, first_visit + timedelta(days=nights)
    else:
        revisit = rng.random() < dcfg["revisit_prob"]
        start = end = (rand_workday(rng, first_visit + timedelta(days=3), first_visit + timedelta(days=30))
                       if revisit else first_visit)
    issue = end

    # 6. 처방 → 7. 금액
    lines = build_orders(rng, cb, dcfg, dx, inst_type, admitted, revisit, nights, start)
    cert_fee = 0
    if rng.random() < cb["diagnosis_fee_on_receipt_prob"]:
        cert = pricing.unit_price(cb["cert_fee_key"], inst_type, start, rng)
        cert_fee = cert[0]
        lines.append(line("CERT_FEE", "제증명수수료(진단서)", "cert", "nonpay", cert, 1, 1, 0.0, "CERT"))
    rv = var["receipt"]
    money = calc.compute(lines, not admitted, rv["prepaid"])

    # 8. 결제: ⑪ 합계 = ⑧ 을 결제수단별로 배분
    paid = money["paid_total"]
    mode = pick(rng, cb["payment_mode"])
    if mode == "card":
        pay = {"card": paid, "cash_receipt": 0, "cash": 0}
    elif mode == "cash":
        key = "cash_receipt" if rng.random() < cb["cash_receipt_prob"] else "cash"
        pay = {"card": 0, "cash_receipt": 0, "cash": 0, key: paid}
    else:
        card = min(paid, max(100, int(round(paid * rng.uniform(*cb["mixed_card_ratio"]), -2))))
        pay = {"card": card, "cash_receipt": 0, "cash": paid - card}
    money["payment"] = pay
    errs = calc.verify(money)
    if errs:
        raise ValueError(f"{case_id} 금액 검산 실패: {errs}")

    serial = rng.randint(1, 120)
    receipt_no = (f"{end:%Y%m%d}-{serial:04d}" if rv["receipt_no_style"] == "dash"
                  else f"{end:%Y%m%d}{serial:05d}")

    # 9. 진단서 문구: 향후 치료기간(주) = 진단별 weeks 범위
    weeks = rng.randint(*dx["weeks"])
    opinions = dcfg.get("inpatient_opinions", dcfg["opinions"]) if admitted else dcfg["opinions"]

    return {
        "case_id": case_id,
        "patient": patient,
        "hospital": make_hospital(rng, cb, dept, inst_type),
        "department": dept,
        "admitted": admitted,
        "revisit": revisit,
        "nights": nights,
        "dates": {"onset": onset, "diagnosis": first_visit, "start": start, "end": end, "issue": issue},
        "diagnosis_cert": {
            "serial_no": f"{issue:%Y}-{rng.randint(1, 9999):05d}",
            "final": var["diagnosis"]["final_diagnosis"],
            "main": dx, "sub": sub_dx,
            "opinion": rng.choice(opinions).format(w=weeks),
            "purpose": pick(rng, cb["purposes"]),
            "remark": "",
        },
        "receipt": {
            "kind": "inpatient" if admitted else "outpatient",
            "room": f"{rng.randint(3, 8)}{rng.randint(1, 20):02d}호" if admitted else "",
            "patient_type": rv["patient_type"],
            "receipt_no": receipt_no,
            "lines": lines,
            "cert_fee": cert_fee,
            **money,
        },
        "variant": var,
    }
