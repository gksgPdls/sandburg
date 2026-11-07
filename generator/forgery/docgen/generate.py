"""[역할] 사건 1건의 모든 값(기관·진단·날짜·시각·처방·금액·기재값)을 만드는 핵심 로직
[쓰는 곳] main.py
[입력 → 출력] codebook.yaml + 단가(pricing) → case dict (data.json 내용)

사건(case) 1건 생성: 고정 환자정보 + 규칙 기반 가상 기관·진료(외래/입원)·처방·금액 → 진단서/계산서 데이터.

생성 순서 (보고서 2장과 같은 순서):
  1. 진료과 → 2. 기관 종류 → 3. 진단 → 4. 기관 정보 → 5. 입원 여부·날짜·시각
  6. 처방(세부내역) → 7. 금액 계산·검산 → 8. 결제 → 9. 진단서·계산서 기재값
모든 확률·목록은 config/codebook.yaml, 모든 단가는 docgen/pricing.py (공식 단가표).
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


def day_kind(d: date, holidays: set) -> str:
    """holiday = 일요일 또는 공휴일 (관공서의 공휴일에 관한 규정), saturday = 공휴일이 아닌 토요일, weekday = 그 외."""
    if d.weekday() == 6 or d in holidays:
        return "holiday"
    return "saturday" if d.weekday() == 5 else "weekday"


def rand_workday(rng: random.Random, lo: date, hi: date, holidays: set) -> date:
    """lo~hi 사이 무작위 평일(월~금)·비공휴일 — 입원일."""
    while True:
        d = lo + timedelta(days=rng.randint(0, (hi - lo).days))
        if day_kind(d, holidays) == "weekday":
            return d


def rand_visit(rng: random.Random, lo: date, hi: date, holidays: set, cb: dict) -> tuple:
    """외래 진료일·시각: 요일 종류(평일/토요일/공휴일)를 확률로 고르고 그 종류의 날짜, 그 안의 시각 구간을 고름.
    → (날짜, "HH:MM", 시간 구분: day | night | sat_morning | holiday)."""
    kind = pick(rng, {k: v["weight"] for k, v in cb["visit_days"].items()})
    while True:
        d = lo + timedelta(days=rng.randint(0, (hi - lo).days))
        if day_kind(d, holidays) == kind:
            break
    slots = cb["visit_days"][kind]["slots"]
    slot = pick(rng, {k: v[0] for k, v in slots.items()})
    (h0, m0), (h1, m1) = (map(int, x.split(":")) for x in slots[slot][1:])
    t = rng.randint(h0 * 60 + m0, h1 * 60 + m1)
    return d, f"{t // 60:02d}:{t % 60:02d}", slot


def person_name(rng: random.Random, cb: dict) -> str:
    return rng.choice(cb["surnames"]) + "".join(rng.sample(cb["given_syllables"], 2))


# ── 4. 기관 정보 ──────────────────────────────────────────────
def make_hospital(rng: random.Random, cb: dict, dept: str, inst_type: str) -> dict:
    """가상 기관. 사업자등록번호: 앞 3자리 = 주소 관할 세무서 코드, 가운데 2자리 = 개설 주체 구분코드.
    입원 가능 기관은 간호관리료 등급과 식대 가산 여부를 기관 속성으로 한 번 뽑음."""
    prefix = rng.choice(cb["hospital_prefixes"])
    if inst_type == "clinic":
        name = f"{prefix}{dept}의원"
    elif inst_type == "hospital":
        name = rng.choice([f"{prefix}{dept}병원", f"{prefix}병원"])
    else:
        name = f"{prefix}{rng.choice(cb['general_suffixes'])}"
    region, roads, area, tax_office = rng.choice(cb["regions"])
    addr = f"{region} {rng.choice(roads)} {rng.randint(1, 450)}"
    if rng.random() < 0.6:
        addr += f", {rng.randint(2, 7)}층"
    doctor = person_name(rng, cb)
    owner = pick(rng, cb["bizno_owner"][inst_type])
    hosp = {
        "name": name, "type": inst_type,
        "type_label": cb["institution_types"][inst_type]["label"],
        "address": addr, "phone": ids.make_phone(rng, area),
        "metro": region.startswith("서울특별시") or "광역시" in region,   # 서울특별시·광역시 구 (입원료 최하위 등급 코드)
        "owner": owner,   # individual: 개인 의사 (대표자 = 진료 의사) / nonprofit: 의료법인 등
        "bizno": ids.make_bizno(rng, tax_office, rng.choice(cb["bizno_mid"][owner])),
        "ceo": doctor if owner == "individual" else person_name(rng, cb),
        "doctor": doctor, "license_no": ids.make_license_no(rng),
    }
    if cb["institution_types"][inst_type]["can_admit"]:
        hosp["nurse_grade"] = str(pick(rng, {str(k): v for k, v in cb["nurse_grades"][inst_type].items()}))
        hosp["meal_addons"] = [k for k, p in cb["meal_addons"].items() if rng.random() < p]
    return hosp


# ── 6. 처방(세부내역) ─────────────────────────────────────────
def line(code: str, name: str, row: str, price: tuple, qty: int, days: int, rate: float, order_set: str) -> dict:
    """처방 1줄. 금액 = 단가 × 수량 × 일수. price = pricing.unit_price 의 (단가, 출처).
    기본은 급여 일부본인부담 — 비급여(제증명수수료)는 만든 뒤 "cover": "nonpay" 를 붙임."""
    unit, src = price
    return {"code": code, "name": name, "row": row, "unit_price": unit, "qty": qty, "days": days,
            "amount": unit * qty * days, "rate": rate, "price_source": src, "order_set": order_set}


def build_orders(rng, cb, dcfg, dx, hosp, revisit, admitted, nights, visit: date, slot: str) -> list:
    """slot: 외래 진료 시간 구분 (day | night | sat_morning | holiday), 입원은 day (평일 오전)."""
    t = hosp["type"]
    it = cb["institution_types"][t]
    rate = cb["inpatient_rate"] if admitted else it["outpatient_rate"]
    price = lambda key, addon=False: pricing.unit_price(key, t, visit, addon_rate=it["addon_rate"] if addon else 0.0)  # noqa: E731

    def item_line(code, qty, days, order_set):
        fi = cb["fee_items"][code]
        return line(code, fi["name"], fi["row"], price(code, fi.get("addon")), qty, days, rate, order_set)

    # 진찰료: 종별 × 초진/재진 수가코드 (입원은 입원일 초진 진찰)
    #   야간·공휴일은 가산 코드, 토요일 09~13시는 기본진찰료 30% 를 별도 1줄 (가-1 주3·주5)
    kind = "REVISIT" if revisit else "FIRST"
    label = "진찰료(재진)" if revisit else "진찰료(초진)"
    suffix = {"night": "_NIGHT", "holiday": "_HOLIDAY"}.get(slot, "")
    lines = [line("CONSULT", label + {"_NIGHT": " 야간", "_HOLIDAY": " 공휴일"}.get(suffix, ""), "consult",
                  price(f"CONSULT_{kind}_{t}{suffix}"), 1, 1, rate, "CONSULT")]
    if slot == "sat_morning" and visit >= date.fromisoformat(str(cb["sat_morning_from"][t])):
        lines.append(line("CONSULT_SAT", "진찰료 토요일 가산(09~13시)", "consult", price(f"CONSULT_{kind}_{t}_SAT"),
                          1, 1, rate, "CONSULT"))
    # 의원 외래 18시~09시·토요일·공휴일의 (별표8) 처치 → 30% 가산 코드 (제9장 산정지침 ⒇)
    offhours = t == "clinic" and not admitted and slot in ("night", "sat_morning", "holiday")

    # 입원료(간호등급 코드 1일 단가 × 입원 일수)·식대(1식 단가 × 3식 × 입원 일수)
    if admitted:
        room_name = pick(rng, {k: r["weight"] for k, r in cb["rooms"].items()})
        room = cb["rooms"][room_name]
        key = pricing.room_key(cb, room, t, hosp["nurse_grade"], hosp["metro"])
        lines.append(line(key, f"입원료({room['label']}, 간호 {hosp['nurse_grade']}등급)", room["row"], price(key),
                          1, nights, room["rate"], "ROOM"))
        meals = 3
        lines.append(line(f"MEAL_{t}", "식대(일반식)", "meal", price(f"MEAL_{t}"), meals, nights, cb["meal_rate"], "MEAL"))
        for k in hosp["meal_addons"]:
            nm = {"MEAL_DIETITIAN": "식대 가산(영양사)", "MEAL_COOK": "식대 가산(조리사)", "MEAL_DIRECT": "식대 가산(직영)"}[k]
            lines.append(line(k, nm, "meal", price(k), meals, nights, cb["meal_rate"], "MEAL"))

    # 처방 묶음: 진단 조건(dx) + 초진/재진(입원) 확률
    for s in (dcfg["inpatient_orders"] if admitted else dcfg["orders"]):
        if "dx" in s and dx["code"] not in s["dx"]:
            continue
        if rng.random() >= (s["p"] if admitted else (s["p_revisit"] if revisit else s["p_first"])):
            continue
        days = nights if (admitted and s.get("per_day")) else 1
        for code, lo, hi in s["items"]:
            if code == "$xray":                 # 진단별 촬영 부위
                code = dx.get("xray")
                if not code:
                    continue
            code = rng.choice(code.split("|"))  # "A|B" → 둘 중 하나
            if offhours:
                code = cb["offhours_items"].get(code, code)
            qty = rng.randint(lo, hi)
            if qty:
                lines.append(item_line(code, qty, days, s["id"]))

    # 같은 날 함께 하면 단가가 달라지는 항목 (표층열치료 + 심층열치료 → 표층열 소정점수 50%, MM015)
    codes = {ln["code"] for ln in lines}
    for i, ln in enumerate(lines):
        for other, alt in cb["fee_items"].get(ln["code"], {}).get("same_day", {}).items():
            if other in codes:
                lines[i] = item_line(alt, ln["qty"], ln["days"], ln["order_set"])

    # 의약품관리료: 원내 투약(주사약·경구약)이 있으면 1줄 (외래 방문당/1일분, 입원 투약일수별 코드)
    if any(ln["row"] in cb["medkeep_rows"] for ln in lines):
        key = f"MEDKEEP_IN_{t}_{nights}" if admitted else f"MEDKEEP_OUT_{t}"
        lines.append(line(key, "의약품관리료", cb["medkeep_row"], price(key), 1, 1, rate, "MEDKEEP"))
    return lines


def opinion(rng, cb, lines, admitted, d) -> str:
    """진단서 소견: 실제로 생성된 처방 행만으로 만든 문장."""
    rows = {ln["row"] for ln in lines}
    acts = [w for r, w in cb["row_acts"].items() if r in rows]
    acts = acts[0] if len(acts) == 1 else ", ".join(acts[:-1]) + " 및 " + acts[-1]
    fmt = lambda x: f"{x.year}년 {x.month}월 {x.day}일"   # noqa: E731
    tpl = rng.choice(cb["opinion_templates"]["inpatient" if admitted else "outpatient"])
    return tpl.format(acts=acts, date=fmt(d["visit"]), admit=fmt(d["visit"]), discharge=fmt(d["end"]))


def make_case(rng: random.Random, case_id: str, patient: dict, cb: dict, var: dict) -> dict:
    # 1. 진료과 → 2. 기관 종류 → 3. 진단 (부 진단: 같은 과의 다른 진단)
    dept = pick(rng, {k: v["weight"] for k, v in cb["departments"].items()})
    dcfg = cb["departments"][dept]
    inst_type = pick(rng, {k: v["weight"] for k, v in cb["institution_types"].items()})
    dx = rng.choice(dcfg["diagnoses"])
    others = [d for d in dcfg["diagnoses"] if d["code"] != dx["code"]]
    sub_dx = rng.choice(others) if others and rng.random() < cb["sub_diagnosis_prob"] else None

    # 4. 기관 정보
    hosp = make_hospital(rng, cb, dept, inst_type)

    # 5. 입원 여부: 입원 가능 진단 × 입원 가능 종별 × 입원 확률. 입원은 진단일(외래 진찰) 당일 평일 오전 입원
    admitted = (dx["code"] in dcfg.get("admit_dx", []) and cb["institution_types"][inst_type]["can_admit"]
                and rng.random() < dcfg["admission_prob"])
    revisit = not admitted and rng.random() < dcfg["revisit_prob"]
    nights = rng.randint(*dcfg["los_days"]) if admitted else 0
    #   날짜: 진단일(첫 진료) ≤ 진료일(외래 재진일 또는 입원일) ≤ 퇴원일 = 발행일
    #   외래 진료일·시각은 요일 종류·시각 구간 확률로, 입원일은 평일 오전 (재진이면 첫 진료 + 3~30일)
    #   진단일과 퇴원일(발행일)이 같은 해에 들도록 다시 뽑음 (단가표가 연도별)
    d_from, d_to = (date.fromisoformat(x) for x in cb["date_range"])
    holidays = {date.fromisoformat(str(h)) for h in cb["holidays"]}
    while True:
        if admitted:
            first_visit, vtime, slot = rand_workday(rng, d_from, d_to - timedelta(days=30), holidays), \
                f"{rng.randint(9, 11):02d}:{rng.randint(0, 59):02d}", "day"
        else:
            first_visit, vtime, slot = rand_visit(rng, d_from, d_to - timedelta(days=30), holidays, cb)
        visit = first_visit
        if revisit:
            visit, vtime, slot = rand_visit(rng, first_visit + timedelta(days=3), first_visit + timedelta(days=30),
                                            holidays, cb)
        end = visit + timedelta(days=nights)
        if first_visit.year == end.year:
            break
    dates = {"diagnosis": first_visit, "visit": visit, "end": end, "issue": end}

    # 6. 처방 → 7. 금액 (계산 후 calc.verify 로 다시 계산해 대조, 다르면 중단)
    lines = build_orders(rng, cb, dcfg, dx, hosp, revisit, admitted, nights, visit, slot)
    # 진단서 발급 수수료 (비급여, 고시 상한금액) — 같은 계산서에 함께 적는 경우만
    if rng.random() < cb["cert_fee_on_receipt_prob"]:
        cert = line(cb["cert_fee_key"], "제증명수수료(일반진단서)", "cert",
                    pricing.unit_price(cb["cert_fee_key"], inst_type, visit), 1, 1, 0.0, "CERT")
        lines.append({**cert, "cover": "nonpay"})
    money = calc.compute(lines, outpatient=not admitted)

    # 8. 결제: ⑧ 환자부담 총액을 결제수단별로 배분
    paid = money["patient_total"]
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
    errs = calc.verify(money, lines)
    if errs:
        raise ValueError(f"{case_id} 금액 검산 실패: {errs}")

    # 9. 진단서·계산서 기재값
    rv = var["receipt"]
    serial = rng.randint(1, 120)
    receipt_no = (f"{end:%Y%m%d}-{serial:04d}" if rv["receipt_no_style"] == "dash"
                  else f"{end:%Y%m%d}{serial:05d}")
    return {
        "case_id": case_id,
        "patient": patient,
        "hospital": hosp,
        "department": dept,
        "admitted": admitted,
        "revisit": revisit,
        "nights": nights,
        "dates": dates,
        "visit_time": vtime,   # 진료(입원) 시각
        "time_slot": slot,     # day | night | sat_morning | holiday
        "diagnosis_cert": {
            "serial_no": f"{end:%Y}-{rng.randint(1, 9999):05d}",
            "final": var["diagnosis"]["final_diagnosis"],
            "main": dx, "sub": sub_dx,
            "opinion": opinion(rng, cb, lines, admitted, dates),
            "purpose": pick(rng, cb["purposes"]),
        },
        "receipt": {
            "kind": "inpatient" if admitted else "outpatient",
            "night": slot == "night", "holiday": slot == "holiday",   # 계산서 '야간(공휴일)진료' 표시
            "room": f"{rng.randint(3, 8)}{rng.randint(1, 20):02d}호" if admitted else "",
            "patient_type": rv["patient_type"],
            "receipt_no": receipt_no,
            "lines": lines,
            **money,
        },
        "variant": var,
    }
