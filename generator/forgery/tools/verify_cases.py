"""[역할] 생성 결과 검증: data.json 을 공식 단가표·규칙으로 다시 계산해 대조
[쓰는 곳] 사람이 생성 후 실행
[입력 → 출력] out/ → 항목별 통과/오류

생성 결과 검증: 출력 폴더의 data.json 을 공식 단가표·규칙으로 다시 계산해 대조.

검사 항목 (김동현 「합성 원본 교차 점검」 5절의 검증 항목 이름을 따름):
  CALC-01  줄 금액 = 단가 × 수량 × 일수, ⑥ = 줄 합, ①② 분할·외래 100원 미만 절사, ⑦⑧⑩⑪ 산식
  PRICE-01 줄 단가 = 진료일이 속한 연도 단가표의 단가 (종별가산 대상은 × (1 + 가산율))
  CALC-05  종별가산: 대상 행위만, 의원 0% / 병원 5% / 종합병원 10%
  RULE-01  표층열치료 + 심층열치료 같은 날 → 표층열 50%(MM015)
  RULE-03  의약품관리료: 원내 투약이 있으면 정확히 1줄(외래 방문당·1일분 / 입원 투약일수별), 없으면 0줄
  RATE-01  본인부담률: 외래 종별 부담률, 입원 20%·식대 50%·2인실 40%·3인실 30%
  IN-01    입원: 입원료 코드 = 기관 간호등급(·소재지) 코드, 입원료 일수 = 박 수, 식대 = 3식 × 박 수
  DATE-01  진단일 ≤ 진료일 ≤ 퇴원일 = 발행일, 진단일·퇴원일 같은 해, 재진 간격 3~30일
  TIME-01  진료 시각 구분이 날짜·시각과 맞고(평일 18시~/토요일 13시~ 야간, 토요일 09~13시, 일요일·공휴일), 진찰료 코드가
           그 구분의 코드(야간·공휴일 가산 / 토요일 별도 1줄: 의원 2024~, 병원급 2025~), 의원 야간·토요·공휴 맥립종 절개술은
           가산 코드(S5400080). 입원일은 평일·비공휴일 오전
  CERT-01  제증명수수료: 단가표의 고시 상한금액(일반진단서 20,000원), 비급여(④), 부담률 0
  CODE-01  주·부 진단이 상병마스터 완전코드이고 병명·영문명이 상병마스터와 같음
  BIZ-01   사업자등록번호 검증번호, 앞 3자리 = 주소 관할 세무서 코드, 가운데 2자리 = 개설 주체 구분코드

실행: .venv/bin/python tools/verify_cases.py [출력 폴더, 기본 out]
"""
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docgen import calc, ids, kcd, pricing  # noqa: E402
from docgen.generate import day_kind  # noqa: E402


def check(c: dict, cb: dict, master: dict, holidays: set) -> list:
    errs = []
    E = lambda k, m: errs.append((k, m))   # noqa: E731
    h, r, dc = c["hospital"], c["receipt"], c["diagnosis_cert"]
    d = {k: date.fromisoformat(v) for k, v in c["dates"].items()}
    t, it = h["type"], cb["institution_types"][h["type"]]
    lines, adm, nights = r["lines"], c["admitted"], c["nights"]
    codes = [ln["code"] for ln in lines]

    # CALC-01
    for msg in calc.verify(r, lines):
        E("CALC-01", msg)
    kind = "REVISIT" if c["revisit"] else "FIRST"
    slot = c["time_slot"]
    consult_key = f"CONSULT_{kind}_{t}" + {"night": "_NIGHT", "holiday": "_HOLIDAY"}.get(slot, "")
    # CERT-01
    for ln in lines:
        if ln.get("cover") == "nonpay":
            want = int(pricing.entry(cb["cert_fee_key"], d["visit"])["price"][t])   # 고시 상한금액
            if ln["code"] != cb["cert_fee_key"] or ln["unit_price"] != want or ln["rate"] != 0.0 or ln["row"] != "cert":
                E("CERT-01", f"{ln['code']} {ln['unit_price']}")
    # PRICE-01 / CALC-05 / RATE-01
    for ln in lines:
        if ln.get("cover") == "nonpay":
            continue
        key = {"CONSULT": consult_key, "CONSULT_SAT": f"CONSULT_{kind}_{t}_SAT"}.get(ln["code"], ln["code"])
        base = int(pricing.entry(key, d["visit"])["price"][t])
        addon = cb["fee_items"].get(ln["code"], {}).get("addon", False)
        want = pricing.with_addon(base, it["addon_rate"]) if addon else base
        if ln["unit_price"] != want:
            E("CALC-05" if ln["unit_price"] in (base, want) else "PRICE-01", f"{ln['code']} {ln['unit_price']} ≠ {want}")
        if ("종별가산" in ln["price_source"]) != bool(addon and it["addon_rate"]):
            E("CALC-05", f"{ln['code']} 가산 표기 불일치")
        if adm:
            room = next((v for v in cb["rooms"].values() if ln["code"].startswith(v["price_key"] + "_")), None)
            want_rate = room["rate"] if room else cb["meal_rate"] if ln["row"] == "meal" else cb["inpatient_rate"]
        else:
            want_rate = it["outpatient_rate"]
        if ln["rate"] != want_rate:
            E("RATE-01", f"{ln['code']} 부담률 {ln['rate']} ≠ {want_rate}")
    # RULE-01
    if "PT_HOT" in codes and "PT_DEEP" in codes:
        E("RULE-01", "표층열·심층열 같은 날 전액")
    # RULE-03
    mk = [ln for ln in lines if ln["code"].startswith("MEDKEEP")]
    want_mk = f"MEDKEEP_IN_{t}_{nights}" if adm else f"MEDKEEP_OUT_{t}"
    has_drug = any(ln["row"] in cb["medkeep_rows"] for ln in lines)
    if (len(mk) != (1 if has_drug else 0)) or (mk and mk[0]["code"] != want_mk):
        E("RULE-03", f"의약품관리료 {[x['code'] for x in mk]} (원내 투약 {has_drug})")
    # IN-01
    if adm:
        rooms = [ln for ln in lines if ln["order_set"] == "ROOM"]
        if len(rooms) != 1 or rooms[0]["days"] != nights:
            E("IN-01", "입원료 줄·일수")
        else:
            room = next(v for v in cb["rooms"].values() if rooms[0]["code"].startswith(v["price_key"] + "_"))
            if rooms[0]["code"] != pricing.room_key(cb, room, t, h["nurse_grade"], h["metro"]):
                E("IN-01", f"입원료 코드 {rooms[0]['code']} ≠ 간호 {h['nurse_grade']}등급")
        meals = [ln for ln in lines if ln["row"] == "meal"]
        if {ln["code"] for ln in meals} != {f"MEAL_{t}", *h["meal_addons"]} or any(
                ln["qty"] != 3 or ln["days"] != nights for ln in meals):
            E("IN-01", "식대 구성·수량")
        if (d["end"] - d["visit"]).days != nights:
            E("IN-01", "퇴원일 - 입원일 ≠ 박 수")
    # DATE-01
    if not (d["diagnosis"] <= d["visit"] <= d["end"] == d["issue"]) or d["diagnosis"].year != d["end"].year:
        E("DATE-01", "날짜 순서·연도")
    if c["revisit"] and not 3 <= (d["visit"] - d["diagnosis"]).days <= 30:
        E("DATE-01", "재진 간격")
    # TIME-01
    hh, mm = map(int, c["visit_time"].split(":"))
    dk = day_kind(d["visit"], holidays)
    want_slot = ("holiday" if dk == "holiday" else
                 ("sat_morning" if (hh, mm) > (9, 0) and hh < 13 else "night") if dk == "saturday" else
                 ("night" if hh >= 18 or hh < 9 else "day"))
    if adm and (dk != "weekday" or not 6 <= hh < 12 or slot != "day"):
        E("TIME-01", f"입원일 {d['visit']} {c['visit_time']}")
    if not adm and slot != want_slot:
        E("TIME-01", f"{d['visit']} {c['visit_time']} 구분 {slot} ≠ {want_slot}")
    sat_line = "CONSULT_SAT" in codes
    sat_due = slot == "sat_morning" and d["visit"] >= date.fromisoformat(str(cb["sat_morning_from"][t]))
    if sat_line != sat_due:
        E("TIME-01", f"토요일 가산 줄 {sat_line} (해당 {sat_due})")
    offhours = t == "clinic" and not adm and slot in ("night", "sat_morning", "holiday")
    for a, b in cb["offhours_items"].items():
        if (a if offhours else b) in codes:
            E("TIME-01", f"{a if offhours else b} (의원 시간 외 처치 가산 불일치)")
    if (r.get("night"), r.get("holiday")) != (slot == "night", slot == "holiday"):
        E("TIME-01", "계산서 야간·공휴일 표시")
    # CODE-01
    for dx in filter(None, (dc["main"], dc["sub"])):
        m = master.get(dx["code"].replace(".", ""))
        if not m or not m[2] or (dx["name"], dx["en"]) != m[:2]:
            E("CODE-01", dx["code"])
    # BIZ-01
    reg = next((g for g in cb["regions"] if h["address"].startswith(g[0] + " ")), None)
    if not ids.biz_is_valid(h["bizno"]) or not reg or h["bizno"][:3] != reg[3] \
            or h["bizno"][4:6] not in cb["bizno_mid"][h["owner"]]:
        E("BIZ-01", h["bizno"])
    return errs


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out"
    cb = yaml.safe_load((ROOT / "config" / "codebook.yaml").read_text(encoding="utf-8"))
    master = kcd.load_master()
    holidays = {date.fromisoformat(str(x)) for x in cb["holidays"]}
    files = sorted(out.glob("case_*/data.json"))
    if not files:
        sys.exit(f"{out} 에 data.json 이 없음")
    fails, n = Counter(), 0
    for f in files:
        errs = check(json.loads(f.read_text(encoding="utf-8")), cb, master, holidays)
        n += 1
        for k, m in errs:
            if fails[k] < 3:
                print(f"  {f.parent.name} {k}: {m}")
            fails[k] += 1
    print(f"\n{n}건 검사: " + ("모든 항목 통과" if not fails else ", ".join(f"{k} {v}건" for k, v in sorted(fails.items()))))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
