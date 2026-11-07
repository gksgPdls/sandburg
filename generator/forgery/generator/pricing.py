"""[역할] 단가 조회: 진료일 연도의 공식 단가표에서 항목 단가를 찾고 종별가산을 적용. 시작 점검(단가 누락)
[쓰는 곳] generate.py, main.py, tools/verify_cases.py
[입력 → 출력] config/prices*.yaml → (단가, 출처)

단가 조회: 모든 단가는 공식 단가표(심평원 수가·약가, 보건복지부 고시 제증명수수료 상한)에서만 가져옴.

단가표 (config/price_sources.yaml 규칙으로 만듦)
  config/prices.yaml        2026  심평원 오픈 API (tools/fetch_prices.py)
  config/prices_<연도>.yaml  2024·2025  그 해 「건강보험요양급여비용」 책자 점수 × 그 해 환산지수 (tools/build_book_prices.py)
                                       + 그 해 1월 1일 약제급여목록 (tools/build_list_prices.py)

진료일에 적용되는 표 = 기준일(valid_from) ≤ 진료일 인 표 중 가장 최근 것.
그 표에 없는 항목은 오류로 멈춤 (다른 해 단가로 대신하지 않음 → 근거 없는 금액이 생성되지 않음).

단가표의 수가는 종별가산 전 금액 (상대가치점수 × 환산지수, 10원 미만 4사5입).
종별가산 대상 행위는 unit_price(addon_rate=) 로 단가 × (1 + 가산율), 원 미만 4사5입
(「건강보험요양급여비용」 제1편 제1부 Ⅰ.1 단서, Ⅱ 요양기관 종별가산율).
"""
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_TABLES = None


def tables() -> list:
    """모든 단가표 (2026 + 과거 연도), 기준일 순."""
    global _TABLES
    if _TABLES is None:
        files = [ROOT / "config" / "prices.yaml", *(ROOT / "config").glob("prices_*.yaml")]
        if not files[0].exists():
            raise SystemExit("config/prices.yaml 이 없습니다: .venv/bin/python tools/fetch_prices.py 를 먼저 실행하세요.")
        _TABLES = sorted((yaml.safe_load(f.read_text(encoding="utf-8")) for f in files),
                         key=lambda t: str(t["valid_from"]))
    return _TABLES


def valid_from() -> date:
    """가장 이른 단가 기준일."""
    return date.fromisoformat(str(tables()[0]["valid_from"]))


def table_for(visit: date) -> dict:
    """진료일에 적용되는 단가표."""
    if visit < valid_from():
        raise ValueError(f"진료일 {visit} 이 단가 기준일 {valid_from()} 이전")
    return [t for t in tables() if date.fromisoformat(str(t["valid_from"])) <= visit][-1]


def entry(key: str, visit: date) -> dict:
    """진료일 단가표의 항목. 없으면 오류."""
    t = table_for(visit)
    if key not in t["items"]:
        raise KeyError(f"{key}: {t['valid_from']} 단가표에 공식 단가 없음")
    return t["items"][key]


def with_addon(price: int, rate: float) -> int:
    """종별가산 적용 단가: 단가 × (1 + 가산율), 원 미만 4사5입."""
    if not rate:
        return price
    return int((Decimal(price) * (1 + Decimal(str(rate)))).quantize(Decimal("1"), ROUND_HALF_UP))


def unit_price(key: str, inst_type: str, visit: date, addon_rate: float = 0.0) -> tuple:
    """항목 단가 → (단가, 출처). key 는 price_sources.yaml 의 항목키 (예: XR_KNEE, CONSULT_FIRST_clinic)."""
    item = entry(key, visit)
    src = item["source"] + (f" × 종별가산 {round(addon_rate * 100)}%" if addon_rate else "")
    return with_addon(int(item["price"][inst_type]), addon_rate), src


def room_key(cb: dict, room: dict, inst_type: str, grade: str, metro: bool) -> str:
    """입원료 단가 키: ROOM_<병실>_<종별>_G<간호등급>[_metro] (최하위 등급만 소재지로 코드가 갈림)."""
    suffix = "_metro" if metro and grade == cb["lowest_grade"][inst_type] else ""
    return f"{room['price_key']}_{inst_type}_G{grade}{suffix}"


def codebook_keys(cb: dict) -> list:
    """codebook 이 쓸 수 있는 모든 단가 키 (처방 항목·제증명 + 진찰료(시간 가산 포함)·의약품관리료 + 입원료·식대)."""
    keys = list(cb["fee_items"]) + [cb["cert_fee_key"]]
    max_los = max((d["los_days"][1] for d in cb["departments"].values() if "los_days" in d), default=0)
    for t, it in cb["institution_types"].items():
        keys += [f"CONSULT_{k}_{t}{x}" for k in ("FIRST", "REVISIT") for x in ("", "_NIGHT", "_HOLIDAY", "_SAT")]
        keys.append(f"MEDKEEP_OUT_{t}")
        if it["can_admit"]:
            keys += [room_key(cb, r, t, str(g), m) for r in cb["rooms"].values() for g in cb["nurse_grades"][t]
                     for m in (False, True)]
            keys += [f"MEAL_{t}", *cb["meal_addons"]] + [f"MEDKEEP_IN_{t}_{d}" for d in range(1, max_los + 1)]
    return list(dict.fromkeys(keys))


def check_codebook(cb: dict) -> None:
    """생성 전 점검: codebook 이 쓸 수 있는 모든 단가 키가 진료기간에 걸치는 모든 단가표에 종별 단가와 함께 있는지,
    진료기간이 가장 이른 기준일 이후인지. 하나라도 빠지면 생성하지 않고 멈춤."""
    start, end = (date.fromisoformat(d) for d in cb["date_range"])
    if start < valid_from():
        raise SystemExit(f"codebook date_range 시작일 {start} 이 단가 기준일 {valid_from()} 이전")
    keys = codebook_keys(cb)
    ts = tables()
    froms = [date.fromisoformat(str(t["valid_from"])) for t in ts] + [date.max]
    for i, t in enumerate(ts):
        if not (froms[i] <= end and froms[i + 1] > start):   # 진료기간과 겹치는 표만
            continue
        missing = [k for k in keys if k not in t["items"]]
        missing += [f"{k}({it})" for k in keys if k in t["items"] for it in cb["institution_types"]
                    if it not in t["items"][k]["price"]]
        if missing:
            raise SystemExit(f"{t['valid_from']} 단가표에 공식 단가가 없는 항목: " + ", ".join(missing))
