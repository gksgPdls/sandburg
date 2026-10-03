"""단가 조회: 모든 급여 단가는 심평원 오픈 API 공식 단가 (config/prices.yaml) 에서만 가져옴.

config/prices.yaml 은 tools/fetch_prices.py 가 config/price_sources.yaml 규칙대로 만든 파일:
  수가(행위·입원료·식대)  수가기준정보조회서비스   종별 단가 (의원 / 병원·종합병원)
  약제                    약가기준정보조회서비스   상한금액
  치료재료                치료재료정보조회서비스   상한단가
  비급여                  비급여진료비정보조회서비스  종별 공표 최소·중간값·최대 (range) → 범위 안에서 뽑음
  제증명수수료            보건복지부 고시 상한금액

종별 단가가 없는 항목(예: 의원의 비급여)은 available() 이 False → 그 처방 줄은 생성하지 않음.
단가표에 없는 항목이나 단가 기준일(valid_from) 이전 진료일은 오류로 멈춤 → 근거 없는 금액이 생성되지 않음.
"""
import random
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NPAY_UNIT = 1000     # 비급여 금액 단위: 병원 신고가 표본 1,881건(1만원 이상) 중 86%가 천 원 단위
_TABLE = None


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        f = ROOT / "config" / "prices.yaml"
        if not f.exists():
            raise SystemExit("config/prices.yaml 이 없습니다: .venv/bin/python tools/fetch_prices.py 를 먼저 실행하세요.")
        _TABLE = yaml.safe_load(f.read_text(encoding="utf-8"))
    return _TABLE


def valid_from() -> date:
    return date.fromisoformat(str(table()["valid_from"]))


def available(key: str, inst_type: str) -> bool:
    """이 종별에 공식 단가가 있는지 (비급여는 병원급 이상만 공개)."""
    item = table()["items"].get(key, {})
    return inst_type in item.get("price", item.get("range", {}))


def unit_price(key: str, inst_type: str, visit: date, rng: random.Random) -> tuple:
    """항목 단가 → (단가, 근거). key 는 price_sources.yaml 의 항목키 (예: XR_ANKLE, CONSULT_FIRST_clinic).

    고정 단가(price)는 그대로. 비급여(range)는 종별 공표 최소~최대 안에서 중간값이 가장 자주 나오는
    삼각분포로 뽑고 천 원 단위로 맞춤 (최소·최대 밖으로는 나가지 않음).
    """
    item = table()["items"].get(key)
    if item is None:
        raise KeyError(f"{key}: 공식 단가 없음 (config/price_sources.yaml 에 출처 추가 후 tools/fetch_prices.py 실행)")
    if visit < valid_from():
        raise ValueError(f"진료일 {visit} 이 단가 기준일 {valid_from()} 이전")
    if "price" in item:
        return int(item["price"][inst_type]), item["source"]
    r = item["range"][inst_type]
    x = round(rng.triangular(r["min"], r["max"], r["med"]) / NPAY_UNIT) * NPAY_UNIT
    return int(min(max(x, r["min"]), r["max"])), f"{item['source']} 종별 공표 범위"


def check_codebook(cb: dict) -> None:
    """생성 전 점검: codebook 의 모든 수가 항목·진찰료·입원료·식대·제증명수수료가 단가표에 있는지, 진료기간이 기준일 이후인지."""
    items = table()["items"]
    keys = list(cb["fee_items"]) + [cb["cert_fee_key"]]
    for t, it in cb["institution_types"].items():
        keys += [f"CONSULT_FIRST_{t}", f"CONSULT_REVISIT_{t}"]
        if it["can_admit"]:
            keys += [f"{r['price_key']}_{t}" for r in cb["rooms"].values()] + [f"MEAL_{t}"]
    missing = [k for k in keys if k not in items]
    if missing:
        raise SystemExit("공식 단가가 없는 항목: " + ", ".join(missing))
    no_type = [f"{k}({t})" for k in keys if "price" in items[k] for t in cb["institution_types"]
               if t not in items[k]["price"]]
    if no_type:
        raise SystemExit("종별 단가가 빠진 급여 항목: " + ", ".join(no_type))
    start = date.fromisoformat(cb["date_range"][0])
    if start < valid_from():
        raise SystemExit(f"codebook date_range 시작일 {start} 이 단가 기준일 {valid_from()} 이전")
