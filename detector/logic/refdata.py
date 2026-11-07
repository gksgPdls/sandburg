"""
공식 근거 데이터 읽기 (data/ 폴더, 한 번만 읽고 재사용).

  prices_<연도>.yaml   심평원 공식 단가표 — 2024·2025: 그 해 「건강보험요양급여비용」 책자 점수 × 환산지수,
                      2026: 심평원 수가기준정보 오픈 API. 만든 곳: generator/forgery/tools (종별가산 전 금액)
  hira_kcd_master_2025_9.csv   심평원 상병마스터 2025.9 (공공데이터포털 15067467, cp949)
  rules_data.yaml     부담률·종별가산율·공휴일·관할 세무서 코드 (generator/forgery/config/codebook.yaml에서 옮김)
"""
import csv
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def rules_data():
    return yaml.safe_load((DATA / "rules_data.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def price_tables():
    """기준일 순 단가표 목록."""
    ts = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in DATA.glob("prices_*.yaml")]
    return sorted(ts, key=lambda t: str(t["valid_from"]))


def price_table(day):
    """진료일에 적용되는 단가표 (기준일 ≤ 진료일 중 가장 최근). 첫 기준일보다 이르면 None."""
    ts = [t for t in price_tables() if date.fromisoformat(str(t["valid_from"])) <= day]
    return ts[-1] if ts else None


def price(day, key, inst_type):
    """진료일·종별 공식 단가. 표에 없으면 None."""
    t = price_table(day)
    item = t and t["items"].get(key)
    return int(item["price"][inst_type]) if item and inst_type in item["price"] else None


def price_source(day, key):
    t = price_table(day)
    item = t and t["items"].get(key)
    return item["source"] if item else ""


@lru_cache(maxsize=None)
def holidays():
    return {date.fromisoformat(d) for d in rules_data()["holidays"]}


def holiday_years():
    return {d.year for d in holidays()}


def is_holiday(d):
    """일요일 또는 관공서 공휴일. 공휴일 목록이 없는 해면 None (판단 불가)."""
    if d.weekday() == 6:
        return True
    if d.year not in holiday_years():
        return None
    return d in holidays()


class KcdEntry:
    __slots__ = ("code", "names", "complete", "main_ok", "sex", "age_max", "age_min")


@lru_cache(maxsize=None)
def kcd_master():
    """{상병기호(점 없이, 대문자): KcdEntry}. 같은 기호의 여러 줄은 첫 줄이 공식 병명, 나머지는 동의어.
    열: 완전코드구분(N=불완전) 주상병사용구분(N=주상병 불가) 성별구분(X=여성만, Y=남성만) 상한연령 하한연령(세)."""
    out = {}
    with (DATA / "hira_kcd_master_2025_9.csv").open(encoding="cp949", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        for code, kor, eng, complete, main_use, _, sex, age_max, age_min, *_ in rows:
            key = code.strip().replace(".", "").upper()
            e = out.get(key)
            if e is None:
                e = KcdEntry()
                e.code, e.names = key, []
                e.complete, e.main_ok = complete.strip() != "N", main_use.strip() != "N"
                e.sex = {"X": "F", "Y": "M"}.get(sex.strip())
                e.age_max = int(age_max) if age_max.strip() else None
                e.age_min = int(age_min) if age_min.strip() else None
                out[key] = e
            e.names.append(kor.strip())
    return out


def normalize_code(code):
    return str(code).strip().replace(".", "").replace(" ", "").upper()


def region_tax_code(address):
    """주소가 rules_data.regions(한 세무서가 구 전체를 관할하는 곳)에 있으면 그 세무서 코드, 아니면 None."""
    for prefix, code in rules_data()["regions"]:
        if str(address).startswith(prefix + " ") or str(address).replace(" ", "").startswith(prefix.replace(" ", "")):
            return code
    return None
