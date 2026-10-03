"""심평원 오픈 API → 수가 항목별 공식 단가 (config/prices.yaml).

출처 규칙: config/price_sources.yaml
사용 API (공공데이터포털, 같은 인증키로 각각 활용신청 필요):
  fee   건강보험심사평가원_수가기준정보조회서비스     https://www.data.go.kr/data/15021028/openapi.do
        mdfeeCrtrInfoService/getDiagnossMdfeeList          수가코드로 조회
        unprc1 = 의원, unprc2 = 병원·종합병원 (진찰료·입원료·식대는 종별 코드가 따로 있어 한쪽 칸만 값이 있음)
  drug  건강보험심사평가원_약가기준정보조회서비스     https://www.data.go.kr/data/15054445/openapi.do
        dgamtCrtrInfoService1.2/getDgamtList               품목명으로 조회 → 주성분코드로 거름
        급여구분 '급여'·'임의불가' 품목의 상한금액(mxCprc) 최빈값
  mat   건강보험심사평가원_치료재료정보조회서비스     https://www.data.go.kr/data/3074384/openapi.do
        mcatInfoService1.2/getPaymentNonPaymentList1.2     품목명으로 조회 → 중분류명으로 거름
        급여 품목의 상한단가(mxUnprc) 최빈값
  npay  건강보험심사평가원_비급여진료비정보조회서비스 https://www.data.go.kr/data/15001700/openapi.do
        nonPaymentDamtInfoService/getNonPaymentItemClcdList      비급여코드별 종별 최소·평균·중간값·최대 (전체 1회 조회)
        병원(prc*Hosp) / 종합병원(prc*Gnhp) 의 최소·중간값·최대 = 생성기가 뽑는 범위
        ※ 병원급 이상만 공개 → 의원은 비급여 단가 없음
  notice 보건복지부 고시 금액 (API 아님) — 제증명수수료 상한

인증키: config/hira_api_key.txt (한 줄, Encoding/Decoding 키 모두 가능) 또는 환경변수 HIRA_API_KEY

실행: .venv/bin/python tools/fetch_prices.py            (이전에 받은 응답 재사용)
      .venv/bin/python tools/fetch_prices.py --fresh    (전부 새로 받기 — 단가 개정 시)
결과:
  config/prices.yaml                       생성기가 쓰는 공식 단가
  assets/references/price_evidence.xlsx    항목별 근거 (코드·명칭·적용일·품목 수·단가)
  assets/references/hira_api/*.json        API 원본 응답
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "references" / "hira_api"
FRESH = "--fresh" in sys.argv
BASE = "https://apis.data.go.kr/B551182/"
SEARCH = {   # API 종류 → (경로, 검색 변수)
    "fee": ("mdfeeCrtrInfoService/getDiagnossMdfeeList", "mdfeeCd"),
    "drug": ("dgamtCrtrInfoService1.2/getDgamtList", "itmNm"),
    "mat": ("mcatInfoService1.2/getPaymentNonPaymentList1.2", "itmNm"),
}
NPAY = "nonPaymentDamtInfoService/"
NPAY_TYPE = {"hospital": "Hosp", "general": "Gnhp"}   # 비급여 종별 통계 필드 접미사: 병원 / 종합병원
SOURCE_LABEL = {"fee": "심평원 수가", "drug": "심평원 약가", "mat": "심평원 치료재료", "notice": "보건복지부 고시"}
TYPES = ("clinic", "hospital", "general")


def api_key() -> str:
    key = os.environ.get("HIRA_API_KEY", "").strip()
    f = ROOT / "config" / "hira_api_key.txt"
    if not key and f.exists():
        key = f.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit("인증키가 없습니다: config/hira_api_key.txt 에 공공데이터포털 인증키를 한 줄로 저장하세요.")
    # 포털의 Encoding 키(%2F, %3D 포함)를 넣어도 되도록 Decoding 키로 되돌림 (요청 시 다시 인코딩함)
    return urllib.parse.unquote(key) if "%" in key else key


def call(key: str, path: str, rows: int = 500, **params) -> list:
    """API 호출 → item 목록 (페이지 끝까지)."""
    items, page = [], 1
    while True:
        root = None
        for wait in (0, 2, 5, 10):                  # 일시적 오류는 잠시 후 재시도
            q = {"ServiceKey": key, "numOfRows": rows, "pageNo": page, **params}
            url = BASE + path + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
            try:
                time.sleep(wait)
                with urllib.request.urlopen(url, timeout=120) as resp:
                    root = ET.fromstring(resp.read())
                break
            except urllib.error.HTTPError as e:
                if "NOT_REGISTERED" in e.read().decode("utf-8", "replace"):
                    raise RuntimeError(f"인증키가 이 API에 활용신청되지 않음 ({path.split('/')[0]})")
            except (urllib.error.URLError, TimeoutError, ET.ParseError):
                continue
        if root is None:
            raise RuntimeError("API 응답 없음 (잠시 후 다시 실행)")
        code = (root.findtext(".//resultCode") or "").strip()
        if code not in ("00", ""):
            raise RuntimeError(f"API 오류 {code}: {root.findtext('.//resultMsg')}")
        batch = [{c.tag: (c.text or "").strip() for c in it} for it in root.iter("item")]
        items += batch
        total = int(root.findtext(".//totalCount") or 0)
        if not batch or len(items) >= total:
            return items
        if page % 20 == 0:
            print(f"    … {path.split('/')[-1]} {len(items):,}/{total:,}", flush=True)
        page += 1
        time.sleep(0.15)


def cached(name: str, load):
    """원본 응답을 hira_api/<name>.json 에 저장해 두고 재사용 (--fresh 면 새로 받음)."""
    raw = RAW / f"{name}.json".replace("/", "_")
    if raw.exists() and not FRESH:
        return json.loads(raw.read_text(encoding="utf-8"))
    res = load()
    raw.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def search(key: str, api: str, query: str) -> list:
    path, param = SEARCH[api]
    return cached(f"{api}_{query}", lambda: call(key, path, **{param: query}))


def npay_stats(key: str) -> dict:
    """비급여코드별 종별 가격 통계 전체 (한 번에 조회) → {비급여코드: item}."""
    rows = cached("npay_clcd", lambda: call(key, NPAY + "getNonPaymentItemClcdList", rows=1000))
    return {it["npayCd"]: it for it in rows}


def won(x) -> int:
    try:
        return int(float(x or 0))
    except ValueError:
        return 0


def iso(d: str):
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if d and len(d) == 8 else None


def mode_price(products: list, field: str) -> tuple:
    """품목들의 상한금액 최빈값 (동률이면 낮은 값) → (단가, 그 단가의 품목들)."""
    cnt = Counter(won(p[field]) for p in products if won(p[field]) > 0)
    if not cnt:
        return 0, []
    price = min(cnt, key=lambda p: (-cnt[p], p))
    return price, [p for p in products if won(p[field]) == price]


def resolve_part(key: str, part: dict, valid_from: str) -> dict:
    """출처 규칙 1개 → {api, code, name, from, price{종별}, evidence}. valid_from 이후에 바뀐 단가는 쓰지 않음."""
    if "fee" in part:
        code = part["fee"]
        hit = next((it for it in search(key, "fee", code) if it.get("mdfeeCd") == code), None)
        if not hit:
            raise LookupError(f"수가코드 {code} 없음")
        if iso(hit.get("adtStaDd", "")) > valid_from:
            raise LookupError(f"수가코드 {code} 적용시작일 {iso(hit['adtStaDd'])} 이 기준일 {valid_from} 이후")
        p1, p2 = won(hit.get("unprc1")), won(hit.get("unprc2"))
        return {"api": "fee", "code": code, "name": hit.get("korNm", "").rstrip("/"),
                "from": iso(hit.get("adtStaDd", "")),
                "price": {"clinic": p1 or p2, "hospital": p2 or p1, "general": p2 or p1},
                "evidence": f"상대가치점수 {hit.get('cvalPnt')} / 의원 {p1:,} / 병원 {p2:,}"}
    if "drug" in part:
        gnl = part["drug"]
        prods = [it for it in search(key, "drug", part["q"])
                 if it.get("gnlNmCd") == gnl and it.get("payTpNm") in ("급여", "임의불가")
                 and (iso(it.get("adtStaDd", "")) or "") <= valid_from]
        price, same = mode_price(prods, "mxCprc")
        if not price:
            raise LookupError(f"주성분코드 {gnl} 의 {valid_from} 이전 적용 급여 품목 없음 (검색어 '{part['q']}')")
        return {"api": "drug", "code": gnl, "name": part["name"],
                "from": min(iso(p.get("adtStaDd", "")) or "9999" for p in same),
                "price": {t: price for t in TYPES},
                "evidence": f"급여 품목 {len(prods)}개 중 {len(same)}개가 상한금액 {price:,}원"}
    if "mat" in part:
        mdiv = part["mat"]
        prods = [it for it in search(key, "mat", part["q"])
                 if it.get("mdivCdNm") == mdiv and it.get("payTpNm") == "급여"]
        price, same = mode_price(prods, "mxUnprc")
        if not price:
            raise LookupError(f"중분류 '{mdiv}' 의 급여 품목 없음 (검색어 '{part['q']}')")
        return {"api": "mat", "code": same[0]["mdivCd"], "name": mdiv, "from": None,
                "price": {t: price for t in TYPES},
                "evidence": f"급여 품목 {len(prods)}개 중 {len(same)}개가 상한단가 {price:,}원"}
    if "notice" in part:
        return {"api": "notice", "code": "제증명수수료 상한", "name": part["name"], "from": None,
                "price": {t: int(part["notice"]) for t in TYPES}, "evidence": part["ref"]}
    raise ValueError(f"알 수 없는 출처 규칙: {part}")


def resolve_npay(rule: dict, stats: dict) -> dict:
    """비급여 항목 → 종별 {min, med, max} (공표된 최소·중간값·최대). 의원은 공개 대상이 아니라 없음."""
    code = rule["npay"]
    st = stats.get(code)
    if not st:
        raise LookupError(f"비급여코드 {code} 없음")
    rng = {}
    for t, sfx in NPAY_TYPE.items():
        lo, med, hi = (won(st.get(f + sfx)) for f in ("prcMin", "middAvg", "prcMax"))
        if hi > 0:
            rng[t] = {"min": lo, "med": med, "max": hi}
    if not rng:
        raise LookupError(f"비급여코드 {code} 의 병원·종합병원 가격 없음")
    return {"code": code, "name": st.get("npayKorNm", ""), "range": rng, "std_date": iso(st.get("stdDate", ""))}


def main():
    key = api_key()
    src = yaml.safe_load((ROOT / "config" / "price_sources.yaml").read_text(encoding="utf-8"))
    rules, valid_from = src["items"], str(src["valid_from"])
    RAW.mkdir(parents=True, exist_ok=True)
    stats = None
    out, evidence, missing = {}, [], []
    for item, rule in rules.items():
        try:
            if isinstance(rule, dict) and "npay" in rule:
                stats = stats or npay_stats(key)
                r = resolve_npay(rule, stats)
                out[item] = {"source": f"심평원 비급여 {r['code']}", "name": r["name"], "from": None,
                             "range": r["range"]}
                for t, v in r["range"].items():
                    evidence.append([item, "심평원 비급여", r["code"], r["name"], 1, r["std_date"], t, None,
                                     f"공표 최소 {v['min']:,} / 중간값 {v['med']:,} / 최대 {v['max']:,}"])
                print(f"  {item:26s} {out[item]['source']:34s} "
                      + " / ".join(f"{t} {v['min']:,}~{v['med']:,}~{v['max']:,}" for t, v in r["range"].items()),
                      flush=True)
                continue
            parts = rule if isinstance(rule, list) else [rule]
            got = [(resolve_part(key, p, valid_from), p.get("n", 1)) for p in parts]
        except (LookupError, RuntimeError) as e:
            print(f"  {item:26s} 실패: {e}", flush=True)
            missing.append(item)
            continue
        price = {t: sum(g["price"][t] * n for g, n in got) for t in TYPES}
        froms = [g["from"] for g, _ in got if g["from"]]
        out[item] = {
            "source": " + ".join(f"{SOURCE_LABEL[g['api']]} {g['code']}" + (f"×{n}" if n > 1 else "") for g, n in got),
            "name": " + ".join(g["name"] for g, _ in got),
            "from": max(froms) if froms else None,
            "price": price,
        }
        for g, n in got:
            evidence.append([item, SOURCE_LABEL[g["api"]], g["code"], g["name"], n, g["from"],
                             g["price"]["clinic"], g["price"]["hospital"], g["evidence"]])
        print(f"  {item:26s} {out[item]['source'][:34]:34s} 의원 {price['clinic']:>7,} / 병원 {price['hospital']:>7,}원",
              flush=True)

    doc = {"_설명": f"심평원 오픈 API 조회 결과 ({date.today()}). 출처 규칙: config/price_sources.yaml, "
                    "근거: assets/references/price_evidence.xlsx. price = 종별 고정 단가, "
                    "range = 비급여 종별 공표 최소·중간값·최대 (생성 시 이 범위에서 뽑음)",
           "valid_from": valid_from,
           "items": out}
    (ROOT / "config" / "prices.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                                                 encoding="utf-8")
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "단가 근거"
    ws.append(["항목", "API", "코드", "명칭", "수량 배수", "적용시작일", "의원 단가(비급여는 종별)", "병원 단가", "근거"])
    for r in evidence:
        ws.append(r)
    wb.save(ROOT / "assets" / "references" / "price_evidence.xlsx")
    print(f"\n완료: {len(out)}/{len(rules)} 항목 → config/prices.yaml")
    if missing:
        print("단가를 받지 못한 항목:", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
