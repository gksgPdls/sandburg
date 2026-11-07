"""[역할] 2024·2025 단가표 만들기 (수가·식대): 그 해 「건강보험요양급여비용」 책자 PDF
[쓰는 곳] 사람이 실행
[입력 → 출력] 책자 PDF → config/prices_<연도>.yaml

심평원 「건강보험요양급여비용」 책자(연도별 PDF) → 그 해 수가 단가 (config/prices_<연도>.yaml).

오픈 API 는 현재 적용 단가만 주므로 과거 연도 수가는 그 해 책자에서 만듦:
  행위·입원료  책자의 상대가치점수 × 그 해 환산지수 (10원 단위 반올림)
               2026 API 단가 1,295건 전부 이 계산과 일치 (의원 = 의원 환산지수, 병원·종합병원 = 병원 환산지수)
  식대         책자에 금액(원)으로 고시됨 → 그대로
  책자 표에 점수 칸이 없는 코드 (DERIVED)
    표층열치료(심층열 동시) MM015 = MM010 소정점수의 50% (사-101 주2)
    간호등급별 입원료: 책자 점수표에는 기준 등급(종합병원 1등급 AB2?1, 병원 4등급 AB3?4)만 있고 나머지 등급은
      산정지침의 가감률로 정해짐 → 기준 등급 책자 점수 × (2026 API 의 그 등급 점수 / 기준 등급 점수)
    진찰료 야간(…010)·공휴일(…050) = 진찰료 점수 + 기본진찰료 × 30%, 토요일 오전(…030) = 기본진찰료 × 30%
      (가-1 주3·주5. 기본진찰료 점수는 책자 제1장 산정지침 1-가-⑴ 에서 읽음: 초진 155.57, 재진 98.03)
    의원 처치 야간·토요·공휴 가산(…080) = 소정점수 × 1.3 (제9장 산정지침 ⒇, 별표8 항목)
대상은 config/price_sources.yaml 의 수가(fee) 항목만. 약제·고시 금액은 이 도구로 만들지 않음
(기존 파일의 약제·고시 항목은 그대로 둠 → tools/build_list_prices.py 를 다시 돌리지 않아도 됨).

책자 (hira.or.kr 전자책):
  2024  https://repository.hira.or.kr/handle/2019.oak/3201   (건강보험요양급여비용 2024년 1월판)
  2025  https://www.hira.or.kr/ebooksc/2025/02/BZ202502271254541.pdf   (2025년 1월 중 고시 반영)

실행: .venv/bin/python tools/build_book_prices.py 2025 <책자.pdf>
결과: config/prices_2025.yaml  (pricing.py 가 진료일 연도에 맞춰 읽음)
      assets/references/price_evidence_2025.xlsx  (항목별 근거 기록)
"""
from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "references" / "hira_api"
TYPES = ("clinic", "hospital", "general")
# 연도별 환산지수 (의원, 병원) — 건보공단 연도별 환산지수 현황 https://www.nhis.or.kr/announce/wbhaec11413m01.do
# (2020 은 그 표에 없어 보도자료 값). 병원 = 병원·종합병원 공통
CONVERSION = {2020: (85.8, 76.2), 2021: (87.6, 77.3), 2022: (90.2, 78.4), 2023: (92.1, 79.7), 2024: (93.6, 81.2),
              2025: (94.1, 82.2), 2026: (95.6, 83.8)}
# 책자에 자기 점수 칸이 없고 다른 코드의 소정점수를 쓰는 코드
SAME_AS = {"MM080": "MM070"}   # 간섭파전류치료: 경피적전기신경자극치료(사-104) 주 "소정점수를 산정한다"
# 입원료 병실 문자 → 그 병실의 기준 문자 (4인실 J·K·L·M, 3인실 N·P·Q·R, 2인실 S·T·U·V; 뒤 셋은 최하위 등급 소재지 구분)
ROOM_GROUP = {c: g for g, cs in (("J", "JKLM"), ("N", "NPQR"), ("S", "STUV")) for c in cs}
NUM = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
CODE_LINE = re.compile(r"[A-Z]{1,2}\d{3,4}[0-9A-Z]?\*?(?:\s|$)")


def basic_consult(text: str) -> dict:
    """책자 제1장 산정지침의 기본진찰료 점수 → {"AA1": 초진, "AA2": 재진} (의원·병원·종합병원 공통)."""
    first = re.search(r"초진의\s*경우\s*AA15\d~AA157은\s*([\d.]+)점", text)
    revisit = re.search(r"재진의\s*경우\s*AA25\d~AA257은\s*([\d.]+)점", text)
    if not (first and revisit):
        raise LookupError("책자에서 기본진찰료 점수를 찾지 못함")
    return {"AA1": Decimal(first.group(1)), "AA2": Decimal(revisit.group(1))}


def derived(code: str):
    """책자 점수표에 없는 코드 → (기준 코드, 비율) 또는 None. 진찰료 시간 가산은 resolve() 에서 따로 계산."""
    if code == "MM015":                     # 사-101 주2 "소정점수의 50%"
        return "MM010", Decimal("0.5")
    if code.startswith("S") and code.endswith("080"):   # 의원 야간·토요·공휴 처치 가산 30%
        return code[:5], Decimal("1.3")
    if code[:3] in ("AB2", "AB3") and len(code) == 5 and code[3] in ROOM_GROUP:
        base = code[:3] + ROOM_GROUP[code[3]] + ("1" if code[:3] == "AB2" else "4")
        if base != code:                    # 산정지침 가감률 = 2026 API 점수 비율
            return base, Decimal(api_2026(code)["cvalPnt"]) / Decimal(api_2026(base)["cvalPnt"])
    return None


def book_value(text: str, code: str) -> float | None:
    """책자 텍스트에서 코드의 상대가치점수 (식대는 금액).

    표 형식: 코드가 줄 첫머리(분류번호 뒤 포함)에 오고 이어지는 몇 줄 안에 숫자만 있는 줄 → 그 값.
    다음 코드나 '주' 가 먼저 나오면 그 코드 칸은 비어 있는 것.
    """
    for m in re.finditer(rf"(?m)^(?:[가-힣]-[\d-]+ )?{code}\b", text):
        lines = text[m.end():m.end() + 400].split("\n")
        for i, s in enumerate(lines[:8]):
            s = s.strip()
            if i and (CODE_LINE.match(s) or s.startswith("주")):
                break
            if NUM.fullmatch(s):
                return float(s.replace(",", ""))
    return None


def won10(points: float, factor: float) -> int:
    return int((Decimal(str(points)) * Decimal(str(factor)) / 10).quantize(0, ROUND_HALF_UP)) * 10


def api_2026(code: str) -> dict:
    """2026 API 응답 (fetch_prices.py 캐시) — 종별 칸(의원/병원) 구분과 점수 비교용."""
    rows = json.loads((RAW / f"fee_{code}.json").read_text(encoding="utf-8"))
    return next(r for r in rows if r["mdfeeCd"] == code)


def resolve(text: str, code: str, year: int) -> dict:
    """수가코드 1개 → 그 해 종별 단가와 근거."""
    now = api_2026(code)
    p1, p2 = int(now["unprc1"] or 0), int(now["unprc2"] or 0)
    if code[:2] == "AA" and code[5:] in ("010", "030", "050"):   # 진찰료 야간·토요·공휴일 (가-1 주3·주5)
        add = (basic_consult(text)[code[:3]] * Decimal("0.3")).quantize(Decimal("0.01"), ROUND_HALF_UP)
        if code[5:] == "030":
            v, note = float(add), f"기본진찰료 × 30% = {add}"
        else:
            b = book_value(text, code[:5])
            v = float(Decimal(str(b)) + add)
            note = f"진찰료 {b} + 기본진찰료 × 30% {add}"
        k1, k2 = CONVERSION[year]
        c, h = (won10(v, k1) if p1 else 0), (won10(v, k2) if p2 else 0)
        return {"code": code, "name": now.get("korNm", "").rstrip("/"),
                "price": {"clinic": c or h, "hospital": h or c, "general": h or c},
                "note": f"상대가치점수 {v} (2026 {now['cvalPnt']}) ← {note}"}
    look, ratio = derived(code) or (SAME_AS.get(code, code), None)
    v = book_value(text, look) if len(look) == 5 else book_value(text, look[:5])   # 8자리 = 5자리 + 세부항목
    if v is None:
        raise LookupError(f"책자에서 {look} 를 찾지 못함")
    if float(now["cvalPnt"] or 0) == 0:     # 식대: 점수 없이 금액 고시
        return {"code": code, "name": now.get("korNm", "").rstrip("/"), "price": {t: int(v) for t in TYPES},
                "note": f"고시금액 {int(v):,}원"}
    if ratio is not None:   # 가감률 적용 총 점수: 소수점 셋째 자리에서 4사5입 (제1부 Ⅰ.2)
        v = float((Decimal(str(v)) * ratio).quantize(Decimal("0.01"), ROUND_HALF_UP))
    k1, k2 = CONVERSION[year]
    c, h = won10(v, k1), won10(v, k2)
    c, h = (c if p1 else 0), (h if p2 else 0)    # 2026 에 없는 종별 칸은 만들지 않음 (fetch_prices 와 같은 규칙)
    price = {"clinic": c or h, "hospital": h or c, "general": h or c}
    note = f"상대가치점수 {v} (2026 {now['cvalPnt']})" + (
        f" ← {look} 책자 점수 × {ratio:.4f}" if ratio is not None else f" ← {look} 소정점수" if look != code else "")
    return {"code": code, "name": now.get("korNm", "").rstrip("/"), "price": price, "note": note}


def write_evidence(year: int, doc: dict) -> None:
    """그 해 단가표의 항목별 근거를 엑셀로 (assets/references/price_evidence_<연도>.xlsx) — 사람이 확인·캡처하는 기록."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = f"{year} 단가 근거"
    ws.append(["항목", "출처", "명칭", "적용일", "의원 단가", "병원·종합병원 단가", "근거 (책자 점수 × 환산지수 / 고시금액 / 약제목록)"])
    for k, v in doc["items"].items():
        ws.append([k, v["source"], v.get("name", ""), v.get("from"), v["price"]["clinic"], v["price"]["hospital"],
                   v.get("basis", "")])
    wb.save(ROOT / "assets" / "references" / f"price_evidence_{year}.xlsx")


def main():
    if len(sys.argv) != 3:
        sys.exit("사용법: .venv/bin/python tools/build_book_prices.py <연도> <책자.pdf>")
    year, pdf = int(sys.argv[1]), Path(sys.argv[2])
    if year not in CONVERSION:
        sys.exit(f"{year}년 환산지수가 CONVERSION 에 없음")
    text = "\n".join(p.get_text() for p in pymupdf.open(pdf))
    rules = yaml.safe_load((ROOT / "config" / "price_sources.yaml").read_text(encoding="utf-8"))["items"]
    out, failed, changed = {}, [], []
    for item, rule in rules.items():
        parts = rule if isinstance(rule, list) else [rule]
        if not all("fee" in p for p in parts):
            continue
        try:
            got = [(resolve(text, p["fee"], year), p.get("n", 1)) for p in parts]
        except LookupError as e:
            print(f"  {item:26s} 실패: {e}")
            failed.append(item)
            continue
        out[item] = {
            "source": f"심평원 {year} 요양급여비용 " + " + ".join(g["code"] + (f"×{n}" if n > 1 else "") for g, n in got),
            "name": " + ".join(g["name"] for g, _ in got),
            "from": f"{year}-01-01",
            "price": {t: sum(g["price"][t] * n for g, n in got) for t in TYPES},
            "basis": " / ".join(g["note"] for g, _ in got),
        }
        changed += [g["code"] for g, _ in got if "2026" in g["note"]
                    and float(g["note"].split()[1]) != float(g["note"].split()[3].rstrip(")"))]
        print(f"  {item:26s} 의원 {out[item]['price']['clinic']:>7,} / 병원 {out[item]['price']['hospital']:>7,}원  {out[item]['basis']}")

    k1, k2 = CONVERSION[year]
    doc = {"_설명": f"심평원 「{year} 건강보험요양급여비용」 책자({pdf.name})의 상대가치점수 × {year} 환산지수"
                    f"(의원 {k1}, 병원 {k2}), 10원 단위 반올림. 식대는 책자 고시금액. 약가는 tools/build_list_prices.py. "
                    "만든 도구: tools/build_book_prices.py",
           "valid_from": f"{year}-01-01",
           "items": out}
    dst = ROOT / "config" / f"prices_{year}.yaml"
    if dst.exists():   # 약제·치료재료·비급여·제증명 등 수가 아닌 항목은 기존 파일 값을 그대로 둠 (price_sources 순서대로)
        old = yaml.safe_load(dst.read_text(encoding="utf-8"))["items"]
        keep = {k: v for k, v in old.items() if k in rules and not all("fee" in q for q in
                                                                      (rules[k] if isinstance(rules[k], list) else [rules[k]]))}
        doc["items"] = {k: out.get(k, keep.get(k)) for k in rules if k in out or k in keep}
    dst.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_evidence(year, doc)
    print(f"\n완료: {len(out)} 항목 → {dst.relative_to(ROOT)}")
    if changed:
        print("2026 과 점수가 다른 코드:", ", ".join(changed))
    if failed:
        print("만들지 못한 항목:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
