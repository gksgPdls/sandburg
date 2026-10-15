"""[역할] 2024·2025 단가표에 약가·고시 금액 넣기: 그 해 약제급여목록 엑셀
[쓰는 곳] 사람이 실행 (build_book_prices.py 다음)
[입력 → 출력] 약제목록 엑셀 → config/prices_<연도>.yaml

심평원 약제급여목록표(엑셀) → 과거 연도 단가표(config/prices_<연도>.yaml)에 약제 단가·고시 금액 추가.

수가는 tools/build_book_prices.py 가 만든 표에 있고, 이 도구는 약제 항목을 같은 파일에 넣음.
  약제  약제급여목록및급여상한금액표 (hira.or.kr 약제급여목록표 게시판, pgmid=HIRAA030014050000)
        같은 주성분코드 품목의 상한금액 최빈값 (동률이면 낮은 값) — fetch_prices.py 와 같은 규칙
        여러 성분 합(경구약 1일분)은 성분별 단가 × 1일 복용 정수(n)의 합
        그 해 1월 1일 목록 값을 그 해 전체에 씀
  고시  제증명수수료 상한 (보건복지부고시 제2021-34호, 2021-02-05 시행) — price_sources.yaml 의 notice 금액 그대로

사용한 파일:
  2024  약제 (2024.1.1.)(22,887)_1.4.수정.xlsx  https://www.hira.or.kr/bbs/bbsCDownLoad.do?apndNo=1&apndBrdBltNo=1674&apndBrdTyNo=1&apndBltNo=59
  2025  약제 (2025.1.1.)(22,011)_공개용.xlsx  https://www.hira.or.kr/bbs/bbsCDownLoad.do?apndNo=1&apndBrdBltNo=1686&apndBrdTyNo=1&apndBltNo=59

실행: .venv/bin/python tools/build_list_prices.py <연도> <약제목록.xlsx>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import openpyxl
import yaml

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("clinic", "hospital", "general")


def mode(prices: list) -> tuple:
    """최빈값 (동률이면 낮은 값) → (값, 그 값 품목 수, 전체 품목 수)."""
    cnt = Counter(prices)
    if not cnt:
        return 0, 0, 0
    p = min(cnt, key=lambda x: (-cnt[x], x))
    return p, cnt[p], len(prices)


def read_rows(path: Path) -> dict:
    """약제목록 엑셀 첫 시트 → {주성분코드: [상한금액, ...]} (상한금액이 숫자이고 0보다 큰 행만)."""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = wb.worksheets[0].iter_rows(values_only=True)
    hdr = list(next(rows))
    k, p = hdr.index("주성분코드"), hdr.index("상한금액")
    out = {}
    for r in rows:
        if isinstance(r[p], (int, float)) and r[p] > 0:
            out.setdefault(str(r[k]).strip(), []).append(int(r[p]))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("실행: ")[1])
    year, drug_xlsx = int(sys.argv[1]), Path(sys.argv[2])
    dst = ROOT / "config" / f"prices_{year}.yaml"
    doc = yaml.safe_load(dst.read_text(encoding="utf-8"))
    rules = yaml.safe_load((ROOT / "config" / "price_sources.yaml").read_text(encoding="utf-8"))["items"]
    drugs = read_rows(drug_xlsx)
    failed = []
    for item, rule in rules.items():
        parts = rule if isinstance(rule, list) else [rule]
        if "notice" in parts[0]:   # 보건복지부 고시 금액 (제증명수수료 상한) — 연도와 무관하게 고시 금액 그대로
            q = parts[0]
            doc["items"][item] = {"source": f"보건복지부 고시 {q['label']}", "name": q["name"], "from": None,
                                  "price": {t: int(q["notice"]) for t in TYPES}, "basis": q["ref"]}
            continue
        if "drug" not in parts[0]:
            continue
        got = []
        for q in parts:
            price, same, total = mode(drugs.get(q["drug"], []))
            if not price:
                print(f"  {item:12s} 실패: {q['drug']} 목록에 없음")
                failed.append(item)
                break
            got.append((q, price, same, total))
        else:
            n = lambda q: q.get("n", 1)   # noqa: E731 — 1일 복용 정수 등 수량 배수
            doc["items"][item] = {
                "source": " + ".join(f"심평원 약가 {q['drug']}" + (f"×{n(q)}" if n(q) > 1 else "") for q, *_ in got),
                "name": " + ".join(q["name"] for q, *_ in got),
                "from": f"{year}-01-01",
                "price": {t: sum(pr * n(q) for q, pr, *_ in got) for t in TYPES},
                "basis": " / ".join(f"{drug_xlsx.name}: {s_}/{tot}개 품목 {pr:,}원" for q, pr, s_, tot in got),
            }
            print(f"  {item:12s} {doc['items'][item]['price']['clinic']:,}원  {doc['items'][item]['basis']}")
    doc["items"] = {k: doc["items"][k] for k in rules if k in doc["items"]}   # price_sources 순서대로
    dst.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n완료: {dst.relative_to(ROOT)} ({len(doc['items'])} 항목)")
    if failed:
        sys.exit("만들지 못한 항목: " + ", ".join(failed))


if __name__ == "__main__":
    main()
