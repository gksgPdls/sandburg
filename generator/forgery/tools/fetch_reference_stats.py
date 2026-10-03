"""심평원 통계 API → 처방 규칙 점검용 참고 통계 (assets/references/reference_stats.xlsx).

생성기는 이 통계를 직접 읽지 않음. codebook.yaml 의 처방 묶음·약제 선택이 실제 진료 통계와 어긋나지 않는지
사람이 확인하는 근거 자료 (보고서 부록).

사용 API (공공데이터포털, config/hira_api_key.txt 인증키로 활용신청 필요):
  건강보험심사평가원_진료행위정보서비스      https://www.data.go.kr/data/15128423/openapi.do
    mdlrtActionInfoService/getMdlrtActionByClassesStats       행위코드(5자리)별 요양기관 종별 환자수·사용량
    mdlrtActionInfoService/getMdlrtActionByGenderIpatOpatStats 행위코드별 입원·외래 환자수
  건강보험심사평가원_의약품사용정보조회서비스 https://www.data.go.kr/data/15047819/openapi.do
    msupUserInfoService1.2/getCmpnSickList1.2                  주성분코드별 상병(3자리)별 사용량

실행: .venv/bin/python tools/fetch_reference_stats.py [--fresh]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_prices import RAW, ROOT, api_key, cached, call, won  # noqa: E402

ACTION_YEAR = "2025"     # 진료행위 통계 최신 연도
DRUG_YM = "202512"       # 의약품 상병별 사용량 최신 월
ACT = "mdlrtActionInfoService/"
MSUP = "msupUserInfoService1.2/getCmpnSickList1.2"
CLASSES = ["의원급", "병원급", "종합병원", "상급종합병원", "보건기관등"]


def parts_of(rule) -> list:
    return rule if isinstance(rule, list) else [rule]


def main():
    key = api_key()
    RAW.mkdir(parents=True, exist_ok=True)
    rules = yaml.safe_load((ROOT / "config" / "price_sources.yaml").read_text(encoding="utf-8"))["items"]
    cb = yaml.safe_load((ROOT / "config" / "codebook.yaml").read_text(encoding="utf-8"))

    # 약제 항목 → 그 약을 쓰는 진단코드(3자리): codebook 처방 묶음에서 역추적
    drug_dx = defaultdict(set)
    for dept in cb["departments"].values():
        for s in dept["orders"] + dept.get("inpatient_orders", []):
            dxs = s.get("dx") or [d["code"] for d in dept["diagnoses"]]
            for code, *_ in s["items"]:
                if code.startswith(("D_", "MED_PO")):
                    drug_dx[code].update(d.replace(".", "")[:3] for d in dxs)

    from openpyxl import Workbook
    wb = Workbook()

    # ── 1. 진료행위: 종별·입원외래 ──────────────────────────
    ws = wb.active
    ws.title = f"진료행위 {ACTION_YEAR}"
    ws.append(["항목", "행위코드(5자리)", "행위명", *[f"{c} 환자수" for c in CLASSES], "의원급 비율",
               "입원 환자수", "외래 환자수", "외래 비율"])
    done = set()
    for item, rule in rules.items():
        for p in parts_of(rule):
            if "fee" not in p:
                continue
            st5 = p["fee"][:5]
            if st5 in done:
                continue
            done.add(st5)
            cls = cached(f"stat_action_cl_{st5}_{ACTION_YEAR}", lambda: call(
                key, ACT + "getMdlrtActionByClassesStats", year=ACTION_YEAR, stdType="1", st5Cd=st5))
            io = cached(f"stat_action_io_{st5}_{ACTION_YEAR}", lambda: call(
                key, ACT + "getMdlrtActionByGenderIpatOpatStats", year=ACTION_YEAR, stdType="1", st5Cd=st5))
            if not cls:
                ws.append([item, st5, "(통계 없음)"])
                continue
            by = {r.get("diagCdNm"): won(r.get("ptntCnt")) for r in cls}
            total = by.get("계") or sum(by.get(c, 0) for c in CLASSES)
            ipat = sum(won(r.get("ptntCnt")) for r in io if r.get("ipatOpatDgsTpCdNm") == "입원")
            opat = sum(won(r.get("ptntCnt")) for r in io if r.get("ipatOpatDgsTpCdNm") == "외래")
            ws.append([item, st5, cls[0].get("st5CdNm"), *[by.get(c, 0) for c in CLASSES],
                       round(by.get("의원급", 0) / total, 3) if total else None, ipat, opat,
                       round(opat / (ipat + opat), 3) if ipat + opat else None])
            print(f"  행위 {st5:6s} {cls[0].get('st5CdNm', '')[:24]:26s} 환자 {total:>10,}", flush=True)

    # ── 2. 의약품: 상병별 사용량 → codebook 진단과 맞는지 ──
    ws = wb.create_sheet(f"약제-상병 {DRUG_YM}")
    ws.append(["항목", "주성분코드", "성분명", "codebook 사용 진단(3자리)", "해당 진단 사용량 순위",
               "상위 상병 1", "상위 상병 2", "상위 상병 3", "상위 상병 4", "상위 상병 5"])
    for item, rule in rules.items():
        for p in parts_of(rule):
            if "drug" not in p:
                continue
            rows = cached(f"stat_drug_sick_{p['drug']}_{DRUG_YM}", lambda: call(
                key, MSUP, rows=1000, diagYm=DRUG_YM, gnlNmCd=p["drug"], insupTp="0", cpmdPrscTp="01"))
            qty = defaultdict(int)
            names = {}
            for r in rows:                        # 보험자구분(건강보험·의료급여 등) 합산
                qty[r["st3SickSym"]] += won(r.get("totUseQty"))
                names[r["st3SickSym"]] = r.get("st3SickSymNm", "")
            ranked = sorted(qty, key=lambda c: -qty[c])
            # 상병기호 앞의 'A'(양방) 표시를 떼고 KCD 3자리로 비교
            kcd = {c: re.sub(r"^[A-Z](?=[A-Z]\d\d$)", "", c) for c in ranked}
            mine = sorted(drug_dx.get(item, set()))
            ranks = [f"{d}:{next((i + 1 for i, c in enumerate(ranked) if kcd[c] == d), '-')}" for d in mine]
            ws.append([item, p["drug"], rows[0]["gnlNmCdNm"] if rows else "", ", ".join(mine), ", ".join(ranks),
                       *[f"{kcd[c]} {names[c]} ({qty[c]:,})" for c in ranked[:5]]])
            print(f"  약제 {p['drug']} {item:9s} 진단순위 {', '.join(ranks)}", flush=True)

    out = ROOT / "assets" / "references" / "reference_stats.xlsx"
    wb.save(out)
    print(f"\n완료 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
