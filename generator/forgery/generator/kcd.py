"""[역할] 진단 코드 점검: 심평원 상병마스터 완전코드·공식 병명인지 (시작 점검)
[쓰는 곳] main.py, tools/verify_cases.py
[입력 → 출력] assets/references/hira_kcd_master_2025_9.csv

진단 코드 점검: codebook 의 모든 진단이 심평원 상병마스터의 완전코드이고 병명·영문명이 상병마스터 표기와 같은지.

상병마스터: assets/references/hira_kcd_master_2025_9.csv
  공공데이터포털 「건강보험심사평가원_상병마스터」 2025.9 (https://www.data.go.kr/data/15067467/fileData.do)
  열: 상병기호, 한글명, 영문명, 완전코드구분(N = 하위 코드가 있는 불완전코드) …
  같은 상병기호가 여러 줄이면 첫 줄이 공식 병명 (나머지는 동의어)
불완전코드는 진단서에 쓸 수 없으므로 codebook 에 있으면 생성하지 않고 멈춤.
"""
import csv
from pathlib import Path

MASTER = Path(__file__).resolve().parent.parent / "assets" / "references" / "hira_kcd_master_2025_9.csv"


def load_master() -> dict:
    """{상병기호(점 없이): (한글명, 영문명, 완전코드 여부)} — 상병기호별 첫 줄."""
    out = {}
    with MASTER.open(encoding="cp949", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        for code, kor, eng, complete_flag, *_ in rows:   # 완전코드구분: N = 불완전코드
            out.setdefault(code.strip().replace(".", ""), (kor.strip(), eng.strip(), complete_flag.strip() != "N"))
    return out


def check_diagnoses(cb: dict) -> None:
    master = load_master()
    errs = []
    for dept, d in cb["departments"].items():
        for dx in d["diagnoses"]:
            m = master.get(dx["code"].replace(".", ""))
            if m is None:
                errs.append(f"{dept} {dx['code']}: 상병마스터에 없음")
            elif not m[2]:
                errs.append(f"{dept} {dx['code']}: 불완전코드 (하위 코드를 써야 함)")
            elif (dx["name"], dx["en"]) != m[:2]:
                errs.append(f"{dept} {dx['code']}: 병명이 상병마스터와 다름 → '{m[0]}' / '{m[1]}'")
    if errs:
        raise SystemExit("codebook 진단 점검 실패:\n  " + "\n  ".join(errs))
