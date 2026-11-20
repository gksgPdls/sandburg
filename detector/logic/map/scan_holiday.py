"""poc-1-raw 영수증 중 공휴일(일요일 포함) 진료 후보를 로컬 OCR 로 읽어 진료일 · 공휴일 체크를 확인. 외부 호출 없음.

    python scan_holiday.py <결과.json> [파일이름 날짜 기준만: 1]
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

CODE = Path.home() / "Desktop/SCRYN/3_탐지기/scryn-combine-web"
sys.path.insert(0, str(CODE))
import _compat  # noqa: E402,F401  (logic_detector 찾기)
RAW = Path.home() / "Desktop/SCRYN/2_데이터셋/원본_실서류_poc-1-raw"


def run(p):
    from logic_detector.logic import inspect
    try:
        r = inspect(p, "receipt")
    except Exception as e:  # noqa: BLE001
        return {"path": p, "error": str(e)}
    rd = r.get("reading", {})
    keep = ("visit.start", "visit.end", "visit.holiday", "visit.night", "issue_date", "visit.inpatient")
    return {"path": p, "verdict": r["verdict"], "findings": [(f["rule"], f["grade"], f["title"]) for f in r["findings"]],
            "reading": {k: str(rd.get(k)) for k in keep if k in rd}}


if __name__ == "__main__":
    from logic_detector.logic import refdata as R
    if len(sys.argv) > 2:   # 파일 목록
        fs = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    else:
        fs = []
        for p in sorted(RAW.rglob("*DTB210*")):
            m = re.search(r"_(\d{8})_DTB210", p.name)
            if m and R.is_holiday(date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:]))):
                fs.append(str(p))
    print(len(fs), "장", flush=True)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run, fs))
    json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("끝")
