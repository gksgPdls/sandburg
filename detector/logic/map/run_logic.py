"""회사 실제 문서(Downloads/samples)를 메인 로직 엔진(logic_detector, 로컬 OCR)으로 검사해 결과를 모은다. 외부 호출 없음.

    python run_logic.py <메인 코드 폴더> <samples 폴더> <결과.jsonl>
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CODE, SAMPLES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
import _compat  # noqa: E402,F401  (logic_detector 찾기)


def jobs():
    for lab in sorted(SAMPLES.glob("*/labels.csv")):
        for r in csv.DictReader(open(lab, encoding="utf-8-sig")):
            d = lab.parent
            kind = "diagnosis" if "진단" in r["doc_type"] else "receipt"
            base = {"id": r["id"], "method": r["method"], "kind": kind, "desc": r["description"],
                    "mask": str(d / r["mask"]) if r.get("mask") else "", "original": str(d / r["original"]) if r.get("original") else ""}
            if r.get("tampered") and (d / r["tampered"]).exists():
                yield {**base, "normal": False, "path": str(d / r["tampered"])}
            if r.get("original") and (d / r["original"]).exists():
                yield {**base, "normal": True, "path": str(d / r["original"]), "id": r["id"] + "_orig"}


def run(j):
    from logic_detector.logic import inspect
    try:
        r = inspect(j["path"], j["kind"])
        return {**j, "verdict": r["verdict"], "findings": r["findings"], "notes": r.get("notes", []),
                "reading": {k: (v if isinstance(v, (str, int, float)) or v is None else str(v)) for k, v in r.get("reading", {}).items()}}
    except Exception as e:  # 한 장이 실패해도 나머지는 계속
        return {**j, "verdict": "오류", "error": f"{type(e).__name__}: {e}", "findings": []}


if __name__ == "__main__":
    js = list(jobs())
    # 같은 원본이 여러 조작에 쓰였으면 한 번만
    seen, uniq = set(), []
    for j in js:
        if j["path"] not in seen:
            seen.add(j["path"]); uniq.append(j)
    print(f"{len(uniq)}장 검사", flush=True)
    with open(OUT, "w", encoding="utf-8") as f, ProcessPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(run, uniq), 1):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); f.flush()
            if i % 25 == 0:
                print(i, flush=True)
    print("끝", flush=True)
