"""
평가셋 모든 이미지의 글자 칸(RapidOCR)을 미리 읽어 둔다 (A 검사 평가가 매번 OCR을 안 하게).

  python detector/pixel/scripts/v2/ocr_cache.py [--max-seconds 150] [--workers 4]   # 끊기면 이어서
결과: outputs/평가셋/ocr.jsonl  한 줄 = {"id", "cells": [...]}
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("EVALSET_DIR", ROOT / "outputs" / "평가셋"))   # 시험할 때 다른 폴더로
CACHE = OUT / "ocr.jsonl"


def work(row):
    from detector.pixel.common.cells import read_cells
    try:
        return {"id": row["id"], "cells": read_cells(ROOT / row["path"])}
    except Exception as e:  # noqa: BLE001
        return {"id": row["id"], "skip": f"{type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig")))
    done = {json.loads(l)["id"] for l in open(CACHE, encoding="utf-8")} if CACHE.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"OCR: 전체 {len(rows)}, 남은 {len(todo)}", flush=True)
    t0 = time.time()
    n = 0
    with open(CACHE, "a", encoding="utf-8") as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(work, todo, chunksize=2):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {len(done) + n}/{len(rows)}  ({time.time() - t0:.0f}초)", flush=True)
            if time.time() - t0 > a.max_seconds:
                pool.terminate()
                print(f"시간 제한 - {len(done) + n}/{len(rows)}, 다시 실행하면 이어서")
                return
    print(f"완료 {len(done) + n}/{len(rows)}")


if __name__ == "__main__":
    main()
