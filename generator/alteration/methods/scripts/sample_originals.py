#!/usr/bin/env python3
"""real_docs 에서 원본을 무작위로 뽑아 _work/pool/{DTB}/ 에 복사한다.

파일 이름의 DTB 코드로 문서 종류를 나눈다 (로컬 OCR 표본 확인 기준).
    DOC210  진료비 계산서·영수증
    DOC203  진단서·소견서
    DOC205  퇴원 영수증·세부내역 계열

- 시드를 고정해 같은 결과를 다시 뽑을 수 있다
- 긴 변이 --min-long 보다 작은 이미지는 거른다
- 이미 pool 에 있는 원본은 다시 뽑지 않는다
- 복사한 파일은 pool 순번으로 이름을 바꾸고, 원래 경로는 manifest.csv 에 남긴다

사용: sample_originals.py DOC210 30 [--seed 0]
"""

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

from PIL import Image

DATASET = Path(__file__).resolve().parent.parent
RAW = DATASET.parent / "real_docs"
POOL = DATASET / "_work" / "pool"
MANIFEST = POOL / "manifest.csv"
NAME = re.compile(r"_(surg|med|dig)_\w+?_(DTB\d+)_")


def read_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="원본 무작위 추출")
    p.add_argument("dtb", help="문서 종류 코드 (예: DOC210)")
    p.add_argument("count", type=int, help="뽑을 장수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-long", type=int, default=1000, help="긴 변 최소 px (기본 1000)")
    a = p.parse_args()

    files = sorted(f for f in RAW.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                   and (m := NAME.search(f.name)) and m.group(2) == a.dtb)
    if not files:
        raise SystemExit(f"{a.dtb} 원본이 없다 ({RAW})")

    rows = read_manifest()
    used = {r["source"] for r in rows}
    random.Random(a.seed).shuffle(files)

    out_dir = POOL / a.dtb
    out_dir.mkdir(parents=True, exist_ok=True)
    start = sum(r["dtb"] == a.dtb for r in rows)
    added, skipped = [], 0
    for f in files:
        if len(added) == a.count:
            break
        src = f.relative_to(RAW).as_posix()
        if src in used:
            continue
        with Image.open(f) as im:
            if max(im.size) < a.min_long:
                skipped += 1
                continue
            size = f"{im.size[0]}x{im.size[1]}"
        pool_id = f"{a.dtb}_{start + len(added) + 1:04d}"
        dst = out_dir / f"{pool_id}{f.suffix.lower()}"
        shutil.copy2(f, dst)
        added.append({"pool_id": pool_id, "dtb": a.dtb, "file": dst.relative_to(POOL).as_posix(),
                      "source": src, "size": size, "seed": a.seed})

    new = not MANIFEST.exists()
    with open(MANIFEST, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pool_id", "dtb", "file", "source", "size", "seed"])
        if new:
            w.writeheader()
        w.writerows(added)
    print(f"{a.dtb}: 전체 {len(files)}장 중 {len(added)}장 추출 → {out_dir.relative_to(DATASET)} "
          f"(저해상도 제외 {skipped}장)")


if __name__ == "__main__":
    main()
