#!/usr/bin/env python3
"""모든 labels.csv 에 원본 파일명 열(source_file, source_folder)을 채운다.

real_docs 에서 원본을 찾을 수 있게 하기 위해서다 (2026-10-03 회의 요청).
- source_file   real_docs 의 원본 파일명
- source_folder 날짜 폴더 (batch08 / batch09)

찾는 순서:
1. source 의 풀 ID 를 대응표에서 찾는다. 대응표는 _work/pool/manifest.csv,
   없으면 _work/pool_manifest_recovered.csv (시드 재계산으로 복원한 것)
2. 그래도 없으면 original 이미지와 바이트가 같은 원본을 real_docs 에서 찾는다 (해시 비교)

다시 실행해도 된다. 이미 채운 칸도 다시 계산한다.
"""

import csv
import hashlib
from pathlib import Path

from labels import ROOT, find_csvs, path_of, read_csv

RAW = ROOT.parent / "real_docs"
MANIFESTS = [ROOT / "_work" / "pool" / "manifest.csv", ROOT / "_work" / "pool_manifest_recovered.csv"]
NEW_COLS = ["source_file", "source_folder"]


def load_manifest():
    for m in MANIFESTS:
        if m.exists():
            with open(m, encoding="utf-8-sig", newline="") as f:
                return {r["pool_id"]: r["source"] for r in csv.DictReader(f)}, m.name
    return {}, None


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    by_id, name = load_manifest()
    print(f"대응표: {name or '없음'} ({len(by_id)}개)")
    by_hash = None
    filled = missing = 0
    for csv_path in find_csvs():
        cols, rows = read_csv(csv_path)
        for r in rows:
            src = by_id.get(r.get("source", ""))
            if not src:
                if by_hash is None:
                    by_hash = {md5(f): f.relative_to(RAW).as_posix() for f in RAW.rglob("*")
                               if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
                orig = path_of(r, "original")
                src = by_hash.get(md5(orig)) if orig and orig.exists() else None
            if src:
                folder, _, fname = src.rpartition("/")
                r["source_file"], r["source_folder"] = fname, folder[-6:]
                filled += 1
            else:
                r["source_file"], r["source_folder"] = "", ""
                missing += 1
        out_cols = list(cols)
        for i, c in enumerate(NEW_COLS):
            if c not in out_cols:
                at = out_cols.index("source") + 1 + i if "source" in out_cols else len(out_cols)
                out_cols.insert(at, c)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"원본 파일명 채움 {filled}개, 못 찾음 {missing}개")


if __name__ == "__main__":
    main()
