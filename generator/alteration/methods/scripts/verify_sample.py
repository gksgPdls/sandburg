#!/usr/bin/env python3
"""labels.csv 와 이미지가 서로 맞는지 검사한다.

막는 것(X)은 평가에 쓸 수 없게 되는 문제뿐이다.
- 필수 칸(id, target_axis, original, description)이 비었다
- ID가 겹친다
- 적힌 파일이 없다
- 원본·조작본·마스크 해상도가 다르다
- 마스크가 0/255 이진이 아니거나 비었거나 전면을 덮는다

나머지는 참고(-)로만 알려준다. 마지막에 외부 AI 투입 건수(상한 20)를 센다.

사용: verify_sample.py [labels.csv 또는 폴더 ...]   인자가 없으면 samples/ 아래 전부
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from labels import AXIS_NAME, REQUIRED, find_csvs, path_of, read_csv, SAMPLES

EXTERNAL_AI_LIMIT = 20


def check_row(row, problems, notes):
    rid = row.get("id") or "(id 없음)"
    empty = [c for c in REQUIRED if not row.get(c)]
    if empty:
        problems.append(f"{rid}: 필수 칸이 비어 있다 {empty}")
    if row.get("target_axis") and row["target_axis"] not in AXIS_NAME:
        notes.append(f"{rid}: target_axis '{row['target_axis']}' 는 목록에 없는 값이다 ({', '.join(AXIS_NAME)})")

    imgs = {}
    for col in ("original", "tampered", "mask"):
        p = path_of(row, col)
        if p is None:
            continue
        if not p.exists():
            problems.append(f"{rid}: {col} 파일이 없다 ({p.name})")
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            problems.append(f"{rid}: {p.name} 을 읽을 수 없다")
        else:
            imgs[col] = img

    if row.get("target_axis") not in ("none", "metadata", "c2pa"):
        if not row.get("tampered"):
            notes.append(f"{rid}: tampered 가 비어 있다")
        if not row.get("mask"):
            notes.append(f"{rid}: mask 가 비어 있다")

    shapes = {k: v.shape[:2] for k, v in imgs.items()}
    if len(set(shapes.values())) > 1:
        detail = ", ".join(f"{k}={s[1]}x{s[0]}" for k, s in shapes.items())
        problems.append(f"{rid}: 해상도가 서로 다르다 ({detail})")

    if "mask" in imgs:
        mask = imgs["mask"]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        extra = [int(v) for v in np.unique(mask) if v not in (0, 255)]
        if extra:
            hint = " (JPG로 저장하면 생긴다. PNG로 저장한다)" \
                if path_of(row, "mask").suffix.lower() in (".jpg", ".jpeg") else ""
            problems.append(f"{rid}: 마스크가 0/255 이진이 아니다{hint}")
        ratio = float((mask > 127).sum()) / mask.size
        if ratio == 0:
            problems.append(f"{rid}: 마스크가 비어 있다")
        elif ratio > 0.5:
            problems.append(f"{rid}: 마스크가 이미지의 {ratio:.0%}를 덮는다")


def main():
    targets = []
    for a in map(Path, sys.argv[1:]):
        a = a if a.exists() else SAMPLES / a
        targets.append(a / "labels.csv" if a.is_dir() else a)
    targets = targets or find_csvs()
    if not targets:
        sys.exit("labels.csv 가 없다")

    seen, total, external, failed = {}, 0, 0, 0
    for csv_path in targets:
        print(f"\n[{csv_path.parent.relative_to(SAMPLES) if SAMPLES in csv_path.parents else csv_path}]")
        if not csv_path.exists():
            print("  X labels.csv 가 없다")
            failed += 1
            continue
        columns, rows = read_csv(csv_path)
        problems, notes = [], []
        missing = [c for c in REQUIRED if c not in columns]
        if missing:
            problems.append(f"필수 열이 없다 {missing}")
        for row in rows:
            rid = row.get("id", "")
            if rid in seen:
                problems.append(f"{rid}: ID가 {seen[rid]} 와 겹친다")
            seen[rid] = csv_path.parent.name
            if not missing:
                check_row(row, problems, notes)
        if not (csv_path.parent / "README.md").exists():
            notes.append("README.md 가 없다 (조작 방식을 적어 두면 좋다)")
        total += len(rows)
        external += sum(r.get("external_ai", "").upper() == "Y" for r in rows)
        for n in notes:
            print(f"  - {n}")
        for p in problems:
            print(f"  X {p}")
        if problems:
            failed += 1
        else:
            print(f"  OK 샘플 {len(rows)}개")

    print(f"\nCSV {len(targets)}개 중 {len(targets) - failed}개 통과, 샘플 {total}개")
    print(f"외부 AI 투입 {external}건 / 상한 {EXTERNAL_AI_LIMIT}건 (external_ai=Y 기준)")
    if external > EXTERNAL_AI_LIMIT:
        print("  X 상한을 넘었다")
        failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
