#!/usr/bin/env python3
"""samples/ 아래 모든 labels.csv 를 하나로 합쳐 dataset/labels_all.csv 를 만든다.

CSV마다 열이 달라도 된다. 모든 열을 모아서 쓰고, 없는 칸은 비운다.
파일 칸은 samples/ 기준 상대 경로로 바꾼다 (예: copymove/CM_001_original.jpg).
"""

import csv

from labels import IMAGE_COLUMNS, ROOT, SAMPLES, find_csvs, read_csv


def main():
    with open(ROOT / "labels_template.csv", encoding="utf-8-sig", newline="") as f:
        columns = next(csv.reader(f))

    rows = []
    for csv_path in find_csvs():
        cols, part = read_csv(csv_path)
        columns += [c for c in cols if c not in columns]
        for row in part:
            for c in IMAGE_COLUMNS:
                if row.get(c):
                    row[c] = (row["_dir"] / row[c]).relative_to(SAMPLES).as_posix()
            rows.append(row)
    rows.sort(key=lambda r: r.get("id", ""))

    out = ROOT / "labels_all.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{out.name}: 샘플 {len(rows)}개, 열 {len(columns)}개")


if __name__ == "__main__":
    main()
