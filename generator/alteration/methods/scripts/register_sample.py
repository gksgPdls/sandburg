#!/usr/bin/env python3
"""(선택) 샘플 하나를 samples/{method}/ 에 복사하고 labels.csv 에 한 줄 추가한다.

꼭 쓸 필요는 없다. 직접 파일을 두고 CSV를 써도 된다. 파일 정리가 귀찮을 때 쓴다.
- 파일을 {id}_original.jpg 식으로 복사하고 프리뷰를 만든다
- --id 를 안 주면 축별 접두어(CM, GL …)에 다음 번호를 붙인다
- 폴더나 labels.csv 가 없으면 만든다

명령줄:
    register_sample.py --method copymove --axis copy_move \\
        --original _work/a.jpg --tampered _work/a_t.jpg --mask _work/a_m.png \\
        --description "무엇을 어떻게 바꿨는지" --detection "어떻게 잡히는지"

자동화 코드에서:
    sys.path.insert(0, "dataset/scripts")
    from register_sample import register
    sid = register(method="copymove", axis="copy_move", original=..., tampered=...,
                   mask=..., description="...")
"""

import argparse
import csv
import datetime
import shutil
import sys
from pathlib import Path

from labels import ROOT, SAMPLES, find_csvs, read_csv
from make_preview import draw_preview

PREFIX = {"copy_move": "CM", "glyph": "GL", "content_logic": "CL", "metadata": "MD",
          "c2pa": "C2", "pixel": "PX", "none": "NM"}


def next_id(prefix):
    nums = [int(r["id"].split("_", 1)[1]) for p in find_csvs() for r in read_csv(p)[1]
            if r.get("id", "").startswith(prefix + "_") and r["id"].split("_", 1)[1].isdigit()]
    return f"{prefix}_{max(nums, default=0) + 1:03d}"


def register(*, method, axis, original, description, tampered=None, mask=None,
             sample_id=None, **fields):
    """샘플을 등록하고 ID를 돌려준다. fields 에는 doc_type, detection, difficulty,
    external_ai, author, notes 등 다른 열을 자유롭게 넘긴다."""
    folder = SAMPLES / method
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "labels.csv"
    if csv_path.exists():
        columns = read_csv(csv_path)[0]
    else:
        with open(ROOT / "labels_template.csv", encoding="utf-8-sig", newline="") as f:
            columns = next(csv.reader(f))
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(columns)

    sid = sample_id or next_id(PREFIX.get(axis, "XX"))

    def place(src, kind):
        if not src:
            return ""
        name = f"{sid}_{kind}{Path(src).suffix.lower()}"
        shutil.copy2(src, folder / name)
        return name

    row = {"id": sid, "method": method, "target_axis": axis,
           "original": place(original, "original"), "tampered": place(tampered, "tampered"),
           "mask": place(mask, "mask"), "description": description,
           "created": datetime.date.today().isoformat(), **fields}
    if row["tampered"] and row["mask"]:
        draw_preview(folder / row["tampered"], folder / row["mask"], folder / f"{sid}_preview.jpg")

    new_cols = [c for c in row if c not in columns]
    if new_cols:
        raise ValueError(f"labels.csv 에 없는 열이다: {new_cols}")
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writerow(row)
    return sid


def main():
    p = argparse.ArgumentParser(description="샘플 하나를 복사하고 labels.csv 에 추가")
    p.add_argument("--method", required=True, help="폴더 이름 = 조작 방법 (예: copymove)")
    p.add_argument("--axis", required=True, help="주 탐지 축 (copy_move, glyph, content_logic, "
                                                 "metadata, c2pa, pixel, none)")
    p.add_argument("--original", required=True)
    p.add_argument("--tampered")
    p.add_argument("--mask")
    p.add_argument("--description", required=True, help="무엇을 어떻게 바꿨는지")
    p.add_argument("--id", dest="sample_id", help="생략하면 자동 (예: CM_003)")
    p.add_argument("--detection", default="")
    p.add_argument("--doc-type", default="")
    p.add_argument("--difficulty", default="")
    p.add_argument("--external-ai", default="N")
    p.add_argument("--author", default="")
    p.add_argument("--notes", default="")
    a = p.parse_args()
    try:
        sid = register(method=a.method, axis=a.axis, original=a.original, tampered=a.tampered,
                       mask=a.mask, description=a.description, sample_id=a.sample_id,
                       detection=a.detection, doc_type=a.doc_type, difficulty=a.difficulty,
                       external_ai=a.external_ai, author=a.author, notes=a.notes)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"등록 실패: {e}")
    print(f"등록: {sid} → samples/{a.method}/")


if __name__ == "__main__":
    main()
