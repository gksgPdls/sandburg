#!/usr/bin/env python3
"""다른 형식의 CSV(팀원 결과 등)를 우리 형식으로 바꿔 samples/{dest}/ 에 가져온다.

열 이름이 달라도 --map 으로 짝지으면 된다. 이미지는 samples/{dest}/ 로 복사하고,
CSV 의 파일 칸은 새 위치로 바꾼다. 짝짓지 않은 원래 열은 그대로 따라온다.
엑셀에서 저장한 한글 CSV(cp949)도 읽는다.

사용 예:
    import_labels.py ~/팀원결과/labels.csv --dest teammate_ai \\
        --map id=번호 original=원본 tampered=조작본 mask=정답 description=설명 \\
        --set target_axis=c2pa author=팀원 \\
        --id-prefix T_

    --map  우리열=그쪽열   (그쪽 열 값을 가져온다)
    --set  우리열=값       (모든 행에 같은 값. 그쪽에 없는 정보를 채울 때)
    --id-prefix            ID 앞에 붙인다. 우리 ID 와 겹치지 않게 (예: CM_001 → T_CM_001)
가져온 뒤 verify_sample.py 와 merge_labels.py 를 돌리면 합쳐진다.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

from labels import IMAGE_COLUMNS, REQUIRED, ROOT, SAMPLES


def read_any(path):
    """UTF-8(BOM 포함) 또는 cp949 로 CSV 를 읽는다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                return [c.strip() for c in r.fieldnames or []], [
                    {k.strip(): (v or "").strip() for k, v in row.items() if k} for row in r]
        except UnicodeDecodeError:
            continue
    sys.exit(f"인코딩을 알 수 없다: {path}")


def pairs(items, what):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"{what} 는 '우리열=값' 형식이다: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    p = argparse.ArgumentParser(description="다른 형식의 CSV 를 우리 형식으로 가져오기")
    p.add_argument("csv", help="가져올 CSV")
    p.add_argument("--dest", required=True, help="samples/ 아래에 만들 폴더 이름")
    p.add_argument("--map", nargs="*", help="우리열=그쪽열")
    p.add_argument("--set", nargs="*", help="우리열=값")
    p.add_argument("--id-prefix", default="", help="ID 앞에 붙일 문자 (겹침 방지)")
    a = p.parse_args()

    src_csv = Path(a.csv).expanduser().resolve()
    src_dir = src_csv.parent
    cols, rows = read_any(src_csv)
    mapping, consts = pairs(a.map, "--map"), pairs(a.set, "--set")
    missing = [c for c in mapping.values() if c not in cols]
    if missing:
        sys.exit(f"그쪽 CSV 에 없는 열: {missing}\n있는 열: {cols}")

    with open(ROOT / "labels_template.csv", encoding="utf-8-sig", newline="") as f:
        ours = next(csv.reader(f))
    used_src = set(mapping.values())
    extras = [c for c in cols if c not in used_src and c not in ours]
    out_cols = ours + extras

    dest = SAMPLES / a.dest
    dest.mkdir(parents=True, exist_ok=True)
    out_rows, problems = [], []
    for i, row in enumerate(rows, 1):
        new = {c: "" for c in out_cols}
        new.update({c: row.get(c, "") for c in extras})
        new.update(consts)
        for ours_col, theirs in mapping.items():
            new[ours_col] = row.get(theirs, "")
        if new.get("id"):
            new["id"] = a.id_prefix + new["id"]
        for c in IMAGE_COLUMNS:
            if not new.get(c):
                continue
            src = (src_dir / new[c]).resolve() if not Path(new[c]).is_absolute() else Path(new[c])
            if not src.exists():
                problems.append(f"{i}행 {c}: 파일이 없다 ({new[c]})")
                new[c] = ""
                continue
            name = f"{new.get('id') or i}_{c}{src.suffix.lower()}"
            shutil.copy2(src, dest / name)
            new[c] = name
        empty = [c for c in REQUIRED if not new.get(c)]
        if empty:
            problems.append(f"{i}행: 필수 칸이 비었다 {empty} (--map 이나 --set 으로 채운다)")
        out_rows.append(new)

    with open(dest / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)
    if not (dest / "README.md").exists():
        (dest / "README.md").write_text(f"# {a.dest}\n\n가져온 원본: `{src_csv.name}`\n\n"
                                        f"TODO: 조작 방식과 사용한 도구를 적는다.\n", encoding="utf-8")
    print(f"{len(out_rows)}행 → samples/{a.dest}/  (그대로 따라온 열: {extras or '없음'})")
    for pr in problems[:20]:
        print("  X", pr)


if __name__ == "__main__":
    main()
