"""labels.csv 를 읽는 공용 함수. 다른 스크립트가 가져다 쓴다.

규칙은 느슨하다.
- 필수 열은 REQUIRED 뿐이다. 템플릿에 없는 열을 더 붙여도 된다
- 파일 칸에는 CSV 파일 위치 기준 상대 경로를 적는다. 이름 규칙은 없다
- labels.csv 는 samples/ 아래 어느 폴더에 있어도 된다
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

REQUIRED = ["id", "target_axis", "original", "description"]
IMAGE_COLUMNS = ["original", "tampered", "mask", "preview"]

AXIS_NAME = {
    "copy_move": "copy-move", "glyph": "글리프", "content_logic": "콘텐츠 로직",
    "metadata": "메타데이터", "c2pa": "C2PA", "pixel": "픽셀", "none": "정상",
}


def find_csvs():
    return sorted(SAMPLES.glob("**/labels.csv"))


def read_csv(csv_path):
    """(열 목록, 행 목록). 행마다 _dir(CSV 폴더)과 _csv 를 붙인다."""
    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            row["_dir"], row["_csv"] = csv_path.parent, csv_path
            rows.append(row)
        return [c.strip() for c in reader.fieldnames or []], rows


def load_all(csv_paths=None):
    rows = []
    for p in csv_paths or find_csvs():
        rows += read_csv(p)[1]
    return rows


def path_of(row, column):
    """파일 칸의 실제 경로. 비어 있으면 None."""
    name = row.get(column, "")
    return (row["_dir"] / name) if name else None


def preview_of(row):
    """preview 칸이 있으면 그것, 없으면 조작본 옆의 {id}_preview.jpg."""
    if row.get("preview"):
        return path_of(row, "preview")
    tampered = path_of(row, "tampered")
    if tampered is None:
        return None
    return tampered.parent / f"{row['id']}_preview.jpg"
