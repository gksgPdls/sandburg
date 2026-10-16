#!/usr/bin/env python3
"""labels.csv 를 읽어 샘플마다 확인용 프리뷰를 만든다.

조작본 위에 마스크의 각 영역을 빨간 사각형으로 표시한다. 사람이 보는 용도이며
정답은 마스크다. preview 칸이 있으면 그 경로에, 없으면 조작본 옆에
{id}_preview.jpg 로 만든다.

사용: make_preview.py [labels.csv 또는 폴더 ...]   인자가 없으면 samples/ 아래 전부
"""

import sys
from pathlib import Path

import cv2

from labels import SAMPLES, find_csvs, path_of, preview_of, read_csv


def draw_preview(tampered_path, mask_path, out_path):
    """조작본에 마스크 영역을 빨간 사각형으로 그려 저장한다. 영역 개수를 돌려준다."""
    img = cv2.imread(str(tampered_path))
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        raise FileNotFoundError(f"읽을 수 없다: {tampered_path} / {mask_path}")
    count, _, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype("uint8"))
    for x, y, w, h, _ in stats[1:]:
        cv2.rectangle(img, (int(x) - 4, int(y) - 4),
                      (int(x + w) + 3, int(y + h) + 3), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return count - 1


def main():
    targets = []
    for a in map(Path, sys.argv[1:]):
        a = a if a.exists() else SAMPLES / a
        targets.append(a / "labels.csv" if a.is_dir() else a)
    for csv_path in targets or find_csvs():
        for row in read_csv(csv_path)[1]:
            tampered, mask = path_of(row, "tampered"), path_of(row, "mask")
            if tampered is None or mask is None:
                continue
            out = preview_of(row)
            try:
                count = draw_preview(tampered, mask, out)
            except FileNotFoundError as e:
                print(f"{row['id']}: {e}")
                continue
            print(f"{row['id']}: {out.name} (영역 {count}개)")


if __name__ == "__main__":
    main()
