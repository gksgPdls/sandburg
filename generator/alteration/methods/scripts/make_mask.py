#!/usr/bin/env python3
"""원본과 조작본을 비교해 정답 마스크(PNG)를 생성한다.

조작 영역은 흰색(255), 나머지는 검은색(0)이다.

JPEG 재저장은 이미지 전역에 미세한 차이를 만들므로 임계값과 형태학 연산으로
압축 잡음을 걸러낸다. 임계값이 너무 낮으면 잡음이 마스크에 섞이고, 너무 높으면
연한 색 변경을 놓친다. --preview로 결과를 눈으로 확인한 뒤 확정한다.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep, regions = np.zeros_like(mask), []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        keep[labels == i] = 255
        regions.append(
            {
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": area,
            }
        )
    return keep, regions


def build_mask(original, tampered, threshold, min_area, close_px, mode, merge_px):
    # 채널별 차이의 최댓값. 밝기만 보면 색상만 바뀐 변경을 놓친다.
    diff = np.abs(original.astype(np.int16) - tampered.astype(np.int16))
    diff = diff.max(axis=2).astype(np.uint8)

    _, raw = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    if mode == "pixel":
        # 고립된 압축 잡음 제거 후 글자 획 사이의 구멍 메우기
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if close_px > 0:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8)
            )
        return _components(mask, min_area)

    # region 모드: 흩어진 글자 획을 하나의 조작 영역으로 묶는다.
    # 배경색이 같은 자리에 붙여넣으면 글자 픽셀만 남으므로, 그대로 두면
    # 정답이 획 조각 수십 개로 쪼개진다.
    # - 획 단위로 걸러내면 가는 획이 잘리므로, 묶은 뒤 묶음 단위로 잡음을 거른다
    # - 사각형은 팽창한 영역이 아니라 실제로 바뀐 픽셀의 경계로 잡는다
    merged = cv2.dilate(raw, np.ones((merge_px, merge_px), np.uint8))
    count, labels = cv2.connectedComponents(merged, connectivity=8)

    filled, blocks = np.zeros_like(raw), []
    for i in range(1, count):
        ys, xs = np.nonzero((labels == i) & (raw > 0))
        if len(xs) < min_area:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        filled[y0:y1 + 1, x0:x1 + 1] = 255
        blocks.append({"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                       "area": (x1 - x0 + 1) * (y1 - y0 + 1)})
    return filled, blocks


def main():
    p = argparse.ArgumentParser(description="원본/조작본 diff로 정답 마스크 생성")
    p.add_argument("original", help="원본 이미지")
    p.add_argument("tampered", help="조작본 이미지")
    p.add_argument("-o", "--output", default="mask.png", help="마스크 출력 경로 (PNG)")
    p.add_argument("-t", "--threshold", type=int, default=30,
                   help="픽셀 차이 임계값 0-255 (기본 30). 잡음이 섞이면 올린다")
    p.add_argument("--min-area", type=int, default=20,
                   help="이 픽셀 수보다 작은 영역은 버린다 (기본 20)")
    p.add_argument("--close", type=int, default=5,
                   help="글자 획 사이를 메우는 커널 크기 (기본 5, 0이면 끔)")
    p.add_argument("-m", "--mode", choices=["pixel", "region"], default="region",
                   help="pixel=바뀐 픽셀만, region=조작 영역을 사각형으로 채움 (기본 region)")
    p.add_argument("--merge-px", type=int, default=40,
                   help="region 모드에서 근접한 조각을 묶는 거리 (기본 40). "
                        "영역이 쪼개지면 올리고, 서로 다른 조작이 하나로 뭉치면 내린다")
    p.add_argument("--preview", metavar="PATH",
                   help="조작 영역을 빨갛게 덧칠한 확인용 이미지 경로")
    args = p.parse_args()

    original = cv2.imread(args.original)
    tampered = cv2.imread(args.tampered)

    if original is None:
        sys.exit(f"원본을 읽을 수 없다: {args.original}")
    if tampered is None:
        sys.exit(f"조작본을 읽을 수 없다: {args.tampered}")
    if original.shape != tampered.shape:
        sys.exit(
            f"해상도가 다르다. 원본 {original.shape[1]}x{original.shape[0]}, "
            f"조작본 {tampered.shape[1]}x{tampered.shape[0]}\n"
            "편집 도구에서 리사이즈가 일어났다. 원본 해상도를 유지해 다시 저장해야 한다."
        )

    mask, regions = build_mask(
        original, tampered, args.threshold, args.min_area,
        args.close, args.mode, args.merge_px,
    )

    if not regions:
        sys.exit(
            "변경 영역을 찾지 못했다.\n"
            f"임계값({args.threshold})을 낮추거나, 조작본이 실제로 저장됐는지 확인한다."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, mask)

    total = mask.size
    changed = int((mask > 0).sum())
    print(f"마스크 저장: {args.output}  (mode={args.mode})")
    print(f"해상도: {mask.shape[1]} x {mask.shape[0]}")
    print(f"조작 픽셀: {changed:,} / {total:,} ({changed / total * 100:.3f}%)")
    print(f"영역 개수: {len(regions)}")
    for i, r in enumerate(regions, 1):
        print(f"  [{i}] x={r['x']} y={r['y']} w={r['w']} h={r['h']} area={r['area']:,}")

    if args.preview:
        overlay = tampered.copy()
        overlay[mask > 0] = (0, 0, 255)
        preview = cv2.addWeighted(tampered, 0.6, overlay, 0.4, 0)
        for r in regions:
            cv2.rectangle(
                preview, (r["x"], r["y"]),
                (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 2
            )
        cv2.imwrite(args.preview, preview)
        print(f"확인용 이미지: {args.preview}")

    if changed / total > 0.3:
        print("\n경고: 조작 영역이 30%를 넘는다. 재저장으로 전체 픽셀이 바뀌었을 수 있다.")
        print("조작본을 원본과 같은 포맷·품질로 저장했는지 확인한다.")


if __name__ == "__main__":
    main()
