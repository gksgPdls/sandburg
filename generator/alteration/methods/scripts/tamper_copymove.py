#!/usr/bin/env python3
"""카피무브 조작본을 만든다.

문서 내 한 영역을 복사해 같은 문서의 다른 위치에 붙여넣는다.
--pick 을 주면 창이 떠서 마우스로 복사할 영역과 붙일 위치를 고른다.

출력: tampered.jpg (조작본). --mask 를 주면 붙여넣은 사각형을 그대로 정답
마스크로 쓴다. 좌표를 알고 있으므로 make_mask.py 의 diff 추정보다 정확하다.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def pick_regions(img):
    scale = min(1.0, 1100 / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()

    print("\n[1/2] 복사할 영역을 드래그하고 ENTER. (취소는 C)")
    x, y, w, h = cv2.selectROI("copy from", view, showCrosshair=False)
    cv2.destroyWindow("copy from")
    if w == 0 or h == 0:
        sys.exit("영역이 선택되지 않았다")

    print("[2/2] 붙여넣을 위치를 드래그하고 ENTER. 크기는 무시하고 좌상단만 쓴다.")
    px, py, pw, ph = cv2.selectROI("paste to", view, showCrosshair=False)
    cv2.destroyWindow("paste to")
    if pw == 0 or ph == 0:
        sys.exit("위치가 선택되지 않았다")

    s = 1 / scale
    return (
        [int(x * s), int(y * s), int(w * s), int(h * s)],
        [int(px * s), int(py * s)],
    )


def main():
    p = argparse.ArgumentParser(description="카피무브 조작본 생성")
    p.add_argument("source", help="원본 이미지")
    p.add_argument("-o", "--output", default="tampered.jpg", help="조작본 출력 경로")
    p.add_argument("--pick", action="store_true", help="마우스로 영역 선택")
    p.add_argument("--src-box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="복사할 영역")
    p.add_argument("--dst", nargs=2, type=int, metavar=("X", "Y"),
                   help="붙여넣을 좌상단 좌표")
    p.add_argument("-q", "--quality", type=int, default=95,
                   help="JPEG 품질 (기본 95). 원본과 맞추면 잡음이 준다")
    p.add_argument("--mask", metavar="PATH",
                   help="붙여넣은 영역을 흰색(255)으로 칠한 정답 마스크(PNG) 경로")
    p.add_argument("--keep-original", metavar="PATH",
                   help="원본을 이 경로로 복사해 둔다")
    args = p.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        sys.exit(f"원본을 읽을 수 없다: {args.source}")
    H, W = img.shape[:2]
    print(f"원본: {args.source}  ({W} x {H})")

    if args.pick:
        src_box, dst = pick_regions(img)
        print(f"선택됨: --src-box {' '.join(map(str, src_box))} --dst {dst[0]} {dst[1]}")
    elif args.src_box and args.dst:
        src_box, dst = args.src_box, args.dst
    else:
        sys.exit("--pick 을 쓰거나 --src-box 와 --dst 를 함께 지정한다")

    x, y, w, h = src_box
    dx, dy = dst
    if x < 0 or y < 0 or x + w > W or y + h > H:
        sys.exit(f"복사 영역이 이미지 밖이다: {src_box}")
    if dx < 0 or dy < 0 or dx + w > W or dy + h > H:
        sys.exit(f"붙여넣을 위치가 이미지 밖이다: {dst} (블록 {w}x{h})")

    out = img.copy()
    out[dy:dy + h, dx:dx + w] = img[y:y + h, x:x + w]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f"조작본 저장: {args.output}")
    print(f"  복사 {w}x{h} @ ({x},{y})  →  ({dx},{dy})")

    if args.mask:
        mask = np.zeros((H, W), np.uint8)
        mask[dy:dy + h, dx:dx + w] = 255
        Path(args.mask).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.mask, mask)
        print(f"마스크 저장: {args.mask}  (붙여넣은 영역 {w}x{h} @ ({dx},{dy}))")

    if args.keep_original:
        Path(args.keep_original).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, args.keep_original)
        print(f"원본 사본: {args.keep_original}")


if __name__ == "__main__":
    main()
