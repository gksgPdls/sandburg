#!/usr/bin/env python3
"""문서의 빈 영역에 텍스트를 추가한 조작본을 만든다.

지우지 않고 여백에 써 넣는 수법(예: 진단서 병명 칸에 진단 추가)을 재현한다.
원본 글자에 맞추기 위해 다음을 조절한다.

- 크기: --fit 으로 원본에 있는 문구를 같은 폰트로 그렸을 때 폭이 맞는 크기를 찾는다
- 색: --ink 잉크 밝기(0-255). 촬영본은 글자가 순검정이 아니다
- 번짐: --blur 가우시안 시그마(px). 카메라 초점·압축으로 뭉개진 정도
- 기울기: --angle 도(반시계 +). 촬영 각도로 기운 행을 따라간다

합성은 흰 박스를 덮지 않고 잉크를 종이에 곱한다. 종이 질감과 뒷면 비침이
글자 사이로 그대로 남는다.

--mask 를 주면 추가한 글자의 외곽 사각형을 정답 마스크로 쓴다.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUPERSAMPLE = 4  # 크게 그린 뒤 줄여서 안티앨리어싱을 원본처럼 부드럽게 만든다


def load_font(path, index, size):
    return ImageFont.truetype(path, max(1, round(size * SUPERSAMPLE)), index=index)


def text_width(font_path, index, size, text):
    f = load_font(font_path, index, size)
    left, _, right, _ = f.getbbox(text)
    return (right - left) / SUPERSAMPLE


def fit_size(font_path, index, text, width):
    """text 를 그렸을 때 폭이 width 에 가장 가까운 글자 크기(px)."""
    sizes = np.arange(6, 60, 1 / SUPERSAMPLE)
    errors = [abs(text_width(font_path, index, s, text) - width) for s in sizes]
    return float(sizes[int(np.argmin(errors))])


def render_alpha(text, font_path, index, size, angle):
    """글자 잉크 농도(0-1) 배열과, 글자 좌상단 기준 오프셋을 돌려준다."""
    f = load_font(font_path, index, size)
    left, top, right, bottom = f.getbbox(text)
    pad = 4 * SUPERSAMPLE
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(canvas).text((pad - left, pad - top), text, font=f, fill=255)

    if angle:
        # 글자 좌상단을 축으로 돌려야 --xy 가 회전 후에도 첫 글자 위치를 가리킨다
        canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=True,
                               center=(pad, pad))

    small = canvas.resize(
        (max(1, canvas.width // SUPERSAMPLE), max(1, canvas.height // SUPERSAMPLE)),
        Image.LANCZOS,
    )
    alpha = np.asarray(small, np.float32) / 255.0

    # 좌상단 기준점이 small 배열에서 어디에 있는지 (회전으로 캔버스가 커진 만큼 보정)
    ys, xs = np.nonzero(alpha > 0.1)
    return alpha, (int(xs.min()), int(ys.min()))


def main():
    p = argparse.ArgumentParser(description="여백에 텍스트 추가 조작본 생성")
    p.add_argument("source", help="원본 이미지")
    p.add_argument("--text", required=True, help="추가할 문구")
    p.add_argument("--xy", nargs=2, type=int, required=True, metavar=("X", "Y"),
                   help="추가할 글자의 좌상단 좌표")
    p.add_argument("--font", default="/System/Library/Fonts/AppleSDGothicNeo.ttc")
    p.add_argument("--font-index", type=int, default=0,
                   help="TTC 안의 굵기 번호 (AppleSDGothicNeo: 0 Regular, 2 Medium, 8 Light)")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--size", type=float, help="글자 크기(px)")
    size.add_argument("--fit", nargs=2, metavar=("REF_TEXT", "WIDTH"),
                      help="원본의 REF_TEXT 가 WIDTH px 폭이 되는 크기를 쓴다")
    p.add_argument("--ink", type=float, default=0, help="잉크 밝기 0-255 (기본 0=검정)")
    p.add_argument("--blur", type=float, default=0, help="번짐 시그마 px (기본 0)")
    p.add_argument("--angle", type=float, default=0, help="기울기 도, 반시계 + (기본 0)")
    p.add_argument("-o", "--output", default="tampered.jpg", help="조작본 출력 경로")
    p.add_argument("-q", "--quality", type=int, default=95, help="JPEG 품질 (기본 95)")
    p.add_argument("--mask", metavar="PATH", help="정답 마스크(PNG) 경로")
    args = p.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        sys.exit(f"원본을 읽을 수 없다: {args.source}")
    H, W = img.shape[:2]

    if args.fit:
        ref_text, ref_width = args.fit[0], float(args.fit[1])
        font_size = fit_size(args.font, args.font_index, ref_text, ref_width)
        print(f"크기 맞춤: '{ref_text}' 폭 {ref_width:g}px → {font_size:g}px")
    else:
        font_size = args.size

    alpha, (ox, oy) = render_alpha(args.text, args.font, args.font_index,
                                   font_size, args.angle)
    if args.blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), args.blur)

    x0, y0 = args.xy[0] - ox, args.xy[1] - oy
    h, w = alpha.shape
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        sys.exit(f"글자가 이미지 밖으로 나간다: 좌상단 ({x0},{y0}), 크기 {w}x{h}")

    region = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    # 잉크를 종이에 곱한다. 종이 밝기 대비 잉크 비율만큼 어두워지므로 질감이 남는다
    paper = float(np.median(region))
    k = min(1.0, args.ink / paper) if paper > 0 else 0.0
    factor = 1.0 - alpha[..., None] * (1.0 - k)
    out = img.copy()
    out[y0:y0 + h, x0:x0 + w] = np.clip(region * factor, 0, 255).astype(np.uint8)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f"조작본 저장: {args.output}")
    print(f"  '{args.text}'  크기 {font_size:g}px  잉크 {args.ink:g}  "
          f"번짐 {args.blur:g}  기울기 {args.angle:g}도  종이 {paper:.0f}")

    if args.mask:
        ys, xs = np.nonzero(alpha > 0.05)
        mx, my = x0 + int(xs.min()), y0 + int(ys.min())
        mw, mh = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        mask = np.zeros((H, W), np.uint8)
        mask[my:my + mh, mx:mx + mw] = 255
        Path(args.mask).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.mask, mask)
        print(f"마스크 저장: {args.mask}  (글자 영역 {mw}x{mh} @ ({mx},{my}))")


if __name__ == "__main__":
    main()
