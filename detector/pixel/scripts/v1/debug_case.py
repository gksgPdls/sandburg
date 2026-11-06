"""
디버깅 이미지 — 탐지기가 '무엇과 무엇을 비교했는지'를 보여준다.
행: 정상 원본 / 위조본,  열: 의심영역 | 비교 패치(탐지기가 고른 것) | 의심영역 노이즈 | 비교 패치 노이즈
노이즈 지도는 웨이블릿으로 분리한 노이즈 성분을 같은 배율로 밝게 표시 (밝을수록 노이즈 많음).
사용: python debug_case.py AIP_0004 AIP_0046 ...  → outputs/reports/evidence/{id}_debug.png
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.restoration import denoise_wavelet

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.v1 import features as F  # noqa: E402
from detector.v1.core import mask_to_bbox  # noqa: E402

CAT = ROOT / "outputs/reports/catalog_v1"
OUT = ROOT / "outputs/reports/evidence"
FONT = next((f for f in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"] if Path(f).exists()), None)


def noise_map(p):
    x = p.astype(np.float32) / 255.0
    return np.abs(x - denoise_wavelet(x, channel_axis=None, rescale_sigma=True)) * 255.0


def tile(arr, scale, vmax=None):
    a = arr.astype(np.float32)
    if vmax is not None:
        a = np.clip(a / vmax * 255.0, 0, 255)
    im = Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def make(sid):
    row = next(r for r in csv.DictReader(open(CAT / "labels.csv", encoding="utf-8-sig")) if r["id"] == sid)
    mask = np.asarray(Image.open((CAT / row["mask"]).resolve()).convert("L"))
    box = mask_to_bbox(mask)
    rows, stats = [], []
    for label, col in (("정상 원본", "original"), ("위조본", "tampered")):
        gray = np.asarray(Image.open((CAT / row[col]).resolve()).convert("L"))
        ins, out = F.sample_outside_patch(gray, box)
        ni, no = noise_map(ins), noise_map(out)
        rows.append((label, ins, out, ni, no))
        stats.append((F.wavelet_noise_std(ins), F.wavelet_noise_std(out), float(ins.mean()), float(out.mean())))
    vmax = max(max(r[3].max(), r[4].max()) for r in rows) * 0.6 + 1e-6
    scale = max(3, min(8, 260 // max(rows[0][1].shape[1], 1)))
    f = ImageFont.truetype(FONT, 20) if FONT else ImageFont.load_default()
    cols = ["의심 영역", "비교 패치 (탐지기 선택)", "의심 영역 노이즈", "비교 패치 노이즈"]
    tw, th = rows[0][1].shape[1] * scale, rows[0][1].shape[0] * scale
    W, H = 150 + 4 * (tw + 16), 80 + 2 * (th + 70)
    c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
    d.text((16, 12), f"{sid}  {row['description'].split(' (')[0]}", fill="black", font=f)
    for j, t in enumerate(cols):
        d.text((150 + j * (tw + 16), 46), t, fill=(90, 90, 90), font=f)
    for i, (label, ins, out, ni, no) in enumerate(rows):
        y = 80 + i * (th + 70)
        d.text((16, y + th // 2 - 10), label, fill="black", font=f)
        for j, im in enumerate([tile(ins, scale), tile(out, scale), tile(ni, scale, vmax), tile(no, scale, vmax)]):
            c.paste(im, (150 + j * (tw + 16), y))
        si, so, mi, mo = stats[i]
        d.text((150, y + th + 8), f"노이즈 {si:.2f} / {so:.2f} = 비율 {si / so if so else float('inf'):.3f}    "
                                  f"평균밝기 {mi:.0f} / {mo:.0f}", fill=(0, 0, 0), font=f)
    OUT.mkdir(parents=True, exist_ok=True)
    c.save(OUT / f"{sid}_debug.png")
    print(sid, [f"{s[0]:.2f}/{s[1]:.2f}={s[0]/s[1]:.3f}" for s in stats])


if __name__ == "__main__":
    for s in sys.argv[1:]:
        make(s)
