"""
판정 근거 이미지 — 샘플마다 1장: 원본과 위조본의 판정 결과(정탐/미탐/오탐/정상) + 탐지기가 실제로 비교한 패치와 노이즈 지도.

  위: 의심영역 주변 확대 (정상 원본 | 위조본), 빨간 박스 = 의심영역, 탐지법별 측정값과 판정
  아래: 원본·위조본 각각
        1줄  의심영역 | 노이즈 비교 패치들 (최대 5개, 이 중앙값과 비교. FFT·픽셀통계는 비교1만 사용)
        2줄  같은 순서의 노이즈 지도 (웨이블릿으로 분리한 노이즈, 밝을수록 노이즈 많음, 같은 배율)

사용: python make_evidence.py              # 전체 (이미 만든 건 건너뜀, 끊겨도 이어서)
      python make_evidence.py AIP_0004 ...  # 지정한 것만
      --fresh 전부 다시 / --max-seconds S
출력: outputs/reports/evidence/정탐 또는 미탐/{id}.jpg, 원본이 오탐이면 evidence/오탐/에도 복사
  (요약표.csv와 읽어보기.txt는 organize_evidence.py가 만듦)
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.restoration import denoise_wavelet

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.v1.core import judge, load_config, mask_to_bbox, measure  # noqa: E402
from detector.v1.methods import Context  # noqa: E402

CAT = ROOT / "outputs/reports/catalog_v1"
OUT = ROOT / "outputs/reports/evidence"
def existing(sid):
    return [p for p in (OUT / "정탐" / f"{sid}.jpg", OUT / "미탐" / f"{sid}.jpg") if p.exists()]
FONT = next((f for f in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"] if Path(f).exists()), None)
COLOR = {"정탐": (20, 90, 200), "미탐": (215, 110, 0), "오탐": (200, 20, 20), "정상": (30, 140, 60)}
CFG = None


def fnt(size):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def noise_map(p):
    x = p.astype(np.float32) / 255.0
    return np.abs(x - denoise_wavelet(x, channel_axis=None, rescale_sigma=True)) * 255.0


def to_img(arr, scale, vmax=None):
    a = arr.astype(np.float32)
    if vmax is not None:
        a = np.clip(a / vmax * 255.0, 0, 255)
    im = Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.NEAREST)


def context_crop(rgb, box, scale, margin=50):
    x0, y0, x1, y1 = box
    c = (max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, rgb.width), min(y1 + margin, rgb.height))
    cr = rgb.crop(c).resize((int((c[2] - c[0]) * scale), int((c[3] - c[1]) * scale)), Image.LANCZOS)
    ImageDraw.Draw(cr).rectangle([(x0 - c[0]) * scale, (y0 - c[1]) * scale, (x1 - c[0]) * scale, (y1 - c[1]) * scale],
                                 outline=(220, 30, 30), width=3)
    return cr


def analyse(path, mask):
    rgb = Image.open(path).convert("RGB")
    res = judge(measure(rgb, mask=mask, config=CFG), CFG)
    ctx = Context(rgb, mask, mask_to_bbox(mask), params=CFG["methods"])
    inside, outs = ctx.comparison("noise")
    return rgb, res, inside, outs


def make(row):
    sid = row["id"]
    mask = np.asarray(Image.open((CAT / row["mask"]).resolve()).convert("L"))
    box = mask_to_bbox(mask)
    A = {}
    for kind, col in (("orig", "original"), ("tamp", "tampered")):
        A[kind] = analyse((CAT / row[col]).resolve(), mask)
    lab = {"orig": "오탐" if A["orig"][1]["verdict"]["suspicious"] else "정상",
           "tamp": "정탐" if A["tamp"][1]["verdict"]["suspicious"] else "미탐"}

    bw, bh = box[2] - box[0], box[3] - box[1]
    cscale = max(1.0, min(5.0, 560 / (bw + 100)))
    ca, cb = context_crop(A["orig"][0], box, cscale), context_crop(A["tamp"][0], box, cscale)
    tscale = max(1.0, min(6.0, 70 / max(bh, 1), 150 / max(bw, 1)))
    tw, th = int(bw * tscale), int(bh * tscale)
    vmax = max(max(noise_map(A[k][2]).max(), *[noise_map(o).max() for o in A[k][3]]) for k in A) * 0.6 + 1e-6

    f_t, f_m, f_s = fnt(26), fnt(19), fnt(15)
    pad = 20
    top_h = 70 + max(ca.height, cb.height) + 150
    sec_h = 40 + 2 * (th + 26) + 20
    W = max(ca.width + cb.width + pad * 3, pad * 2 + 120 + 6 * (tw + 10))
    H = top_h + 2 * sec_h + 10
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    d.text((pad, 14), f"{sid}   {row['description'].split(' (')[0]}", fill="black", font=f_t)
    for j, (k, title, img) in enumerate((("orig", "정상 원본 (같은 위치)", ca), ("tamp", "위조본", cb))):
        x = pad + j * (ca.width + pad)
        d.text((x, 50), f"{title}  :  ", fill=(80, 80, 80), font=f_m)
        tx = x + d.textlength(f"{title}  :  ", font=f_m)
        d.text((tx, 50), lab[k], fill=COLOR[lab[k]], font=f_m)
        cv.paste(img, (x, 80))
        y = 80 + max(ca.height, cb.height) + 10
        res = A[k][1]
        for i, (mk, nm) in enumerate((("noise", "노이즈"), ("fft", "FFT"), ("pixel_std", "픽셀통계"))):
            m = res["methods"][mk]
            r = m.get("rarity")
            d.text((x, y + i * 26), f"{nm}: {m.get('score')}  ({'걸림' if m['flagged'] else '통과'}"
                                    + (f", 정상 중 {r*100:.1f}%" if r is not None else "") + ")",
                   fill=(180, 0, 0) if m["flagged"] else "black", font=f_m)
        v = res["verdict"]
        d.text((x, y + 3 * 26 + 4), f"종합: {'위조 의심' if v['suspicious'] else '정상 판정'} ({v['votes']}/{v['enabled_methods']})",
               fill=(180, 0, 0) if v["suspicious"] else "black", font=f_m)

    for s, (k, title) in enumerate((("orig", "정상 원본"), ("tamp", "위조본"))):
        _, res, inside, outs = A[k]
        y0 = top_h + s * sec_h
        d.line([(pad, y0 + 4), (W - pad, y0 + 4)], fill=(220, 220, 220), width=1)
        d.text((pad, y0 + 10), f"{title} — 노이즈 비교: 의심영역 {res['methods']['noise'].get('inside')} ÷ "
                               f"비교 패치 중앙값 {res['methods']['noise'].get('outside')} = {res['methods']['noise'].get('score')}"
                               f"   (FFT·픽셀통계는 비교1만 사용)", fill=(60, 60, 60), font=f_s)
        tiles = [("의심영역", inside)] + [(f"비교{i + 1}", o) for i, o in enumerate(outs)]
        for r_i, use_map in enumerate((False, True)):
            y = y0 + 40 + r_i * (th + 26)
            d.text((pad, y + th // 2 - 8), "노이즈 지도" if use_map else "패치", fill=(90, 90, 90), font=f_s)
            for t_i, (nm, p) in enumerate(tiles):
                x = pad + 120 + t_i * (tw + 10)
                cv.paste(to_img(noise_map(p), tscale, vmax) if use_map else to_img(p, tscale), (x, y))
                from detector.v1.features import wavelet_noise_std
                cap = f"{nm} {wavelet_noise_std(p):.2f}" if use_map else nm
                d.text((x, y + th + 3), cap, fill=(90, 90, 90), font=f_s)
    for d in ("정탐", "미탐", "오탐"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
        stale = OUT / d / f"{sid}.jpg"
        if stale.exists():  # 판정이 바뀌었을 수 있으니 예전 파일은 지우고 새로 저장
            stale.unlink()
    cv.save(OUT / lab["tamp"] / f"{sid}.jpg", quality=82)
    if lab["orig"] == "오탐":
        cv.save(OUT / "오탐" / f"{sid}.jpg", quality=82)
    return sid, lab["orig"], lab["tamp"]


def _init():
    global CFG
    CFG = load_config()


def _safe(row):
    try:
        return make(row)
    except Exception as e:  # noqa: BLE001
        return row["id"], "오류", str(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(CAT / "labels.csv", encoding="utf-8-sig")))
    if a.ids:
        rows = [r for r in rows if r["id"] in a.ids]
    elif not a.fresh:
        rows = [r for r in rows if not existing(r["id"])]
    t0 = time.time(); n = 0
    with Pool(4, initializer=_init) as pool:
        for sid, lo, lt in pool.imap_unordered(_safe, rows, chunksize=2):
            n += 1
            if lo == "오류":
                print("오류", sid, lt)
            if time.time() - t0 > a.max_seconds:
                pool.terminate(); break
    left = len([r for r in csv.DictReader(open(CAT / 'labels.csv', encoding='utf-8-sig')) if not existing(r["id"])])
    print(f"이번에 {n}장 생성, 남은 것 {left}장")
