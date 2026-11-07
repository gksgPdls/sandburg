"""
여러 변조 방식 결과를 한 곳에 합친다 (2026-10-30).

  python generator/alteration/forge/4_compare/merge_runs.py --out 통합_8장 --runs 한칸_8:논리끔 이전논리_8:이전논리맞춤 VLM_8:VLM

- outputs/고도화/<out>/비교/<원본>.jpg : 고친 칸마다 한 줄, [원본 | 방식1 | 방식2 | ...] (방식이 안 고친 칸도 그대로 보여줌)
- outputs/고도화/<out>/통합점수.csv  : 장마다 방식별 고친 칸 수, 변경 내용, 배경결·노이즈(원본 대비), 탐지기(고친 칸마다 따로)
- 화면에 방식별 요약
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # generator/alteration/forge
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
from detector.pixel import detect  # noqa: E402
import harmonize as HZ  # noqa: E402

OUT = ROOT / "outputs/고도화"


def _font(sz):
    for p in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="폴더:이름 ...")
    a = ap.parse_args()
    runs = [(s.split(":", 1) + [s])[:2] for s in a.runs]
    dst = OUT / a.out
    (dst / "비교").mkdir(parents=True, exist_ok=True)
    data = {}   # base -> name -> (img, cells, 변경)
    for folder, name in runs:
        d = OUT / folder / "AI"
        log = {r["id"]: r for r in csv.DictReader(open(d / "생성기록.csv", encoding="utf-8-sig"))}
        for f in sorted(d.glob("*_mask.png")):
            sid = f.name[:-9]
            p = next((q for q in (d / f"{sid}.jpg", d / f"{sid}.png") if q.exists()), None)
            if p is None:
                continue
            base = sid.split("_", 1)[-1]
            data.setdefault(base, {})[name] = (p, HZ.cell_boxes(np.asarray(Image.open(f).convert("L"))),
                                               log.get(sid, {}).get("변경", ""))
    names = [n for _, n in runs]
    rows = []
    font = _font(28)
    for base, per in sorted(data.items()):
        orig_p = next(q for q in (ROOT / "dataset/original/bankbook").glob(base + ".*") if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
        orig = Image.open(orig_p).convert("RGB")
        og = np.asarray(orig.convert("L"))
        # 모든 방식의 고친 칸을 모아 줄로 (겹치는 칸은 하나로)
        allc = []
        for n in names:
            for b in per.get(n, (None, [], ""))[1]:
                if not any(abs(b[0] - c[0]) < 10 and abs(b[1] - c[1]) < 10 for c in allc):
                    allc.append(b)
        allc.sort(key=lambda b: (b[1], b[0]))
        ims = {n: Image.open(per[n][0]).convert("RGB") for n in names if n in per}
        S, padx, pady = 3, 40, 10
        lines = []
        for b in allc:
            x, y, w, h = b
            box = (max(x - padx, 0), max(y - pady, 0), min(x + w + padx, orig.width), min(y + h + pady, orig.height))
            tiles = [orig.crop(box)] + [ims[n].crop(box) if n in ims else Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white") for n in names]
            tw, th = tiles[0].width * S, tiles[0].height * S
            line = Image.new("RGB", (len(tiles) * (tw + 6), th), (255, 255, 255))
            for i, t in enumerate(tiles):
                line.paste(t.resize((tw, th), Image.NEAREST), (i * (tw + 6), 0))
            lines.append(line)
        if lines:
            W = max(l.width for l in lines)
            head = Image.new("RGB", (W, 40), "white")
            dr = ImageDraw.Draw(head)
            tw = lines[0].width // (len(names) + 1)
            for i, t in enumerate(["원본"] + names):
                dr.text((i * tw + 6, 6), t, fill="black", font=font)
            sheet = Image.new("RGB", (W, 40 + sum(l.height + 6 for l in lines)), (210, 210, 210))
            sheet.paste(head, (0, 0))
            yy = 40
            for l in lines:
                sheet.paste(l, (0, yy)); yy += l.height + 6
            sheet.save(dst / "비교" / f"{base}.jpg", quality=92)
        for n in names:
            if n not in per:
                continue
            p, cells, ch = per[n]
            im = ims[n]
            g = np.asarray(im.convert("L"))
            bad, nz = [], []
            for k, b in enumerate(cells):
                r = detect(im, bbox=b, padding=0)
                r0 = detect(orig, bbox=b, padding=0)
                s1, s0 = r["methods"]["noise"]["score"], r0["methods"]["noise"]["score"]
                if s1 is not None and s0:
                    nz.append(s1 / s0)
                if r["verdict"]["suspicious"]:
                    bad.append(f"{k + 1}칸(" + ",".join(m["name"].split("_")[-1] for m in r["methods"].values() if m.get("flagged")) + ")")
            rows.append({"원본": base, "방식": n, "고친칸": len(cells), "변경": ch,
                         "노이즈_원본대비_최저": round(min(nz), 3) if nz else "",
                         "탐지기_의심": "Y" if bad else "", "의심칸": " ".join(bad)})
        print(base, " / ".join(f"{r['방식']} {r['고친칸']}칸 {'의심' if r['탐지기_의심'] else '통과'}" for r in rows if r["원본"] == base), flush=True)
    with open(dst / "통합점수.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n[고친 칸 위치를 알려줬을 때, 칸마다 따로 탐지]")
    print(f"{'방식':<14}{'장수':>4}{'평균 고친칸':>10}{'의심 장':>8}{'의심 칸':>8}{'노이즈 최저 중앙값':>18}")
    for n in names:
        s = [r for r in rows if r["방식"] == n]
        if not s:
            continue
        cells = sum(r["고친칸"] for r in s)
        badc = sum(len(r["의심칸"].split()) for r in s if r["의심칸"])
        nzs = [r["노이즈_원본대비_최저"] for r in s if r["노이즈_원본대비_최저"] != ""]
        print(f"{n:<14}{len(s):>4}{cells / len(s):>10.1f}{sum(bool(r['탐지기_의심']) for r in s):>8}{f'{badc}/{cells}':>8}"
              f"{np.median(nzs) if nzs else 0:>18.3f}")
    print(f"\n저장: {dst}")


if __name__ == "__main__":
    main()
