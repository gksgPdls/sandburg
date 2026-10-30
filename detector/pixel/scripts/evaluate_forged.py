"""
우리 변조 결과가 탐지기에 얼마나 걸리는지 (2026-10-30).

  python scripts/detect/evaluate_forged.py --runs VLM_8:VLM 이전논리_8:이전논리맞춤 한칸_8:논리끔

각 결과 폴더(outputs/고도화/<폴더>/AI)의 이미지마다 두 가지로 잰다.
  1) 모든 칸 스캔 (실사용과 같음): 좌표 없이 RapidOCR로 찾은 모든 글자 칸에 탐지기를 돌리고, 가장 이상한 칸의 희귀도가
     문서 기준(locate_evaluation_rapid.json, 정상 문서 오탐 10% 지점)보다 작으면 '위조 의심'.
     같은 원본도 똑같이 재서, 원본은 정상인데 결과만 의심이면 '들킴'.
     '위치까지 맞힘' = 가장 이상한 칸이 실제로 고친 칸.
  2) 위치를 알려줬을 때: 고친 칸마다 따로 탐지기 (한 칸이라도 걸리면 의심).
결과: outputs/고도화/탐지평가_<날짜>.csv 와 화면 표
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/generate"))
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
from detector import detect  # noqa: E402
from detector.core import load_config  # noqa: E402
from detector.locate import locate  # noqa: E402
import harmonize as HZ  # noqa: E402

OUT = ROOT / "outputs/고도화"


def doc_threshold(only="all"):
    p = ROOT / "outputs/reports" / ("locate_evaluation_rapid.json" if only == "all" else f"locate_evaluation_rapid_{only}.json")
    if not p.exists():
        sys.exit(f"먼저 탐지기_모든칸평가.command 를 돌려야 함 ({p.name} 없음)")
    return json.load(open(p, encoding="utf-8"))["문서_기준_희귀도"]


def overlaps(b, boxes, tol=4):
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    return any(bx - tol <= cx <= bx + bw + tol and by - tol <= cy <= by + bh + tol for bx, by, bw, bh in boxes)


def scan(img, thr, only):
    r = locate(img, ocr="rapid", only=only)
    c = [x for x in r["candidates"] if x["rarity"] is not None]
    top = c[0] if c else None
    return (bool(top and top["rarity"] < thr), top)


_orig_cache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="폴더:이름 ...")
    ap.add_argument("--only", default="all", choices=["all", "digits", "money"])
    a = ap.parse_args()
    thr = doc_threshold(a.only)
    rows = []
    for spec in a.runs:
        folder, name = (spec.split(":", 1) + [spec])[:2]
        d = OUT / folder / "AI"
        log = {r["id"]: r for r in csv.DictReader(open(d / "생성기록.csv", encoding="utf-8-sig"))}
        lst = None
        for f in sorted(d.glob("*_mask.png")):
            sid = f.name[:-9]
            img_p = next((p for p in (d / f"{sid}.jpg", d / f"{sid}.png") if p.exists()), None)
            if img_p is None:
                continue
            base = sid.split("_", 1)[-1]   # vlm_auto00017 → auto00017
            orig_p = next(p for p in (ROOT / "dataset/original/bankbook").glob(base + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            mask = np.asarray(Image.open(f).convert("L"))
            cells = HZ.cell_boxes(mask)
            im = Image.open(img_p).convert("RGB")
            if base not in _orig_cache:
                _orig_cache[base] = scan(Image.open(orig_p).convert("RGB"), thr, a.only)
            o_sus, o_top = _orig_cache[base]
            f_sus, f_top = scan(im, thr, a.only)
            hit = bool(f_top and overlaps(f_top["bbox"], cells))
            per = []
            for b in cells:
                rr = detect(im, bbox=b, padding=0)
                if rr["verdict"]["suspicious"]:
                    per.append("/".join(m["name"].split("_")[-1] for m in rr["methods"].values() if m.get("flagged")))
            rows.append({"방식": name, "id": sid, "고친칸": len(cells),
                         "변경": log.get(sid, {}).get("변경", "")[:120],
                         "원본_의심": "Y" if o_sus else "",
                         "결과_의심": "Y" if f_sus else "",
                         "들킴": "Y" if f_sus and not o_sus else "",
                         "위치까지": "Y" if f_sus and hit else "",
                         "가장이상한칸": f_top["text"] if f_top else "",
                         "희귀도": round(f_top["rarity"], 4) if f_top else "",
                         "위치알려줌_의심칸": len(per), "위치알려줌_탐지법": " ".join(per)})
            print(name, sid, "결과", rows[-1]["결과_의심"] or "-", "원본", rows[-1]["원본_의심"] or "-",
                  "위치", rows[-1]["위치까지"] or "-", "칸별", f"{len(per)}/{len(cells)}", flush=True)
    p = OUT / f"탐지평가_{date.today():%Y%m%d}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[모든 칸 스캔, 문서 기준 희귀도 < {thr:.4f}]  /  [위치 알려줌: 고친 칸마다]")
    print(f"{'방식':<12}{'장수':>4}{'결과 의심':>10}{'원본 의심':>10}{'들킴':>6}{'위치까지':>8}{'칸별 의심 장':>12}")
    for name in dict.fromkeys(r["방식"] for r in rows):
        s = [r for r in rows if r["방식"] == name]
        n = len(s)
        c = lambda k: sum(bool(r[k]) for r in s)
        print(f"{name:<12}{n:>4}{c('결과_의심'):>10}{c('원본_의심'):>10}{c('들킴'):>6}{c('위치까지'):>8}"
              f"{sum(r['위치알려줌_의심칸'] > 0 for r in s):>12}")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
