"""
같은 숫자 비교(digit_shape)가 고친 칸과 안 고친 칸을 얼마나 가르는지 (AUC) — 변조 위치는 채점에만 씀.

  python detector/pixel/scripts/v2/analyze_digits.py [--group 고도화변조] [--normal 40]
결과: 화면 표 + outputs/평가셋/숫자비교_<group>.csv
"""
import argparse, csv, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from analyze_diff import auc, mask_boxes, overlap  # noqa: E402
from detector.pixel.v2.digit_shape import digit_scores  # noqa: E402

OUT = Path(os.environ.get("EVALSET_DIR", ROOT / "outputs" / "평가셋"))
KEYS = ["score", "shape", "aspect", "height", "dark"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="고도화변조")
    ap.add_argument("--normal", type=int, default=40, help="정상 문서 몇 장으로 칸 오탐 기준을 볼지 (tune 쪽)")
    a = ap.parse_args()
    man = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig")))
    cache = {d["id"]: d.get("cells") for d in map(json.loads, open(OUT / "ocr.jsonl", encoding="utf-8"))}
    rows = [r for r in man if r["group"] == a.group and cache.get(r["id"])]
    norm = [r for r in man if r["label"] == "0" and r["split"] == "tune" and cache.get(r["id"])][:a.normal]
    pos = {k: [] for k in KEYS}; neg = {k: [] for k in KEYS}; nrm = {k: [] for k in KEYS}
    n_edit = n_edit_scored = 0; recs = []
    for r in rows + norm:
        cells = cache[r["id"]]
        g = cv2.imread(str(ROOT / r["path"]), 0)
        sc = digit_scores(g, cells)
        gts = mask_boxes(r["mask"]) if r.get("mask") else []
        for c in cells:
            if c["kind"] not in ("money", "number"):
                continue
            edited = bool(gts) and overlap(c["bbox"], gts)
            if r in rows and edited:
                n_edit += 1
            d = sc.get(c["id"])
            if not d:
                continue
            tgt = nrm if r in norm else (pos if edited else neg)
            n_edit_scored += int(r in rows and edited)
            for k in KEYS:
                tgt[k].append(d[k])
            recs.append({"id": r["id"], "group": "정상" if r in norm else a.group, "cell": c["id"], "text": c["text"],
                         "edited": int(edited), **{k: round(d[k], 3) for k in KEYS}, "worst": d["worst"]})
    print(f"[{a.group}] 문서 {len(rows)}장: 고친 칸 {n_edit}개 중 숫자비교 점수 받은 칸 {n_edit_scored} "
          f"({n_edit_scored / max(n_edit, 1):.0%}), 안 고친 칸 {len(neg['score'])}, 정상문서 {len(norm)}장 칸 {len(nrm['score'])}")
    print(f"{'특징':8s} {'AUC(고친 vs 안고친)':>18s} {'AUC(고친 vs 정상)':>16s} {'고친 z중앙':>9s} {'안고친':>7s} {'정상':>7s}")
    for k in KEYS:
        md = lambda v: f"{np.median(v):.2f}" if v else "-"
        print(f"{k:8s} {auc(pos[k], neg[k]):18.3f} {auc(pos[k], nrm[k]):16.3f} {md(pos[k]):>9s} {md(neg[k]):>7s} {md(nrm[k]):>7s}")
    if nrm["score"]:
        for q in (0.90, 0.95):
            th = float(np.quantile(nrm["score"], q))
            tp = np.mean(np.array(pos["score"]) > th) if pos["score"] else float("nan")
            fp = np.mean(np.array(neg["score"]) > th) if neg["score"] else float("nan")
            print(f"정상칸 {int(q * 100)}% 기준 z>{th:.2f}: 고친 칸 TP {tp:.1%}, 같은 문서 안 고친 칸 FP {fp:.1%}, 정상칸 FP {1 - q:.0%}")
    with open(OUT / f"숫자비교_{a.group}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)


if __name__ == "__main__":
    main()
