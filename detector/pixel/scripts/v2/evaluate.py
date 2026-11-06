"""
A 검사 평가 (원본 없이, 칸 위치도 안 알려주고 문서 전체를 검사하는 실사용 방식).

  python scripts/detect/v2/evaluate.py [--checks A0_기존탐지기,A8_형식] [--fpr 0.10] [--max-seconds 150]

1. 검사마다 모든 문서의 칸 점수를 계산해 outputs/평가셋/scores_<검사>.jsonl 에 저장 (끊기면 이어서)
2. 문서 점수 = 가장 의심스러운 칸 점수. 기준값 = 튜닝용 정상 문서에서 오탐이 --fpr 되는 지점
3. 시험용으로: 정상 문서 오탐률(95% 구간), 정상 칸 오탐률, 묶음별 정탐·미탐·재현율·정밀도(정상과 위조가 같은 수라고 볼 때)·F1·AUC,
   위조 문서의 안 고친 칸 오탐률, 위조 칸 정탐, 위치 적중률(가장 의심 칸이 정답 칸과 70% 이상 겹침)
결과: outputs/평가셋/평가결과_<날짜>.csv + 화면 표
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.common.cells import group_cells  # noqa: E402
from detector.v2.checks import CHECKS  # noqa: E402

OUT = Path(os.environ.get("EVALSET_DIR", ROOT / "outputs" / "평가셋"))   # 시험할 때 다른 폴더로


def load_manifest():
    return list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig")))


def load_ocr():
    p = OUT / "ocr.jsonl"
    if not p.exists():
        sys.exit("먼저 OCR 캐시: python scripts/detect/v2/ocr_cache.py")
    return {r["id"]: r.get("cells") for r in map(json.loads, open(p, encoding="utf-8"))}


def mask_boxes(path):
    if not path:
        return []
    import cv2
    m = (np.asarray(Image.open(ROOT / path).convert("L")) > 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n)]


def hit(b, gts, need=0.7):
    """표시 상자가 정답 칸과 겹치는 면적 / 둘 중 작은 상자 면적 >= need"""
    x, y, w, h = b
    for gx, gy, gw, gh in gts:
        ix = max(0, min(x + w, gx + gw) - max(x, gx)); iy = max(0, min(y + h, gy + gh) - max(y, gy))
        if ix * iy >= need * max(1, min(w * h, gw * gh)):
            return True
    return False


def score_all(name, rows, ocr, max_seconds):
    p = OUT / f"scores_{name}.jsonl"
    done = {json.loads(l)["id"] for l in open(p, encoding="utf-8")} if p.exists() else set()
    fn, t0, n = CHECKS[name], time.time(), 0
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in done or ocr.get(r["id"]) is None:
                continue
            if time.time() - t0 > max_seconds:
                print(f"  {name}: 시간 제한, 다시 실행하면 이어서"); return False
            cells = [dict(c, bbox=tuple(c["bbox"])) for c in ocr[r["id"]]]
            img = Image.open(ROOT / r["path"]).convert("RGB")
            res = fn(img, cells, group_cells(cells))
            f.write(json.dumps({"id": r["id"], "cells": [{**c, "bbox": list(c["bbox"])} for c in res]},
                               ensure_ascii=False) + "\n")
            n += 1
    return True


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / d, (c + h) / d


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    a, b = np.array(pos, float), np.array(neg, float)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def report(name, rows, fpr):
    """문서 점수 = 가장 의심스러운 칸 점수. 기준값 = 튜닝용 정상 문서 오탐 fpr 지점. 시험용으로 문서·칸 단위 수치."""
    sc = {}
    for l in open(OUT / f"scores_{name}.jsonl", encoding="utf-8"):
        d = json.loads(l)
        sc[d["id"]] = sorted(d["cells"], key=lambda c: -c["score"])
    dscore = lambda i: sc[i][0]["score"] if sc.get(i) else -1e9
    tune_norm = [dscore(r["id"]) for r in rows if r["split"] == "tune" and r["label"] == "normal" and r["id"] in sc]
    thr = float(np.quantile(tune_norm, 1 - fpr)) if tune_norm else 0.0
    test = [r for r in rows if r["split"] == "test" and r["id"] in sc]
    norm = [r for r in test if r["label"] == "normal"]
    fp = sum(dscore(r["id"]) > thr for r in norm)
    ncells = sum(len(sc[r["id"]]) for r in norm)
    nflag = sum(c["score"] > thr for r in norm for c in sc[r["id"]])
    out = []
    lo, hi = wilson(fp, len(norm))
    clo, chi = wilson(nflag, ncells)
    out.append({"검사": name, "묶음": "정상 문서", "장수": len(norm), "정탐": "", "미탐": "", "오탐": fp, "정상통과": len(norm) - fp,
                "재현율": "", "오탐률": f"{fp / max(len(norm), 1):.1%} ({lo:.1%}~{hi:.1%})", "정밀도": "", "F1": "", "AUC": "",
                "칸오탐": f"{nflag}/{ncells} = {nflag / max(ncells, 1):.1%} ({clo:.1%}~{chi:.1%})", "안고친칸오탐": "", "위조칸정탐": "",
                "위치적중": ""})
    by = defaultdict(list)
    for r in test:
        if r["label"] == "forged":
            by[(r["group"], r["method"])].append(r)
    for (g, m), rs in sorted(by.items()):
        tp_rs = [r for r in rs if dscore(r["id"]) > thr]
        tp, fn = len(tp_rs), len(rs) - len(tp_rs)
        prec = tp / max(tp + fp * len(rs) / max(len(norm), 1), 1e-9)   # 정상과 위조가 같은 수라고 볼 때
        rec = tp / len(rs)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        a = auc([dscore(r["id"]) for r in rs], [dscore(r["id"]) for r in norm])
        lo, hi = wilson(tp, len(rs))
        other = tgt = tgt_hit = other_flag = 0
        loc = 0
        for r in rs:
            gts = mask_boxes(r["mask"])
            for c in sc[r["id"]]:
                if gts and hit(tuple(c["bbox"]), gts, need=0.3):
                    tgt += 1; tgt_hit += c["score"] > thr
                else:
                    other += 1; other_flag += c["score"] > thr
            if r in tp_rs and gts and hit(tuple(sc[r["id"]][0]["bbox"]), gts):
                loc += 1
        out.append({"검사": name, "묶음": f"{g}/{m}", "장수": len(rs), "정탐": tp, "미탐": fn, "오탐": "", "정상통과": "",
                    "재현율": f"{rec:.1%} ({lo:.1%}~{hi:.1%})", "오탐률": "", "정밀도": f"{prec:.1%}", "F1": f"{f1:.2f}",
                    "AUC": f"{a:.3f}", "칸오탐": "",
                    "안고친칸오탐": f"{other_flag}/{other} = {other_flag / max(other, 1):.1%}",
                    "위조칸정탐": f"{tgt_hit}/{tgt}", "위치적중": f"{loc}/{tp}"})
    return thr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default=",".join(CHECKS))
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    rows, ocr = load_manifest(), load_ocr()
    allrows = []
    for name in [c.strip() for c in a.checks.split(",") if c.strip()]:
        if not score_all(name, rows, ocr, a.max_seconds):
            return
        thr, res = report(name, rows, a.fpr)
        print(f"\n[{name}]  기준값 {thr:.4f} (튜닝용 정상 오탐 {a.fpr:.0%} 지점)")
        for r in res:
            print(f"  {r['묶음']}: " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("검사", "묶음") and v != ""))
        allrows += res
    p = OUT / f"평가결과_{date.today():%Y%m%d}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
