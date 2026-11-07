"""
v2 변조가 '같은 문서의 같은 종류 칸'과 어떤 글자 특징에서 다른지 분석 (어떤 검사를 만들지 정하는 용도, 2026-11-06).

  python detector/pixel/scripts/v2/analyze_diff.py [--group 고도화변조] [--max-seconds 150]

위조 문서마다: 칸 찾기(평가셋 OCR 캐시를 같이 씀, 없으면 여기서 읽고 캐시에 추가) -> 칸마다 글자 특징 -> 같은 묶음 안에서
튀는 정도 z = |값 - 중앙값| / (1.4826 x MAD). 정답 마스크와 겹치는 칸 = 고친 칸.
특징마다 '고친 칸의 z가 안 고친 칸의 z보다 큰가'를 AUC로 (0.5 = 구분 못 함, 1 = 완벽).
원본과 같은 칸의 값 차이도 같이 적는다 (원본은 분석에만 쓰고, 탐지에는 안 씀).
결과: outputs/평가셋/차이분석_<group>.csv + 화면 표
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from detector.pixel.common.cells import group_cells, read_cells  # noqa: E402
from detector.pixel.v2.glyph import cell_features  # noqa: E402

OUT = Path(os.environ.get("EVALSET_DIR", ROOT / "outputs" / "평가셋"))
FEATS = ["height", "stroke_w", "ink", "contrast", "edge_w", "halo", "gap", "gap_comma", "angle"]


def mask_boxes(path):
    m = (cv2.imread(str(ROOT / path), 0) > 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n)]


def overlap(b, gts):
    x, y, w, h = b
    for gx, gy, gw, gh in gts:
        ix = max(0, min(x + w, gx + gw) - max(x, gx)); iy = max(0, min(y + h, gy + gh) - max(y, gy))
        if ix * iy >= 0.3 * max(1, min(w * h, gw * gh)):
            return True
    return False


def robust_z(vals):
    v = np.array([np.nan if x is None else x for x in vals], float)
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return [None] * len(vals)
    med = np.median(v[ok]); mad = np.median(np.abs(v[ok] - med)) * 1.4826
    mad = max(mad, 1e-6, 0.05 * abs(med))
    return [None if np.isnan(x) else float(abs(x - med) / mad) for x in v]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    a, b = np.array(pos), np.array(neg)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="고도화변조")
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig")) if r["group"] == a.group]
    cache_p = OUT / "ocr.jsonl"
    cache = {d["id"]: d.get("cells") for d in map(json.loads, open(cache_p, encoding="utf-8"))} if cache_p.exists() else {}
    t0 = time.time()
    with open(cache_p, "a", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in cache:
                continue
            if time.time() - t0 > a.max_seconds:
                print("시간 제한: 칸 찾기 중, 다시 실행하면 이어서"); return
            cells = read_cells(ROOT / r["path"])
            cache[r["id"]] = cells
            f.write(json.dumps({"id": r["id"], "cells": cells}, ensure_ascii=False) + "\n"); f.flush()
    pos = {k: [] for k in FEATS}; neg = {k: [] for k in FEATS}; diffs = {k: [] for k in FEATS}
    nf = nu = 0
    for r in rows:
        cells = [dict(c, bbox=tuple(c["bbox"])) for c in cache[r["id"]]]
        g = cv2.imread(str(ROOT / r["path"]), 0)
        base = r["source_id"]
        og = cv2.imread(str(next(p for p in (ROOT / "dataset/original/bankbook").glob(base + ".*")
                                 if p.suffix.lower() in (".jpg", ".jpeg", ".png"))), 0)
        gts = mask_boxes(r["mask"])
        feats = {c["id"]: cell_features(g, c["bbox"]) for c in cells if c["kind"] in ("money", "number", "date")}
        for gid, members in group_cells(cells).items():
            ms = [c for c in members if feats.get(c["id"])]
            if len(ms) < 3:
                continue
            for k in FEATS:
                zs = robust_z([feats[c["id"]][k] for c in ms])
                for c, z in zip(ms, zs):
                    if z is None:
                        continue
                    (pos if overlap(c["bbox"], gts) else neg)[k].append(z)
        for c in cells:
            if feats.get(c["id"]) and overlap(c["bbox"], gts):
                nf += 1
                fo = cell_features(og, c["bbox"])
                if fo:
                    for k in FEATS:
                        if feats[c["id"]][k] is not None and fo[k] is not None:
                            diffs[k].append(feats[c["id"]][k] - fo[k])
            elif feats.get(c["id"]):
                nu += 1
    res = []
    for k in FEATS:
        d = np.array(diffs[k]) if diffs[k] else np.array([np.nan])
        res.append({"특징": k, "AUC(고친칸 z > 안고친칸 z)": round(auc(pos[k], neg[k]), 3), "고친칸 수": len(pos[k]),
                    "안고친칸 수": len(neg[k]), "고친칸 z 중앙값": round(float(np.median(pos[k])), 2) if pos[k] else "",
                    "안고친칸 z 중앙값": round(float(np.median(neg[k])), 2) if neg[k] else "",
                    "원본 대비 변화 중앙값": round(float(np.nanmedian(d)), 4), "변화 절댓값 중앙값": round(float(np.nanmedian(np.abs(d))), 4)})
    res.sort(key=lambda r: -(r["AUC(고친칸 z > 안고친칸 z)"] if r["AUC(고친칸 z > 안고친칸 z)"] == r["AUC(고친칸 z > 안고친칸 z)"] else 0))
    print(f"[{a.group}] 문서 {len(rows)}장, 특징을 잰 고친 칸 {nf} / 안 고친 칸 {nu}")
    for r in res:
        print("  " + ", ".join(f"{k} {v}" for k, v in r.items()))
    with open(OUT / f"차이분석_{a.group}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)


if __name__ == "__main__":
    main()
