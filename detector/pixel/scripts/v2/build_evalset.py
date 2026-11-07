"""
A 검사(픽셀·글자 쪽 새 탐지) 평가셋 만들기 (2026-10-30).

  python detector/pixel/scripts/v2/build_evalset.py [--max-seconds 150]   # 끊기면 다시 실행하면 이어서

결과: outputs/평가셋/manifest.csv  (+ outputs/평가셋/jpg/ 다시 저장한 위조본)
  id, path, label(normal/forged), group, method, mask, split(tune/test), source_id
  group: 정상 / 티나는변조(v1 흰 박스, 동현님 샘플) / 고도화변조(v2_한칸_평가용 300장 + VLM_8·이전논리_8·한칸_8) / 복사붙여넣기(칸_복사 400장)
  split: v1 원본·위조는 기존 탐지기 튜닝과 같은 나눔(seed 0), 나머지는 id로 고정 나눔

주의: v1 위조본은 PNG, 원본은 JPEG라 그대로 비교하면 '파일 형식'을 잡는다.
그래서 위조본을 원본과 같은 압축표·색 샘플링으로 JPEG 다시 저장한 사본을 쓴다 (harmonize.save_like와 같은 방식).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "dataset"
OUT = ROOT / "outputs" / "평가셋"
FIELDS = ["id", "path", "label", "group", "method", "mask", "split", "source_id"]


def v1_split():
    recs = [json.loads(l) for l in open(ROOT / "outputs/reports/detector_measurements.jsonl", encoding="utf-8")]
    bases = [r["base"] for r in recs if "skip" not in r]
    idx = np.random.default_rng(0).permutation(len(bases))
    tune = {bases[i] for i in idx[: len(bases) // 2]}
    return tune, set(bases)


def hsplit(key):
    return "tune" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def save_like(src_png, orig_jpg, dst):
    """원본이 JPEG면 같은 압축표·샘플링으로 다시 저장 (형식 차이 제거)."""
    im = Image.open(src_png).convert("RGB")
    o = Image.open(orig_jpg)
    if o.format == "JPEG" and getattr(o, "quantization", None):
        kw = {"qtables": o.quantization}
        try:
            sub = JpegImagePlugin.get_sampling(o)
            if sub in (0, 1, 2):
                kw["subsampling"] = sub
        except Exception:
            pass
        dst = dst.with_suffix(".jpg")
        im.save(dst, **kw)
    else:
        dst = dst.with_suffix(".png")
        im.save(dst)
    return dst


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    t0 = time.time()
    (OUT / "jpg").mkdir(parents=True, exist_ok=True)
    tune, v1bases = v1_split()
    rows, seen_norm = [], set()
    labels = list(csv.DictReader(open(DATA / "labels_all.csv", encoding="utf-8-sig")))
    done_all = True
    for r in labels:
        smp = DATA / "samples"
        orig = (smp / r["original"]).resolve() if r["original"] else None
        tamp = (smp / r["tampered"]).resolve() if r["tampered"] else None
        mask = (smp / r["mask"]).resolve() if r["mask"] else None
        if r["method"] == "ai_inpainting":
            base = orig.stem
            if base not in v1bases:
                continue
            sp = "tune" if base in tune else "test"
            dst = OUT / "jpg" / f"{r['id']}"
            have = [p for p in (dst.with_suffix(".jpg"), dst.with_suffix(".png")) if p.exists()]
            if have:
                tp = have[0]
            elif time.time() - t0 > a.max_seconds:
                done_all = False
                continue
            else:
                tp = save_like(tamp, orig, dst)
            rows.append({"id": r["id"], "path": rel(tp), "label": "forged", "group": "티나는변조",
                         "method": "v1_흰박스", "mask": rel(mask), "split": sp, "source_id": base})
            if base not in seen_norm:
                seen_norm.add(base)
                rows.append({"id": base, "path": rel(orig), "label": "normal", "group": "정상",
                             "method": "원본", "mask": "", "split": sp, "source_id": base})
        else:
            sp = hsplit(r["id"].split("_")[0] + r["id"])
            if r["method"] != "normal" and tamp and tamp.exists():
                rows.append({"id": r["id"], "path": rel(tamp), "label": "forged", "group": "티나는변조",
                             "method": r["method"], "mask": rel(mask) if mask and mask.exists() else "",
                             "split": sp, "source_id": r["id"]})
            if orig and orig.exists() and str(orig) not in seen_norm:
                seen_norm.add(str(orig))
                rows.append({"id": r["id"] + "_orig", "path": rel(orig), "label": "normal", "group": "정상",
                             "method": "샘플원본", "mask": "", "split": sp, "source_id": r["id"]})
    # 고도화 변조 (오늘 만든 세 방식): 오탐 확인용
    # v2_한칸_평가용 (generator/alteration/run/8, 300장): 원본의 튜닝/시험 나눔을 그대로 따라감 -> v2 탐지기 기준값 튜닝에도 씀
    for folder, method in (("v2_한칸_평가용", "고도화_한칸"), ("VLM_8", "고도화_VLM"), ("이전논리_8", "고도화_자동규칙"), ("한칸_8", "고도화_한칸_8장")):
        d = ROOT / "outputs/고도화" / folder / "AI"
        if not d.exists():
            continue
        for f in sorted(d.glob("*_mask.png")):
            sid = f.name[:-9]
            p = next((q for q in (d / f"{sid}.jpg", d / f"{sid}.png") if q.exists()), None)
            if p:
                base = sid.split("_", 1)[-1]
                sp = ("tune" if base in tune else "test") if folder == "v2_한칸_평가용" else "test"
                rows.append({"id": f"{folder}_{sid}", "path": rel(p), "label": "forged", "group": "고도화변조",
                             "method": method, "mask": rel(f), "split": sp, "source_id": base})
    # 복사-붙여넣기 평가용 (generator/alteration/forge/make_copymove.py, 11-06): 원본의 튜닝/시험 나눔을 그대로
    cm = ROOT / "outputs/고도화/copymove_평가용/AI"
    if (cm / "목록.csv").exists():
        for r in csv.DictReader(open(cm / "목록.csv", encoding="utf-8-sig")):
            p = next((q for q in (cm / f"{r['id']}.jpg", cm / f"{r['id']}.png") if q.exists()), None)
            if p:
                rows.append({"id": r["id"], "path": rel(p), "label": "forged", "group": "복사붙여넣기",
                             "method": "칸_복사", "mask": rel(cm / f"{r['id']}_mask.png"), "split": r["split"],
                             "source_id": r["source_id"]})
    with open(OUT / "manifest.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    c = Counter((r["group"], r["split"]) for r in rows)
    print("평가셋:", dict(sorted(c.items())), "합계", len(rows))
    print("완료" if done_all else "시간 제한 - 다시 실행하면 이어서 (위조본 JPEG 다시 저장 중)")


if __name__ == "__main__":
    main()
