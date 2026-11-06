"""
탐지법별 혼동행렬(TP/FN/FP/TN) + 정밀도/정확도 표.

같은 bbox 위치에 대해
  - 위조본(dataset/tampered/...)에 돌려서 "탐지됨" → TP, 아니면 FN
  - 정상 원본(dataset/original/...)에 돌려서 "탐지됨" → FP, 아니면 TN
두 쪽 모두 같은 마스크 파일(labels/masks)을 써서 조건을 똑같이 맞춘다.

사용: python evaluate_detectors.py [--limit N] [--max-seconds S] [--fresh]
  샘플별 결과를 outputs/reports/detector_evaluation_cache.jsonl에 쌓아두고, 다시 실행하면
  이미 끝난 샘플은 건너뛴다 (중간에 끊겨도 이어서 가능). 임계치를 바꿨으면 --fresh로 새로 시작.
출력: 표 출력 + outputs/reports/detector_evaluation.json / .csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
from config import INPUT_DIR, MASK_DIR, META_DIR, REPORT_DIR, TAMPERED_DIR, ensure_dirs
import detectors as det

METHODS = [
    "압축흔적_ELA", "신호패턴_노이즈분석", "신호패턴_FFT주파수",
    "영역불일치_합성경계분석", "AI탐지_노이즈지문분석", "AI탐지_패치통계이상",
]


def run_all(image_path, mask_arr):
    gray = np.asarray(Image.open(image_path).convert("L"))
    bbox = det.mask_bbox_from_mask(mask_arr)
    return {
        "압축흔적_ELA": det.compute_ela_detect(image_path, mask_arr)[0],
        "신호패턴_노이즈분석": det.compute_noise_detect(gray, mask_arr, bbox)[0],
        "신호패턴_FFT주파수": det.compute_fft_detect(gray, bbox)[0],
        "영역불일치_합성경계분석": det.compute_boundary_detect(gray, mask_arr)[0],
        "AI탐지_노이즈지문분석": det.compute_noise_fingerprint_detect(gray, bbox)[0],
        "AI탐지_패치통계이상": det.compute_patch_cleanliness_detect(gray, bbox)[0],
    }


def find_original(base):
    for ext in (".jpg", ".jpeg", ".png"):
        p = INPUT_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="이 시간 지나면 저장하고 중단")
    ap.add_argument("--fresh", action="store_true", help="캐시 지우고 처음부터")
    args = ap.parse_args()
    import time
    t0 = time.time()
    cache_path = REPORT_DIR / "detector_evaluation_cache.jsonl"
    ensure_dirs()

    labels = sorted(META_DIR.glob("*_ai_inpaint.json"))
    if args.limit:
        labels = labels[: args.limit]

    if args.fresh and cache_path.exists():
        cache_path.unlink()
    done = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line); done[r["base"]] = r
    cache = open(cache_path, "a", encoding="utf-8")
    for i, lp in enumerate(labels, 1):
        base = lp.stem.replace("_ai_inpaint", "")
        if base in done:
            continue
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            cache.close()
            print(f"시간 제한 — {len(done)}/{len(labels)} 완료, 다시 실행하면 이어서 함")
            return
        orig = find_original(base)
        tamp = TAMPERED_DIR / f"{base}_ai_inpaint.png"
        mask = MASK_DIR / f"{base}_ai_inpaint_mask.png"
        if orig is None or not tamp.exists() or not mask.exists():
            rec = {"base": base, "skip": "파일 없음"}
        else:
            try:
                mask_arr = np.asarray(Image.open(mask).convert("L"))
                rec = {"base": base, "tamp": run_all(tamp, mask_arr), "orig": run_all(orig, mask_arr)}
            except Exception as exc:  # noqa: BLE001
                rec = {"base": base, "skip": str(exc)}
        rec = json.loads(json.dumps(rec, default=bool))
        done[base] = rec
        cache.write(json.dumps(rec, ensure_ascii=False) + "\n"); cache.flush()
        if i % 100 == 0:
            print(f"  {i}/{len(labels)}", flush=True)

    cache.close()
    cm = {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in METHODS}
    n, skipped = 0, []
    for r in done.values():
        if "skip" in r:
            skipped.append(f"{r['base']} ({r['skip']})"); continue
        n += 1
        for m in METHODS:
            cm[m]["TP" if r["tamp"][m] else "FN"] += 1
            cm[m]["FP" if r["orig"][m] else "TN"] += 1

    pct = lambda a, b: a / b * 100 if b else 0.0
    rows = []
    print(f"\n평가 샘플 {n}쌍 (위조본 {n} + 정상원본 {n}), 제외 {len(skipped)}")
    hdr = ["탐지법", "TP(정탐)", "FN(미탐)", "FP(오탐)", "TN", "정밀도", "정확도"]
    print("\t".join(hdr))
    for m in METHODS:
        c = cm[m]
        prec = pct(c["TP"], c["TP"] + c["FP"])
        acc = pct(c["TP"] + c["TN"], 2 * n)
        row = [m, f"{c['TP']} ({pct(c['TP'], n):.1f}%)", f"{c['FN']} ({pct(c['FN'], n):.1f}%)",
               f"{c['FP']} ({pct(c['FP'], n):.1f}%)", f"{c['TN']} ({pct(c['TN'], n):.1f}%)",
               f"{prec:.1f}%", f"{acc:.1f}%"]
        print("\t".join(row)); rows.append(row)

    with open(REPORT_DIR / "detector_evaluation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    json.dump({"n_pairs": n, "skipped": skipped, "thresholds": det.THRESHOLDS, "confusion": cm},
              open(REPORT_DIR / "detector_evaluation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {REPORT_DIR / 'detector_evaluation.csv'}")


if __name__ == "__main__":
    main()
