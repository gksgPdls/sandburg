"""
신호패턴_노이즈분석 임계치 튜닝 + 검증 (튜닝/검증 데이터 분리).

1) 전체 샘플에서 같은 bbox 위치의 노이즈 비율(안쪽÷바깥쪽, 웨이블릿)을
   정상 원본 / 위조본 각각 계산한다.
2) 샘플을 무작위로 반반 나눠서
   - 튜닝용 절반: 정상 원본 분포의 하위 N% 지점 = 임계치 후보 (오탐 목표 N%)
   - 검증용 절반: 그 임계치로 실제 오탐률(정상 원본) / 탐지율(위조본) 측정
   → 임계치를 정한 데이터와 성능을 재는 데이터가 달라서 숫자를 믿을 수 있다.

사용: python tune_noise_threshold.py [--limit N] [--seed 0]
출력: outputs/reports/noise_threshold_tuning.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
from config import INPUT_DIR, META_DIR, REPORT_DIR, TAMPERED_DIR, ensure_dirs
import detectors as det

PAD = 3
FP_TARGETS = (0.05, 0.10)


def bbox_xyxy(b, shape):
    return (max(b["x"] - PAD, 0), max(b["y"] - PAD, 0),
            min(b["x"] + b["w"] + PAD, shape[1]), min(b["y"] + b["h"] + PAD, shape[0]))


def noise_ratio(path, b):
    gray = np.asarray(Image.open(path).convert("L"))
    box = bbox_xyxy(b, gray.shape)
    _, v_in, v_out = det.compute_noise_detect(gray, None, box)
    return v_in / v_out if v_out > 1e-6 else float("nan")


def rates(orig, tamp, low):
    return float(np.mean(orig < low)), float(np.mean(tamp < low))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()

    labels = sorted(META_DIR.glob("auto*_ai_inpaint.json"))
    if args.limit:
        labels = labels[: args.limit]

    rows = []
    for i, lp in enumerate(labels, 1):
        base = lp.stem.replace("_ai_inpaint", "")
        orig = INPUT_DIR / f"{base}.jpg"
        tamp = TAMPERED_DIR / f"{base}_ai_inpaint.png"
        if not (orig.exists() and tamp.exists()):
            continue
        b = json.load(open(lp, encoding="utf-8"))["forged_regions"][0]["bbox"]
        try:
            ro, rt = noise_ratio(orig, b), noise_ratio(tamp, b)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(ro) and np.isfinite(rt):
            rows.append((base, ro, rt))
        if i % 100 == 0:
            print(f"  {i}/{len(labels)}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    half = len(rows) // 2
    tune, test = idx[:half], idx[half:]
    o = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])

    cur_low = det.THRESHOLDS["noise_ratio_low"]
    report = {"n_total": len(rows), "n_tune": int(len(tune)), "n_test": int(len(test)),
              "seed": args.seed, "candidates": {}}
    print(f"\n샘플 {len(rows)}장 (튜닝 {len(tune)} / 검증 {len(test)})")
    print(f"{'기준':<22}{'임계치':>8}{'검증 오탐':>10}{'검증 탐지':>10}")

    fp, tp = rates(o[test], t[test], cur_low)
    print(f"{'현재 값':<22}{cur_low:>8.3f}{fp*100:>9.1f}%{tp*100:>9.1f}%")
    report["current"] = {"low": cur_low, "test_fp": fp, "test_tp": tp}

    for target in FP_TARGETS:
        low = float(np.percentile(o[tune], target * 100))
        fp, tp = rates(o[test], t[test], low)
        name = f"튜닝 오탐목표 {int(target*100)}%"
        print(f"{name:<22}{low:>8.3f}{fp*100:>9.1f}%{tp*100:>9.1f}%")
        report["candidates"][f"fp{int(target*100)}"] = {"low": low, "test_fp": fp, "test_tp": tp}

    auc = float(np.mean((-np.log(t[test]))[:, None] > (-np.log(o[test]))[None, :]))
    report["test_auc_lower_is_forged"] = auc
    report["orig_percentiles"] = {p: float(np.percentile(o, p)) for p in (5, 10, 50, 90, 95)}
    report["tamp_percentiles"] = {p: float(np.percentile(t, p)) for p in (5, 10, 50, 90, 95)}
    print(f"\n검증셋 구분력 AUC = {auc:.3f}  (0.5=동전던지기, 1.0=완벽)")

    out = REPORT_DIR / "noise_threshold_tuning.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
