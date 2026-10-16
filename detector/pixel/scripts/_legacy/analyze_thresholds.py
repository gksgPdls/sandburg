"""
Step 1 of threshold re-tuning: collect the RAW metric values (not just
True/False) that each detector computes, run against UNTOUCHED originals
(same bbox geometry as the later forgery, nothing actually changed there).

This tells us the real distribution of each metric on normal documents, so
thresholds in label_detections.py can be picked from an actual percentile
(e.g. "95th percentile of normal data") instead of guessed by feel.

Output: printed percentile table + JSON with the raw value arrays, so a
follow-up step can pick exact threshold numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
from config import INPUT_DIR, META_DIR, REPORT_DIR, ensure_dirs
import detectors as det

PAD = 3  # matches ai_inpainting.PADDING


def find_original(base):
    for ext in (".jpg", ".jpeg", ".png"):
        p = INPUT_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def collect_raw_values(original_path, bbox_dict):
    orig = Image.open(original_path).convert("RGB")
    gray = np.asarray(orig.convert("L"))
    x0 = max(bbox_dict["x"] - PAD, 0)
    y0 = max(bbox_dict["y"] - PAD, 0)
    x1 = min(bbox_dict["x"] + bbox_dict["w"] + PAD, gray.shape[1])
    y1 = min(bbox_dict["y"] + bbox_dict["h"] + PAD, gray.shape[0])
    bbox = (x0, y0, x1, y1)
    mask_arr = np.zeros(gray.shape, dtype=np.uint8)
    mask_arr[y0:y1, x0:x1] = 255

    values = {}

    _, ela_in, ela_out = det.compute_ela_detect(original_path, mask_arr)
    values["ela_ratio"] = ela_in / ela_out if ela_out > 1e-6 else float("nan")

    _, v_in, v_out = det.compute_noise_detect(gray, mask_arr, bbox)
    values["noise_ratio"] = v_in / v_out if v_out > 1e-6 else float("nan")

    _, hf_in, hf_out = det.compute_fft_detect(gray, bbox)
    values["fft_diff"] = abs(hf_in - hf_out)

    _, ring_mean, bg_mean = det.compute_boundary_detect(gray, mask_arr)
    values["boundary_ratio"] = ring_mean / bg_mean if bg_mean > 1e-6 else float("nan")

    _, fp_in, fp_out = det.compute_noise_fingerprint_detect(gray, bbox)
    values["noise_fingerprint_ratio"] = fp_in / fp_out if fp_out > 1e-6 else float("nan")

    _, c_in, c_out = det.compute_patch_cleanliness_detect(gray, bbox)
    values["patch_clean_ratio"] = c_in / c_out if c_out > 1e-6 else float("nan")

    return values


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    ensure_dirs()
    labels = sorted(META_DIR.glob(f"*_ai_inpaint_{args.version}.json"))
    legacy = False
    if not labels:
        labels = sorted(META_DIR.glob("*_ai_inpaint.json"))
        legacy = True
    print(f"Scanning {len(labels)} untouched-original samples (legacy={legacy})")

    all_values = {}
    errors = 0
    for label_path in labels:
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        suffix = "_ai_inpaint" if legacy else f"_ai_inpaint_{args.version}"
        base = label_path.stem.replace(suffix, "")
        original_path = find_original(base)
        if original_path is None:
            continue
        region = label["forged_regions"][0]
        try:
            vals = collect_raw_values(original_path, region["bbox"])
            for k, v in vals.items():
                if np.isfinite(v):
                    all_values.setdefault(k, []).append(v)
        except Exception:  # noqa: BLE001
            errors += 1

    print(f"\n정상 데이터 {len(labels) - errors}장 기준 실제 수치 분포 (백분위수)")
    print(f"{'지표':<28} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    report = {}
    for key, vals in all_values.items():
        arr = np.array(vals)
        pcts = np.percentile(arr, [50, 75, 90, 95, 99])
        print(f"{key:<28} {pcts[0]:>8.3f} {pcts[1]:>8.3f} {pcts[2]:>8.3f} {pcts[3]:>8.3f} {pcts[4]:>8.3f} {arr.max():>8.3f}")
        report[key] = {
            "n": len(vals), "p50": float(pcts[0]), "p75": float(pcts[1]),
            "p90": float(pcts[2]), "p95": float(pcts[3]), "p99": float(pcts[4]),
            "max": float(arr.max()), "min": float(arr.min()),
        }

    out_path = REPORT_DIR / f"threshold_distribution_{args.version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
