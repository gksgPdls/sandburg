"""
False-positive check for the pixel-level detection heuristics (ELA, noise
analysis, FFT frequency, boundary analysis, noise fingerprint, patch
cleanliness): run them against the UNTAMPERED original image at the exact
same bbox location that was later forged, instead of the forged image.

Nothing was changed there, so any method reporting "True" (detected) is a
false positive. This reuses the saved bbox from each sample's regions.json
— no new annotation needed.

Output: a text + JSON summary of per-method false-positive rate, so the
0.05 / 1.8x / etc. thresholds (picked by feel, not tuned) can be judged
against real data instead of assumed to work.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
from config import INPUT_DIR, META_DIR, REPORT_DIR, ensure_dirs
import detectors as det

METHODS = [
    ("압축흔적_ELA", "ela"),
    ("신호패턴_노이즈분석", "noise"),
    ("신호패턴_FFT주파수", "fft"),
    ("영역불일치_합성경계분석", "boundary"),
    ("AI탐지_노이즈지문분석(픽셀)", "noise_fingerprint"),
    ("AI탐지_패치통계이상(픽셀)", "patch_clean"),
]


def find_original(base):
    for ext in (".jpg", ".jpeg", ".png"):
        p = INPUT_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def run_on_untouched(original_path, bbox_dict):
    """Run every detector against the ORIGINAL image at the bbox location
    (same geometry build_mask would have used), with no forgery applied."""
    pad = 3  # matches ai_inpainting.PADDING, the mask margin used at generation time
    orig = Image.open(original_path).convert("RGB")
    gray = np.asarray(orig.convert("L"))
    x0 = max(bbox_dict["x"] - pad, 0)
    y0 = max(bbox_dict["y"] - pad, 0)
    x1 = min(bbox_dict["x"] + bbox_dict["w"] + pad, gray.shape[1])
    y1 = min(bbox_dict["y"] + bbox_dict["h"] + pad, gray.shape[0])
    bbox = (x0, y0, x1, y1)

    mask_arr = np.zeros(gray.shape, dtype=np.uint8)
    mask_arr[y0:y1, x0:x1] = 255

    results = {}

    ela_ok, *_ = det.compute_ela_detect(original_path, mask_arr)
    results["ela"] = ela_ok

    noise_ok, *_ = det.compute_noise_detect(gray, mask_arr, bbox)
    results["noise"] = noise_ok

    fft_ok, *_ = det.compute_fft_detect(gray, bbox)
    results["fft"] = fft_ok

    bnd_ok, *_ = det.compute_boundary_detect(gray, mask_arr)
    results["boundary"] = bnd_ok

    fp_ok, *_ = det.compute_noise_fingerprint_detect(gray, bbox)
    results["noise_fingerprint"] = fp_ok

    clean_ok, *_ = det.compute_patch_cleanliness_detect(gray, bbox)
    results["patch_clean"] = clean_ok

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2")
    parser.add_argument("--limit", type=int, default=None, help="max samples to check (default: all)")
    args = parser.parse_args()

    ensure_dirs()
    labels = sorted(META_DIR.glob(f"*_ai_inpaint_{args.version}.json"))
    legacy = False
    if not labels:
        labels = sorted(META_DIR.glob("*_ai_inpaint.json"))
        legacy = True
    if args.limit:
        labels = labels[: args.limit]
    print(f"Checking {len(labels)} sample(s) against their UNTOUCHED originals (legacy={legacy})")

    counts = {key: 0 for _, key in METHODS}
    errors = 0
    total = 0

    for label_path in labels:
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        version_suffix = "_ai_inpaint" if legacy else f"_ai_inpaint_{args.version}"
        base = label_path.stem.replace(version_suffix, "")
        original_path = find_original(base)
        if original_path is None:
            continue
        region = label["forged_regions"][0]
        try:
            results = run_on_untouched(original_path, region["bbox"])
            total += 1
            for _, key in METHODS:
                if results.get(key):
                    counts[key] += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1

    print(f"\n=== False-positive rate (정상 원본인데 '탐지됨'으로 잘못 판정한 비율) ===")
    print(f"{'방법':<20} {'오탐 수':>8} {'전체':>8} {'오탐률':>8}")
    report = {"version": args.version, "total_checked": total, "errors": errors, "methods": {}}
    for col, key in METHODS:
        rate = counts[key] / total * 100 if total else 0.0
        print(f"{col:<20} {counts[key]:>8} {total:>8} {rate:>7.1f}%")
        report["methods"][col] = {"false_positives": counts[key], "total": total, "false_positive_rate_pct": round(rate, 2)}

    out_path = REPORT_DIR / f"detector_false_positive_check_{args.version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print("\n주의: 오탐률이 높은 방법(예: >20~30%)은 지금 임계치가 너무 민감하다는 뜻 — 임계치를 완화해야 함.")


if __name__ == "__main__":
    main()
