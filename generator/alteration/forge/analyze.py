"""
Analyze ai_inpainting results: batch stats, PSNR/SSIM quality metrics,
pixel-diff heatmaps, and an ELA-based auto-detectability check.

Reads the run identified by outputs/logs/latest_run.json (written by
ai_inpainting.py), or accepts --version/--timestamp overrides.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INPUT_DIR, TAMPERED_DIR, MASK_DIR, META_DIR, LOG_DIR, VIZ_DIR, REPORT_DIR, ensure_dirs

ELA_JPEG_QUALITY = 90
ELA_SCALE = 15


def load_run_info(args):
    if args.version and args.timestamp:
        return args.version, args.timestamp
    latest_run_path = LOG_DIR / "latest_run.json"
    if not latest_run_path.exists():
        raise RuntimeError("No latest_run.json found; run ai_inpainting.py first or pass --version/--timestamp")
    with open(latest_run_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info["version"], info["timestamp"]


def list_labels(version):
    return sorted(META_DIR.glob(f"*_ai_inpaint_{version}.json"))


def compute_ela(image_path, out_path):
    original = Image.open(image_path).convert("RGB")
    from io import BytesIO

    buf = BytesIO()
    original.save(buf, format="JPEG", quality=ELA_JPEG_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(original, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)
    max_diff = diff_arr.max() if diff_arr.max() > 0 else 1.0
    scale = min(255.0 / max_diff, ELA_SCALE)
    ela_arr = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)

    ela_image = Image.fromarray(ela_arr)
    ela_image.save(out_path)
    return float(diff_arr.mean()), float(diff_arr.max())


def region_score(diff_arr, mask_arr):
    """Mean ELA intensity inside vs outside the forged mask."""
    inside = diff_arr[mask_arr > 0]
    outside = diff_arr[mask_arr == 0]
    inside_mean = float(inside.mean()) if inside.size else 0.0
    outside_mean = float(outside.mean()) if outside.size else 0.0
    return inside_mean, outside_mean


def analyze_pair(label, base, run_dir, version):
    source_name = label["source"]
    original_path = INPUT_DIR / source_name
    forged_path = TAMPERED_DIR / f"{base}_ai_inpaint_{version}.png"
    mask_path = MASK_DIR / f"{base}_ai_inpaint_mask_{version}.png"

    if not original_path.exists() or not forged_path.exists():
        return {"base": base, "status": "failed", "reason": "missing_image"}

    original = np.asarray(Image.open(original_path).convert("RGB").resize(
        Image.open(forged_path).size
    )).astype(np.float64)
    forged = np.asarray(Image.open(forged_path).convert("RGB")).astype(np.float64)

    psnr = peak_signal_noise_ratio(original, forged, data_range=255)
    ssim = structural_similarity(original, forged, channel_axis=2, data_range=255)

    diff = np.abs(original - forged).mean(axis=2)
    heatmap_path = run_dir / f"{base}_diff_heatmap.png"
    heatmap_norm = (diff / (diff.max() if diff.max() > 0 else 1.0) * 255).astype(np.uint8)
    Image.fromarray(heatmap_norm).save(heatmap_path)

    ela_path = run_dir / f"{base}_ela.png"
    ela_mean, ela_max = compute_ela(forged_path, ela_path)

    ela_image = np.asarray(Image.open(ela_path).convert("L")).astype(np.float64)
    detectable = None
    inside_mean = outside_mean = None
    if mask_path.exists():
        mask_arr = np.asarray(Image.open(mask_path).convert("L").resize(
            (ela_image.shape[1], ela_image.shape[0])
        ))
        inside_mean, outside_mean = region_score(ela_image, mask_arr)
        detectable = bool(inside_mean > outside_mean * 1.5 and inside_mean > 5)

    return {
        "base": base,
        "status": "ok",
        "psnr": float(psnr),
        "ssim": float(ssim),
        "ela_mean": ela_mean,
        "ela_max": ela_max,
        "ela_region_inside_mean": inside_mean,
        "ela_region_outside_mean": outside_mean,
        "ela_detectable": detectable,
        "heatmap_path": str(heatmap_path),
        "ela_path": str(ela_path),
    }


def classify_failure(reason):
    reason_lower = (reason or "").lower()
    if "api" in reason_lower or "gemini" in reason_lower or "inline" in reason_lower:
        return "api_error"
    if "image" in reason_lower or "size" in reason_lower:
        return "image_error"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version")
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    ensure_dirs()
    version, timestamp = load_run_info(args)

    run_dir = VIZ_DIR / f"{version}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = list_labels(version)
    print(f"Found {len(labels)} label(s) in {META_DIR} (version={version})")

    log_path = LOG_DIR / f"ai_inpainting_{version}_{timestamp}_q95.log"
    failure_reasons = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "FAILED" in line:
                failure_reasons.append(line)

    per_file_results = []
    for label_path in labels:
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        base = label_path.stem.replace(f"_ai_inpaint_{version}", "")
        result = analyze_pair(label, base, run_dir, version)
        per_file_results.append(result)
        status = result["status"]
        if status == "ok":
            print(f"  [ok] {base}: psnr={result['psnr']:.2f} ssim={result['ssim']:.4f} "
                  f"detectable={result['ela_detectable']}")
        else:
            print(f"  [fail] {base}: {result.get('reason')}")

    ok_results = [r for r in per_file_results if r["status"] == "ok"]
    failure_categories = {}
    for reason in failure_reasons:
        category = classify_failure(reason)
        failure_categories[category] = failure_categories.get(category, 0) + 1

    report = {
        "version": version,
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(),
        "batch_stats": {
            "total_labels": len(labels),
            "analyzed_ok": len(ok_results),
            "analyzed_failed": len(per_file_results) - len(ok_results),
            "success_rate": (len(ok_results) / len(labels)) if labels else 0.0,
            "failure_categories": failure_categories,
        },
        "quality_metrics": {
            "mean_psnr": float(np.mean([r["psnr"] for r in ok_results])) if ok_results else None,
            "mean_ssim": float(np.mean([r["ssim"] for r in ok_results])) if ok_results else None,
        },
        "ela_summary": {
            "detectable_count": sum(1 for r in ok_results if r.get("ela_detectable")),
            "not_detectable_count": sum(1 for r in ok_results if r.get("ela_detectable") is False),
            "unknown_count": sum(1 for r in ok_results if r.get("ela_detectable") is None),
        },
        "per_file": per_file_results,
    }

    report_path = REPORT_DIR / f"ai_inpainting_{version}_{timestamp}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
