"""
Visualize ai_inpainting results: per-file compare + mask overlay images,
plus a batch summary grid.

Reads the run identified by outputs/logs/latest_run.json (written by
ai_inpainting.py), or accepts --version/--timestamp overrides.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INPUT_DIR, TAMPERED_DIR, MASK_DIR, META_DIR, LOG_DIR, VIZ_DIR, ensure_dirs


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


def load_label(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_original(source_name):
    candidate = INPUT_DIR / source_name
    return candidate if candidate.exists() else None


def draw_regions(image, regions, color=(255, 0, 0), width=3):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for region in regions:
        bbox = region["bbox"]
        draw.rectangle(
            [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]],
            outline=color, width=width,
        )
    return out


def make_compare_image(label, original, forged, out_path, base, timestamp):
    original_boxed = draw_regions(original, label["forged_regions"])
    forged_boxed = draw_regions(forged, label["forged_regions"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_boxed)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(forged_boxed)
    axes[1].set_title("Forged (ai_inpainting)")
    axes[1].axis("off")

    fig.suptitle(f"{base} | {label['forgery_type']} | {timestamp}", fontsize=10)
    fig.text(0.99, 0.01, f"{base} · {timestamp}", ha="right", va="bottom",
              fontsize=7, color="gray", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_mask_overlay(original, mask, out_path):
    base_rgba = original.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    red_layer = Image.new("RGBA", base_rgba.size, (255, 0, 0, 120))
    overlay.paste(red_layer, (0, 0), mask.convert("L"))
    combined = Image.alpha_composite(base_rgba, overlay)
    combined.convert("RGB").save(out_path)


def make_summary_grid(entries, out_path, version, timestamp):
    n = len(entries)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, entry in zip(axes, entries):
        ax.axis("off")
        if entry["image"] is not None:
            ax.imshow(entry["image"])
        color = "green" if entry["status"] == "success" else "red"
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_title(entry["base"], fontsize=8, color=color)
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                              edgecolor=color, linewidth=4)
        ax.add_patch(rect)

    for ax in axes[len(entries):]:
        ax.axis("off")

    fig.suptitle(f"ai_inpainting {version} summary — {timestamp}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


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

    summary_entries = []
    for label_path in labels:
        label = load_label(label_path)
        base = label_path.stem.replace(f"_ai_inpaint_{version}", "")
        source_name = label["source"]

        original_path = find_original(source_name)
        forged_path = TAMPERED_DIR / f"{base}_ai_inpaint_{version}.png"
        mask_path = MASK_DIR / f"{base}_ai_inpaint_mask_{version}.png"

        if original_path is None or not forged_path.exists():
            print(f"  [skip] {base}: missing original or forged image")
            summary_entries.append({"base": base, "status": "failed", "image": None})
            continue

        try:
            original = Image.open(original_path).convert("RGB")
            forged = Image.open(forged_path).convert("RGB")

            compare_path = run_dir / f"{base}_compare.png"
            make_compare_image(label, original, forged, compare_path, base, timestamp)

            if mask_path.exists():
                mask = Image.open(mask_path)
                overlay_path = run_dir / f"{base}_mask_overlay.png"
                make_mask_overlay(original, mask, overlay_path)

            print(f"  [ok] {base} -> {compare_path.name}")
            summary_entries.append({"base": base, "status": "success", "image": forged})
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {base}: {exc}")
            summary_entries.append({"base": base, "status": "failed", "image": None})

    summary_path = run_dir / "summary_grid.png"
    make_summary_grid(summary_entries, summary_path, version, timestamp)
    print(f"Summary grid: {summary_path}")
    print(f"Visualization output dir: {run_dir}")


if __name__ == "__main__":
    main()
