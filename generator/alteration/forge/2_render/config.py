"""Shared paths and constants for the ai_inpainting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = ROOT / "dataset" / "original" / "bankbook"
TAMPERED_DIR = ROOT / "dataset" / "tampered" / "ai_inpainting"
MASK_DIR = ROOT / "dataset" / "labels" / "masks"
META_DIR = ROOT / "dataset" / "labels" / "meta"

LOG_DIR = ROOT / "outputs" / "logs"
VIZ_DIR = ROOT / "outputs" / "viz"
REPORT_DIR = ROOT / "outputs" / "reports"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def ensure_dirs():
    for d in (INPUT_DIR, TAMPERED_DIR, MASK_DIR, META_DIR, LOG_DIR, VIZ_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
