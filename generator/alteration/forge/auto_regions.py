"""
원본 문서에서 위조할 금액/날짜 필드를 OCR로 자동 찾아서, dataset/original/bankbook/에
{base}.regions.json(위조 지시서)과 이미지 복사본을 만든다. 사람이 일일이 좌표를
찾지 않아도 ai_inpainting.py를 대량으로 돌릴 수 있게 하기 위함.

원본(dataset/original/real_docs/)은 읽기만 하고, 쓰기는 dataset/original/bankbook/에만 한다.

절차:
  1. tesseract OCR로 원본 문서 전체 단어+좌표 추출
  2. 후보 필터링 — 금액 패턴(쉼표로 3자리씩 끊긴 숫자, 예: 140,997) 또는
     날짜 패턴(YYYY-MM-DD / YYYY.MM.DD)에 맞는 단어만 후보로 남김.
     즉 숫자(금액)·날짜만 대상이고, 환자명/병명/주소 등 다른 텍스트는
     애초에 후보에서 제외됨
  3. 랜덤 선택 — 후보가 여러 개면 그중 완전 랜덤으로 1개(MAX_REGIONS_PER_IMAGE)를
     고름. "총액"처럼 중요한 필드를 의도적으로 고르는 게 아니라, 금액/날짜
     후보 중 아무거나 랜덤하게 뽑힘 (운에 따라 작은 항목이 걸릴 수도 있음)
  4. 값도 랜덤 변경 — 금액이면 2~10배 중 랜덤 배수, 날짜면 ±14~60일 중 랜덤 이동
  5. 원본을 bankbook/에 복사 + {base}.regions.json 자동 생성
"""
import argparse
import json
import random
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INPUT_DIR, IMAGE_EXTENSIONS, ensure_dirs

MAX_REGIONS_PER_IMAGE = 1
MIN_OCR_CONFIDENCE = 40
MONEY_RE = re.compile(r"^\d{1,3}(,\d{3})+$")          # e.g. 140,997
DATE_RE = re.compile(r"^\d{4}[-.]\d{2}[-.]\d{2}$")     # e.g. 2025-01-31

random.seed(0)  # reproducible picks across runs


def find_source_images(source_dir, limit=None):
    files = []
    for p in sorted(Path(source_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
            if limit and len(files) >= limit:
                break
    return files


def ocr_words(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] not in ("-1", "") else -1
        if not text or conf < MIN_OCR_CONFIDENCE:
            continue
        words.append({
            "text": text,
            "x": data["left"][i], "y": data["top"][i],
            "w": data["width"][i], "h": data["height"][i],
        })
    return words


def forge_money(text):
    value = int(text.replace(",", ""))
    factor = random.choice([2, 3, 5, 10])
    new_value = value * factor
    return f"{new_value:,}"


def forge_date(text):
    sep = "-" if "-" in text else "."
    dt = datetime.strptime(text, f"%Y{sep}%m{sep}%d")
    shift_days = random.choice([-60, -30, -14, 14, 30, 60])
    new_dt = dt + timedelta(days=shift_days)
    return new_dt.strftime(f"%Y{sep}%m{sep}%d")


def pick_regions(words):
    candidates = []
    for w in words:
        if MONEY_RE.match(w["text"]):
            candidates.append((w, "money"))
        elif DATE_RE.match(w["text"]):
            candidates.append((w, "date"))
    random.shuffle(candidates)
    regions = []
    for w, kind in candidates[:MAX_REGIONS_PER_IMAGE]:
        forged_value = forge_money(w["text"]) if kind == "money" else forge_date(w["text"])
        if forged_value == w["text"]:
            continue
        regions.append({
            "original_value": w["text"],
            "forged_value": forged_value,
            "bbox": {"x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"]},
        })
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "dataset" / "original" / "real_docs"))
    parser.add_argument("--limit", type=int, default=20, help="max source images to scan")
    parser.add_argument("--max-written", type=int, default=None, help="stop once this many manifests are written")
    args = parser.parse_args()

    ensure_dirs()
    source_images = find_source_images(args.source, limit=args.limit)
    print(f"Scanning {len(source_images)} source images from {args.source}")

    written = 0
    skipped_no_field = 0
    for idx, src_path in enumerate(source_images, start=1):
        try:
            image = Image.open(src_path)
            words = ocr_words(image)
            regions = pick_regions(words)
            if not regions:
                skipped_no_field += 1
                continue

            base = f"auto{idx:05d}"
            dest_image = INPUT_DIR / f"{base}{src_path.suffix.lower()}"
            shutil.copy2(src_path, dest_image)

            manifest = {
                "difficulty": 4,
                "source_original_name": src_path.name,
                "forged_regions": regions,
            }
            manifest_path = INPUT_DIR / f"{base}.regions.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            written += 1
            print(f"  [{idx}/{len(source_images)}] {src_path.name} -> {base} ({len(regions)} region(s))")
            if args.max_written and written >= args.max_written:
                print(f"Reached --max-written={args.max_written}, stopping early.")
                break
        except Exception as exc:  # noqa: BLE001
            print(f"  [{idx}/{len(source_images)}] FAILED {src_path.name}: {exc}")

    print(f"Done. Written: {written} | Skipped (no money/date field found): {skipped_no_field}")


if __name__ == "__main__":
    main()
