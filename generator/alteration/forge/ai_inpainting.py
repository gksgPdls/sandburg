"""
Generate AI-inpainting forgery samples with a hybrid pipeline: the target
text is first rendered with a font matched to the surrounding style (so the
digits/characters are always exactly right), then a LOCAL Stable Diffusion
inpainting model (diffusers) does a low-strength pass over just that region
to blend/harmonize it with the surrounding print/photo noise. Fully
offline/free after the one-time model download — no API key, no billing,
no network calls at inference time.

Why not pure SD inpainting: SD1.5-class models cannot reliably spell exact
digits/text (see outputs/logs from v1 runs) — they draw text-like texture,
not the requested characters. Rendering the text ourselves guarantees
correctness; the SD pass only adds realistic blending noise on top.

For each image in dataset/original/bankbook/, look for a sibling manifest
named "{base}.regions.json" describing which text regions to forge:

    {
      "difficulty": 4,
      "forged_regions": [
        {
          "original_value": "1,000,000",
          "forged_value": "10,000,000",
          "bbox": {"x": 300, "y": 150, "w": 120, "h": 25}
        }
      ]
    }

For each region: sample the background/text color from the original bbox,
render forged_value into that box with a matched font, then run the SD
inpainting pipeline at low strength over a padded square crop (resized to
512x512) to harmonize texture, resize back, and composite into the
full-resolution image using the mask (so pixels outside the mask are never
touched). The composited image + full-resolution binary mask + JSON label
are written to dataset/forged/ai_inpainting, dataset/labels/masks, and
dataset/labels/meta respectively. Images with no manifest are skipped
(logged, not fatal).
"""
import json
import logging
import sys
import time
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import INPUT_DIR, TAMPERED_DIR, MASK_DIR, META_DIR, LOG_DIR, ensure_dirs

VERSION = "v2"  # v2 = ring-mask fix (SD never touches glyph pixels; v1 had ~58% wrong digits, see outputs/logs)
MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
OUTPUT_QUALITY = 95       # 저장 품질 (JPEG 경로에서만 적용, 메타에는 항상 기록)
MAX_RETRIES = 3           # 실패 시 재시도 횟수
RETRY_DELAY = 2           # 재시도 대기 시간(초)
PADDING = 3               # 마스크 여백(px)
CROP_CONTEXT_MARGIN = 80  # bbox 주변에 추가로 포함할 컨텍스트(px)
RING_MARGIN = 14          # 글자 bbox 바깥으로 SD가 블렌딩할 테두리 폭(px) — 이 안쪽(글자)은 SD가 절대 못 건드림
MODEL_RESOLUTION = 512    # SD 네이티브 해상도
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SD_BLEND_STRENGTH = 0.35  # 낮을수록 렌더링한 텍스트를 더 많이 보존, 높을수록 SD가 더 많이 새로 그림
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # 한국 문서 숫자체에 Arial보다 가까움
FONT_INDEX_REGULAR = 0
FONT_INDEX_BOLD = 6
BOLD_FG_RATIO_THRESHOLD = 0.22  # bbox 내 글자(어두운) 픽셀 비율이 이보다 크면 굵은 폰트로 렌더링

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"ai_inpainting_{VERSION}_{TIMESTAMP}_q{OUTPUT_QUALITY}.log"

logger = logging.getLogger("ai_inpainting")


def setup_logging():
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_hf_token():
    """Optional: only needed if the model requires accepting a license on the HF Hub."""
    from dotenv import load_dotenv
    import os

    load_dotenv(__import__("pathlib").Path(__file__).resolve().parents[2] / ".env")
    return os.environ.get("HF_API_TOKEN") or None


def load_pipeline(hf_token):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading %s on device=%s (first run downloads the model, ~5GB)...", MODEL, device)
    start = time.time()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        token=hf_token,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    logger.info("Model loaded in %.1fs", time.time() - start)
    return pipe


def find_inputs():
    from config import IMAGE_EXTENSIONS

    files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return files


def load_manifest(image_path):
    manifest_path = image_path.with_suffix("")
    manifest_path = manifest_path.parent / f"{manifest_path.name}.regions.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mask(image_size, regions):
    """Binary mask (white = forged region incl. padding + SD blend ring, black = untouched)."""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        bbox = region["bbox"]
        margin = PADDING + RING_MARGIN
        x0 = max(bbox["x"] - margin, 0)
        y0 = max(bbox["y"] - margin, 0)
        x1 = min(bbox["x"] + bbox["w"] + margin, image_size[0])
        y1 = min(bbox["y"] + bbox["h"] + margin, image_size[1])
        draw.rectangle([x0, y0, x1, y1], fill=255)
    return mask


def build_prompt(region):
    return (
        f"a printed financial document field showing the text \"{region['forged_value']}\", "
        "black sans-serif font, white background, sharp clean printed text, high detail"
    )


def sample_region_style(image, bbox):
    """Estimate (background_color, text_color, fg_pixel_ratio) from the pixels currently in bbox.
    fg_pixel_ratio approximates how "bold" the original glyphs are (more dark pixels = thicker strokes)."""
    crop = image.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])).convert("RGB")
    pixels = list(crop.getdata())
    bg_pixels = [p for p in pixels if sum(p) / 3 > 200]
    fg_pixels = [p for p in pixels if sum(p) / 3 < 140]

    def avg(pixel_list, default):
        if not pixel_list:
            return default
        n = len(pixel_list)
        return tuple(sum(c) // n for c in zip(*pixel_list))

    bg_color = avg(bg_pixels, (255, 255, 255))
    fg_color = avg(fg_pixels, (20, 20, 20))
    fg_ratio = len(fg_pixels) / len(pixels) if pixels else 0.0
    return bg_color, fg_color, fg_ratio


def render_text_region(crop, local_bbox, text, bg_color, fg_color, font_index=FONT_INDEX_REGULAR):
    """Fill local_bbox in `crop` with bg_color, then draw `text` in fg_color, font-fit to the box."""
    lx0, ly0, lx1, ly1 = local_bbox
    draw = ImageDraw.Draw(crop)
    draw.rectangle([lx0, ly0, lx1, ly1], fill=bg_color)

    box_w, box_h = lx1 - lx0, ly1 - ly0
    font_size = max(int(box_h * 0.85), 6)
    font = ImageFont.truetype(FONT_PATH, font_size, index=font_index)
    while font.getlength(text) > box_w * 0.95 and font_size > 6:
        font_size -= 1
        font = ImageFont.truetype(FONT_PATH, font_size, index=font_index)

    text_w = font.getlength(text)
    tx = lx0 + max((box_w - text_w) / 2, 0)
    ty = ly0 + max((box_h - font_size) / 2, 0)
    draw.text((tx, ty), text, fill=fg_color, font=font)
    return crop


def compute_crop_box(bbox, image_size):
    """Square crop box around bbox, padded with context margin, clamped to image bounds."""
    cx = bbox["x"] + bbox["w"] / 2
    cy = bbox["y"] + bbox["h"] / 2
    side = max(bbox["w"], bbox["h"]) + 2 * CROP_CONTEXT_MARGIN
    x0 = int(max(cx - side / 2, 0))
    y0 = int(max(cy - side / 2, 0))
    x1 = int(min(cx + side / 2, image_size[0]))
    y1 = int(min(cy + side / 2, image_size[1]))
    return x0, y0, x1, y1


def run_inpaint_region(pipe, image, region):
    """Render forged_value in-place with a matched font (text is NEVER passed
    through SD), then SD-blend only a thin ring around it so the seam looks
    natural. Return the full-size image with that region composited in.

    Dedicated inpainting checkpoints (like the one used here) regenerate
    whatever is under the mask from scratch, ignoring the input pixels there
    — so if the text glyphs themselves were inside the SD mask, the model
    would happily redraw them as different digits (verified empirically:
    varies by random seed, not by strength). The fix is to never let SD see
    the glyph area as "to be generated" at all: only a border ring around it
    is masked for SD, and the final composite force-pins the glyph area back
    to our exact font render regardless of what SD produced there.
    """
    prompt = build_prompt(region)
    bbox = region["bbox"]
    bg_color, fg_color, fg_ratio = sample_region_style(image, bbox)
    font_index = FONT_INDEX_BOLD if fg_ratio > BOLD_FG_RATIO_THRESHOLD else FONT_INDEX_REGULAR

    crop_box = compute_crop_box(bbox, image.size)
    crop = image.crop(crop_box).convert("RGB")

    # inner box = the exact glyph area (+small padding) — SD must never alter this
    ix0 = max(bbox["x"] - PADDING - crop_box[0], 0)
    iy0 = max(bbox["y"] - PADDING - crop_box[1], 0)
    ix1 = min(bbox["x"] + bbox["w"] + PADDING - crop_box[0], crop.size[0])
    iy1 = min(bbox["y"] + bbox["h"] + PADDING - crop_box[1], crop.size[1])

    # outer box = inner box dilated by RING_MARGIN — only this ring is handed to SD
    ox0 = max(ix0 - RING_MARGIN, 0)
    oy0 = max(iy0 - RING_MARGIN, 0)
    ox1 = min(ix1 + RING_MARGIN, crop.size[0])
    oy1 = min(iy1 + RING_MARGIN, crop.size[1])

    ring_mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(ring_mask)
    draw.rectangle([ox0, oy0, ox1, oy1], fill=255)
    draw.rectangle([ix0, iy0, ix1, iy1], fill=0)  # carve out the glyph area

    inner_mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(inner_mask).rectangle([ix0, iy0, ix1, iy1], fill=255)

    rendered_crop = render_text_region(
        crop.copy(), (ix0, iy0, ix1, iy1), region["forged_value"], bg_color, fg_color, font_index=font_index
    )

    crop_512 = rendered_crop.resize((MODEL_RESOLUTION, MODEL_RESOLUTION))
    ring_mask_512 = ring_mask.resize((MODEL_RESOLUTION, MODEL_RESOLUTION))

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            logger.debug(
                "Inpaint attempt %d/%d | prompt=%r | crop_box=%s | strength=%s | ring_margin=%d",
                attempt, MAX_RETRIES, prompt, crop_box, SD_BLEND_STRENGTH, RING_MARGIN,
            )
            result = pipe(
                prompt=prompt,
                image=crop_512,
                mask_image=ring_mask_512,
                strength=SD_BLEND_STRENGTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]
            elapsed_ms = (time.time() - start) * 1000
            logger.debug("Inpaint finished in %.1fms", elapsed_ms)

            result_crop = result.resize(crop.size)
            # start from the ORIGINAL crop, blend in SD's ring-only output,
            # then force-pin the glyph area back to our exact render —
            # guarantees text fidelity regardless of what SD did anywhere.
            composited_crop = Image.composite(result_crop, crop, ring_mask)
            composited_crop = Image.composite(rendered_crop, composited_crop, inner_mask)

            full = image.convert("RGB").copy()
            full.paste(composited_crop, (crop_box[0], crop_box[1]))
            return full, elapsed_ms
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.time() - start) * 1000
            last_error = exc
            logger.warning(
                "Inpaint attempt %d/%d failed after %.1fms: %s", attempt, MAX_RETRIES, elapsed_ms, exc
            )
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %ds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Local inpaint failed after {MAX_RETRIES} attempts: {last_error}")


def process_file(pipe, image_path):
    base = image_path.stem
    file_start = time.time()
    file_size = image_path.stat().st_size

    logger.info("--- Processing %s ---", image_path.name)
    logger.info("File size: %d bytes", file_size)

    manifest = load_manifest(image_path)
    if manifest is None:
        logger.warning("No manifest found for %s (expected %s.regions.json), skipping", base, base)
        return {"file": image_path.name, "status": "skipped", "reason": "no_manifest"}

    regions = manifest.get("forged_regions", [])
    if not regions:
        logger.warning("Manifest for %s has no forged_regions, skipping", base)
        return {"file": image_path.name, "status": "skipped", "reason": "empty_regions"}

    image = Image.open(image_path)
    logger.info("Image size: %dx%d px", image.width, image.height)
    logger.info("Regions to forge: %d", len(regions))

    try:
        forged_image = image.convert("RGB")
        total_inpaint_ms = 0.0
        for region in regions:
            forged_image, elapsed_ms = run_inpaint_region(pipe, forged_image, region)
            total_inpaint_ms += elapsed_ms

        forged_path = TAMPERED_DIR / f"{base}_ai_inpaint_{VERSION}.png"
        forged_image.save(forged_path, format="PNG", optimize=True)

        mask = build_mask(image.size, regions)
        mask_path = MASK_DIR / f"{base}_ai_inpaint_mask_{VERSION}.png"
        mask.save(mask_path, format="PNG")

        label = {
            "version": VERSION,
            "model": MODEL,
            "forgery_type": "ai_inpainting",
            "difficulty": manifest.get("difficulty", 4),
            "source": image_path.name,
            "timestamp": datetime.now().isoformat(),
            "params": {"quality": OUTPUT_QUALITY, "padding": PADDING},
            "forged_regions": regions,
        }
        meta_path = META_DIR / f"{base}_ai_inpaint_{VERSION}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        elapsed_ms = (time.time() - file_start) * 1000
        logger.info(
            "SUCCESS %s | inpaint=%.1fms total=%.1fms | forged=%s mask=%s meta=%s",
            base, total_inpaint_ms, elapsed_ms, forged_path, mask_path, meta_path,
        )
        return {
            "file": image_path.name,
            "status": "success",
            "elapsed_ms": elapsed_ms,
            "forged_path": str(forged_path),
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.time() - file_start) * 1000
        logger.error("FAILED %s after %.1fms: %s", base, elapsed_ms, exc, exc_info=True)
        return {"file": image_path.name, "status": "failed", "reason": str(exc), "elapsed_ms": elapsed_ms}


def main():
    ensure_dirs()
    setup_logging()

    logger.info("=== ai_inpainting run start ===")
    logger.info("Version: %s | Model: %s", VERSION, MODEL)
    logger.info(
        "Config: quality=%d max_retries=%d retry_delay=%ds padding=%dpx",
        OUTPUT_QUALITY, MAX_RETRIES, RETRY_DELAY, PADDING,
    )

    inputs = find_inputs()
    logger.info("Input files found: %d (in %s)", len(inputs), INPUT_DIR)

    if not inputs:
        logger.warning("No input images found in %s. Nothing to do.", INPUT_DIR)
        logger.info("=== ai_inpainting run end (no inputs) ===")
        return

    hf_token = load_hf_token()
    pipe = load_pipeline(hf_token)

    results = []
    batch_start = time.time()
    for image_path in inputs:
        results.append(process_file(pipe, image_path))

    total_elapsed = time.time() - batch_start
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    avg_ms = (
        sum(r["elapsed_ms"] for r in succeeded) / len(succeeded) if succeeded else 0.0
    )

    logger.info("=== Batch complete ===")
    logger.info("Total: %d | Success: %d | Failed: %d | Skipped: %d",
                len(results), len(succeeded), len(failed), len(skipped))
    logger.info("Total time: %.1fs | Avg per success: %.1fms", total_elapsed, avg_ms)

    if failed:
        logger.info("--- Failed files ---")
        for r in failed:
            logger.info("  %s: %s", r["file"], r["reason"])
    if skipped:
        logger.info("--- Skipped files ---")
        for r in skipped:
            logger.info("  %s: %s", r["file"], r["reason"])

    logger.info("Log file: %s", LOG_PATH)
    logger.info("=== ai_inpainting run end ===")

    latest_run_path = LOG_DIR / "latest_run.json"
    with open(latest_run_path, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "timestamp": TIMESTAMP, "quality": OUTPUT_QUALITY}, f)


if __name__ == "__main__":
    main()
