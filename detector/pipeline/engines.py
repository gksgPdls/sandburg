"""두 탐지 엔진을 같은 결과 형식으로 감싼다. 엔진 코드는 고치지 않고 불러만 쓴다.

결과 형식 (엔진 공통)
    {"engine": "logic" | "pixel", "level": "의심" | "확인필요" | "정상" | "보류",
     "strong": bool,          # 정상 서류에선 나올 수 없는 근거인가 (로직 확정·강함 / 픽셀 2% 기준 넘음)
     "findings": [...],       # 엔진 원래 근거
     "boxes": ["x,y,w,h", ...], "score": float | None, "seconds": float, "note": str}

- 로직 = logic_detector 규칙 (A 산식 · B 수가 · C 상병 · D 날짜·번호 · F 주목 칸). 판독은 VLM 이 한 것(Reading)을 받는다
- 픽셀 = detector v2 A11 '다시 그린 칸' (원본·칸 위치 없이 문서 전체). 칸 찾기만 로컬 RapidOCR (읽기가 아니라 글자 상자 위치)
  문서 점수 = 가장 의심스러운 칸 점수. 기준값은 정상 서류로 정한 detector/pipeline/pixel_thresholds.json (calibrate_pixel.py)
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
THRESHOLDS = Path(__file__).resolve().parent / "pixel_thresholds.json"


# ── 로직 ─────────────────────────────────────────────────────────────────────

def logic(reading):
    from detector.logic import focus
    from detector.logic.rules import GRADES, check
    t0 = time.time()
    found = sorted(check(reading) + focus.check(reading), key=lambda f: GRADES.index(f["grade"]))
    strong = [f for f in found if f["grade"] in ("확정", "강함")]
    level = "의심" if strong else "확인필요" if found else "정상"
    return {"engine": "logic", "level": level, "strong": bool(strong), "findings": found,
            "fields": sorted({x for f in (strong or found) for x in f.get("fields", [])}),
            "boxes": [], "score": None, "seconds": round(time.time() - t0, 2), "note": ""}


# ── 픽셀 ─────────────────────────────────────────────────────────────────────

def pixel_score(image):
    """사진 → (문서 점수, 칸 결과 [{bbox, score, why}] 점수 높은 순). 칸을 못 찾으면 (None, [])."""
    from PIL import Image, ImageOps

    from detector.pixel.common.cells import group_cells, read_cells
    from detector.pixel.v2.checks import a11_redrawn
    img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
    cells = read_cells(img)
    if not cells:
        return None, []
    res = sorted(a11_redrawn(img, cells, group_cells(cells)), key=lambda c: -c["score"])
    return (res[0]["score"] if res else None), res


def load_thresholds():
    if not THRESHOLDS.exists():
        return None
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


def pixel(image, thr=None):
    thr = thr if thr is not None else load_thresholds()
    t0 = time.time()
    try:
        score, cells = pixel_score(image)
    except Exception as e:  # noqa: BLE001
        return {"engine": "pixel", "level": "보류", "strong": False, "findings": [], "boxes": [], "score": None,
                "seconds": round(time.time() - t0, 2), "note": f"오류 {type(e).__name__}: {e}"[:200]}
    out = {"engine": "pixel", "score": None if score is None else round(float(score), 4), "seconds": round(time.time() - t0, 2),
           "findings": [{"bbox": list(c["bbox"]), "score": round(float(c["score"]), 3), "why": c["why"]} for c in cells[:5]],
           "boxes": [], "strong": False, "note": ""}
    if score is None:
        return {**out, "level": "보류", "note": "글자 칸을 못 찾음"}
    if not thr:
        return {**out, "level": "보류", "note": "기준값 없음 (detector/pipeline/calibrate_pixel.py 먼저)"}
    hi, lo = thr["hi"], thr["lo"]
    out["level"] = "의심" if score > hi else "확인필요" if score > lo else "정상"
    out["strong"] = score > hi
    out["boxes"] = [",".join(str(int(v)) for v in c["bbox"]) for c in cells if c["score"] > lo][:3]
    return out
