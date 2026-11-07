"""
v2 입구 (2026-11-06): 사진 한 장 → 칸을 스스로 찾아(로컬 OCR) 3단계 판정 + 의심 칸 + 이유. 사진은 밖으로 안 나감.

    from detector.pixel.v2.locate import locate
    r = locate("문서.jpg")            # ocr="rapid" (기본) | "tesseract"
    r["verdict"]    -> "의심" | "확인필요" | "정상"
    r["score"]      -> 문서 점수 (A11 가장 높은 칸)
    r["candidates"] -> [{"bbox", "text", "score", "check", "level", "why"}, ...]  점수 높은 순 (A7 복사 칸 먼저)
    r["summary"], r["n_candidates"], r["thresholds"]

판정 (detector/pixel/v2/config.json, 튜닝용 정상 576장에서 정함):
  A7 복사-붙여넣기 점수 > A7_위조의심 (정상 상위 0.5%)  → 의심
  A11 다시 그린 칸 점수 > A11_위조의심 (정상 상위 2%)  → 의심
  A11 점수 > A11_확인필요 (정상 상위 10%)               → 확인필요
  그 밖                                                 → 정상
v1 의 locate() 와 달리 칸이 많아도 오탐이 늘지 않음: 칸마다 같은 문서의 비슷한 칸과 비교한 z 점수라서 (v1 rapid: 정상 100% 의심 → v2: 10.5%).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .copymove import copymove_scores
from .smooth import smooth_scores

CFG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
PW = {"lapvar": "테두리가 주변 칸보다 매끈", "inknoise": "획 속이 주변 칸보다 깨끗", "inkchroma": "글자 색이 주변 칸보다 균일",
      "digit_lap": "같은 숫자보다 테두리가 매끈"}


def _cells(img_path, ocr):
    if ocr == "rapid":
        from ..common.cells import read_cells
        return read_cells(str(img_path))
    from ..common.ocr import ocr_candidates
    out = []
    for i, c in enumerate(ocr_candidates(Image.open(img_path))):
        out.append({"id": f"c{i}", "text": c.get("text", ""), "kind": c.get("kind"), "bbox": tuple(int(v) for v in c["bbox"])})
    return out


def locate(img, ocr="rapid", cells=None):
    """img: 경로 또는 PIL 이미지. cells: 이미 읽은 칸 목록([{id,text,kind,bbox}])이 있으면 OCR을 건너뜀."""
    import cv2
    pil = img if isinstance(img, Image.Image) else Image.open(img)
    if cells is None:
        if not isinstance(img, (str, Path)):
            raise ValueError("cells 없이 PIL 이미지만 주면 OCR 경로가 필요합니다: 파일 경로를 주세요")
        cells = _cells(img, ocr)
    cells = [dict(c, bbox=tuple(c["bbox"])) for c in cells]
    by = {c["id"]: c for c in cells}
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hi, lo, cth = CFG["A11_위조의심"], CFG["A11_확인필요"], CFG["A7_위조의심"]
    cand = []
    for cid, d in smooth_scores(bgr, cells).items():
        parts = {k: v for k, v in d.items() if k in PW}
        top = max(parts, key=parts.get) if parts else None
        lv = "의심" if d["score"] > hi else "확인필요" if d["score"] > lo else "정상"
        cand.append({"bbox": list(by[cid]["bbox"]), "text": by[cid]["text"], "score": round(d["score"], 3), "check": "A11_다시그린칸",
                     "level": lv, "why": f"{PW[top]} ({parts[top]:.1f})" if top else ""})
    cm = []
    for cid, d in copymove_scores(gray, cells).items():
        if d["score"] > cth:
            cm.append({"bbox": list(by[cid]["bbox"]), "text": by[cid]["text"], "score": round(d["score"], 3), "check": "A7_복사붙여넣기",
                       "level": "의심", "why": f"'{by[d['with']]['text']}' 칸과 노이즈까지 같음 (복사-붙여넣기)"})
    cand.sort(key=lambda c: -c["score"])
    cm.sort(key=lambda c: -c["score"])
    score = cand[0]["score"] if cand else None
    if cm:
        verdict = "의심"
    elif score is not None and score > hi:
        verdict = "의심"
    elif score is not None and score > lo:
        verdict = "확인필요"
    else:
        verdict = "정상"
    flagged = cm + [c for c in cand if c["level"] != "정상"]
    summary = ("; ".join(f"'{c['text']}' {c['why']}" for c in flagged[:3]) if flagged
               else (f"이상한 칸 없음 (가장 높은 칸 {score:.2f} ≤ {lo:.2f})" if score is not None else "비교할 금액 칸이 부족함"))
    return {"verdict": verdict, "suspicious": verdict != "정상", "score": score, "candidates": cm + cand,
            "n_candidates": len(cand), "summary": summary, "thresholds": {"A11_위조의심": hi, "A11_확인필요": lo, "A7_위조의심": cth}}
