"""로직 엔진: 서류 이미지 → 판독(Reading) → 공식 근거 규칙 검사. 설계: docs/로직엔진_설계.md

    from logic_detector.logic import inspect
    r = inspect("영수증.jpg")
    r["verdict"]     # 위변조 의심 / 확인 필요 / 정상
    r["findings"]    # [{rule, grade, title, boxes, expected, actual, source}]  주목 규칙 검사는 rule = "F:<id>"
    r["focus"]       # 보험금이 걸린 칸 [{id, name, what, boxes, why, sources}] — 판정이 아니라 '여기를 보라' (data/focus_rules.yaml)
"""
from PIL import Image, ImageOps

from . import focus, ocr
from .reading import Reading
from .rules import GRADES, check, verdict

__all__ = ["Reading", "check", "verdict", "inspect", "doc_type"]


def doc_type(img):
    """서식 제목으로 서류 종류: receipt(진료비 계산서·영수증) / diagnosis(진단서) / None."""
    text = " ".join(t["t"] for t in ocr.vision(img) if t["cy"] < 0.25).replace(" ", "")
    if "계산서" in text or "영수증" in text:
        return "receipt"
    if "진단서" in text or "진단" in text:
        return "diagnosis"
    return None


def inspect(src, kind=None):
    """이미지 한 장 검사. kind를 주지 않으면 서식 제목으로 판별."""
    from . import read_diagnosis, read_receipt
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    kind = kind or doc_type(img)
    if kind is None:
        return {"kind": None, "verdict": "확인 필요", "findings": [], "notes": ["서류 종류를 모름 (영수증·진단서 아님)"]}
    reading, notes = (read_receipt if kind == "receipt" else read_diagnosis).read(img)
    found = sorted(check(reading) + focus.check(reading), key=lambda f: GRADES.index(f["grade"]))
    return {"kind": kind, "verdict": verdict(found), "findings": found, "focus": focus.focus(reading), "notes": notes,
            "reading": {k: v for k, v in reading.fields.items()}}
