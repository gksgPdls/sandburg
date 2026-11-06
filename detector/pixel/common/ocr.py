"""
글자 칸 찾기 OCR (v1·v2 공통, 2026-11-06 detector/v1/locate.py에서 분리).

  ocr_candidates(image)              tesseract, 금액·날짜 모양 칸만 (v1 자동 탐색 기본값)
  rapid_candidates(image, only=...)  RapidOCR, 모든 글자 줄 (기울고 흐린 사진에서 위조 칸 후보 포함률 47% -> 90%)
"""
import re
from pathlib import Path

from PIL import Image

MONEY_RE = re.compile(r"^\d{1,3}([,.]\d{3})+$")   # 140,997 / 160.000 (위조본에서 쉼표가 마침표로 찍히는 경우 포함)
DATE_RE = re.compile(r"^\d{4}[-.]\d{2}[-.]\d{2}$")
MIN_CONF = 0                                      # 위조 글자는 OCR 신뢰도가 낮게 나올 수 있어 패턴만으로 거름


def _tesseract():
    import shutil
    import pytesseract
    if not shutil.which("tesseract"):
        for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):  # 맥 홈브루 기본 위치
            if Path(p).exists():
                pytesseract.pytesseract.tesseract_cmd = p
                break
    return pytesseract


def ocr_candidates(image, relaxed=False):
    """OCR로 금액·날짜 모양 칸을 찾는다. relaxed=True면 숫자 3개 이상인 깨진 글자도 포함
    (위조 칸 포함률 47%→71%로 오르지만 후보가 문서당 6칸→21칸으로 늘어 문서 판정은 오히려 나빠짐 — 기본 False)."""
    pytesseract = _tesseract()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    d = pytesseract.image_to_data(img.convert("RGB"), output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip().strip("원")
        try:
            conf = float(d["conf"][i])
        except ValueError:
            conf = -1
        if not t or conf < MIN_CONF:
            continue
        kind = "money" if MONEY_RE.match(t) else ("date" if DATE_RE.match(t) else None)
        if kind is None and relaxed:
            # 위조 글자는 OCR이 '107.' '\\®_700.000'처럼 깨져 읽는 경우가 많아서 (2026-10-16 실측, 위조 칸 OCR 인식률
            # 원본 73% vs 위조본 47%), 숫자가 3개 이상이고 글자 절반 이상이 숫자면 후보로 포함
            digits = sum(ch.isdigit() for ch in t)
            if digits >= 3 and digits >= 0.5 * len(t):
                kind = "number"
        if kind:
            out.append({"text": t, "kind": kind, "conf": conf,
                        "bbox": (d["left"][i], d["top"][i], d["width"][i], d["height"][i])})
    return out


_RAPID = None


def rapid_candidates(image, only="all"):
    """RapidOCR로 문서의 글자 줄을 전부 찾는다 (2026-10-30). 글자를 잘못 읽어도 위치는 잡히므로
    tesseract 금액·날짜 패턴 방식보다 위조 칸을 덜 놓친다.
    only: "all"(모든 글자 칸), "digits"(숫자가 들어 있는 칸), "money"(금액·날짜 모양)"""
    global _RAPID
    import numpy as np
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    res, _ = _RAPID(arr)
    out = []
    for quad, text, conf in res or []:
        t = text.strip().replace(" ", "")
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs)) - x, int(max(ys)) - y
        if w < 6 or h < 6:
            continue
        nd = sum(ch.isdigit() for ch in t)
        tt = t.strip("원")
        kind = ("money" if MONEY_RE.match(tt) else "date" if DATE_RE.match(tt) else
                "number" if nd >= 1 else "text")
        if only == "digits" and nd == 0:
            continue
        if only == "money" and kind not in ("money", "date"):
            continue
        out.append({"text": t, "kind": kind, "conf": float(conf) * 100, "bbox": (x, y, w, h)})
    return out


