"""
의심 위치 자동 탐색 — 좌표 없이 문서 이미지만 넣으면, OCR로 금액·날짜 칸을 찾아 칸마다 탐지기를 돌린다.

    from detector.locate import locate
    r = locate("문서.jpg")
    r["suspicious"]        # 문서 단위 판정
    r["candidates"][0]     # 가장 의심스러운 칸 (희귀도 순 정렬)

흐름: OCR(tesseract) → 금액/날짜 패턴 칸만 후보 → 칸마다 detect() → 희귀도(정상 문서 중 이만큼 이상한 비율)로 정렬
문서 판정: 가장 이상한 칸의 희귀도가 config.json locate.doc_max_rarity 미만이면 위조 의심.
  (칸이 많을수록 우연히 걸리는 칸도 늘어나서, 칸 하나 기준(투표)보다 엄격한 기준을 따로 둠)
"""
import re
from pathlib import Path

from PIL import Image

from .core import detect, load_config

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


def locate(image, config=None, candidates=None, relaxed=False, ocr="tesseract", only="all"):
    config = config or load_config()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("RGB")
    if candidates is not None:
        cands = candidates
    elif ocr == "rapid":
        cands = rapid_candidates(img, only=only)
    else:
        cands = ocr_candidates(img, relaxed=relaxed)
    results = []
    for c in cands:
        try:
            r = detect(img, bbox=c["bbox"], config=config)
        except Exception:  # noqa: BLE001  (가장자리 등 측정 불가 칸은 건너뜀)
            continue
        results.append({**c, "rarity": r["verdict"]["rarity"], "votes": r["verdict"]["votes"],
                        "flagged_methods": [m["name"] for m in r["methods"].values() if m.get("flagged")],
                        "detail": r})
    results.sort(key=lambda x: (x["rarity"] if x["rarity"] is not None else 1.0))
    thr = config.get("locate", {}).get("doc_max_rarity", 0.01)
    top = results[0] if results else None
    suspicious = bool(top and top["rarity"] is not None and top["rarity"] < thr)
    return {"suspicious": suspicious, "n_candidates": len(results), "doc_max_rarity": thr,
            "summary": (f"후보 {len(results)}칸 중 가장 이상한 칸 '{top['text']}' 희귀도 {top['rarity']:.4f} "
                        f"(기준 {thr}) → {'위조 의심' if suspicious else '정상 판정'}") if top else "금액·날짜 칸을 찾지 못함",
            "candidates": results}
