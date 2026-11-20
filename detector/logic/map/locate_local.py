"""VLM 이 읽은 값 → 사진 위 위치 (로컬 OCR 글자 위치로 찾기). 외부 호출 없음.

VLM 판독에는 값만 있고 위치가 없어, 같은 값을 로컬 OCR(Apple Vision + RapidOCR, 캐시됨) 글자들 중에서 찾아 상자를 돌려준다.
    boxes_for(image, fields, reading) → [[x, y, w, h], ...]
"""
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path.home() / "Desktop/SCRYN/3_탐지기/scryn-combine-web"))
import _compat  # noqa: E402,F401  (logic_detector 찾기)
from logic_detector.logic import ocr  # noqa: E402

_TOK = {}


def tokens(image):
    key = str(image)
    if key not in _TOK:
        img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
        W, H = img.size
        out = []
        for t in ocr.vision(img) + ocr.rapid(img):
            out.append({"t": t["t"], "n": ocr.norm(t["t"]).upper(),
                        "box": [int(t["x0"] * W), int(t["y0"] * H), int((t["x1"] - t["x0"]) * W), int((t["y1"] - t["y0"]) * H)]})
        _TOK[key] = out
    return _TOK[key]


def _val(v):
    return v.get("value") if isinstance(v, dict) and "value" in v else v


def targets(value):
    """읽은 값 → 찾을 글자들 (정규화)."""
    v = _val(value)
    if v is None or v == "" or v is False or v is True:
        return []
    if isinstance(v, (int, float)):
        return [str(int(v))] if abs(v) >= 100 else []          # 0 · 작은 수는 너무 흔해서 찾지 않음
    if isinstance(v, list):                                     # 진단 목록
        return [ocr.norm(str(_val(d.get("code")) or "")).upper() for d in v if isinstance(d, dict) and d.get("code")]
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return [s.replace("-", "")]
    n = ocr.norm(s).upper()
    return [n[:8], n[:6], n[:4]] if len(n) > 8 else ([n] if len(n) >= 3 else [])


def find(image, value):
    out = []
    tgs = targets(value)
    prefix_mode = len(tgs) == 3 and tgs[1] == tgs[0][:6]
    for tg in tgs:
        if prefix_mode and out:
            break
        toks = tokens(image)
        exact = [t for t in toks if t["n"] == tg or re.sub(r"\D", "", t["n"]) == tg and tg.isdigit()]
        part = [t for t in toks if tg in t["n"]]
        hit = exact or part
        if not hit and not tg.isdigit() and len(tg) >= 6:              # 긴 글: OCR 한두 글자 차이(레이저/레이져)는 같은 것으로
            import difflib
            hit = [t for t in toks if difflib.SequenceMatcher(None, tg, t["n"][:len(tg)]).ratio() >= 0.75]
        if hit:
            # 글자 덩어리가 값보다 훨씬 길면(줄 전체) 값 비율만큼 오른쪽을 잘라 씀
            t = min(hit, key=lambda t: len(t["n"]))
            x, y, w, h = t["box"]
            if len(t["n"]) > len(tg) * 1.6 and tg in t["n"]:
                i = t["n"].index(tg)
                x, w = x + int(w * i / len(t["n"])), int(w * len(tg) / len(t["n"])) + 4
            out.append([x - 3, y - 3, w + 6, h + 6])
    return out


def boxes_for(image, fields, reading):
    out = []
    for f in fields:
        if f in reading:
            for b in find(image, reading[f]):
                if b not in out:
                    out.append(b)
    return out
