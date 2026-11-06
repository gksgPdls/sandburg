"""
판독기용 OCR 두 개 + 글자 정리 도우미.

  vision(image)  맥 내장 Apple Vision (ocrmac). 한글 칸 이름·숫자 모두. 로컬, 인터넷 전송 없음
  rapid(image)   RapidOCR. 한글은 못 읽고 숫자만 믿을 만함 → 숫자 교차 확인용
토큰: {"t", "conf", "x0", "y0", "x1", "y1", "cx", "cy", "h"} — 좌표는 0~1 비율, 원점 왼쪽 위.
결과는 이미지 내용 해시로 outputs/detect_v2/ocr_cache/ 에 저장해 다시 읽는다.

글자 도우미(norm·jamo·similar·best_label·parse_number·split_words·deskew)는 real_data_analyzer/core/text.py에서 가져옴.
"""
import hashlib
import json
import re
from datetime import date
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "detect_v2" / "ocr_cache"
MAX_SIDE = 2600   # Vision 입력 긴 변 (속도). RapidOCR는 원본 크기

NUM_RE = re.compile(r"^[-−]?\d{1,3}([,.]\d{3})+$|^[-−]?\d+$")
DATE_RE = re.compile(r"(20\d{2})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
KCD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.?\d{1,3})?)\b")
MARKS = set("■☑✓✔√Vv■")


def _key(img, engine):
    buf = BytesIO()
    img.save(buf, "PNG")
    return f"{engine}_{hashlib.sha1(buf.getvalue()).hexdigest()[:16]}"


def _cached(img, engine, fn):
    path = CACHE / f"{_key(img, engine)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    toks = fn(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(toks, ensure_ascii=False), encoding="utf-8")
    return toks


def _tok(t, conf, x0, y0, x1, y1):
    return {"t": t, "conf": round(float(conf), 3), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "h": y1 - y0}


def _vision(img):
    from ocrmac import ocrmac
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    small = img.resize((int(w * s), int(h * s))) if s < 1 else img
    res = ocrmac.OCR(small, language_preference=["ko-KR", "en-US"], recognition_level="accurate").recognize()
    return [_tok(t, c, x, 1 - y - bh, x + bw, 1 - y) for t, c, (x, y, bw, bh) in res]   # Vision 원점 왼쪽 아래


_RAPID = None


def _rapid_engine():
    global _RAPID
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    return _RAPID


def _rapid(img):
    import cv2
    engine = _rapid_engine
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    if s < 1:   # 휴대폰 원본(4284×5712)은 줄여서 읽음 (속도). 좌표는 비율이라 그대로
        img = img.resize((int(w * s), int(h * s)))
        w, h = img.size
    res, _ = engine()(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    out = []
    for quad, text, conf in res or []:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        out.append(_tok(text, conf, min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h))
    return out


def vision(img):
    return _cached(img, "vision", _vision)


def rapid(img):
    return _cached(img, "rapid", _rapid)


# ── 글자 도우미 ────────────────────────────────────────────────────────────

def norm(s):
    return re.sub(r"[\s\[\]()<>·ㆍ•.,:;'\"|_\-~]", "", s)


_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(s):
    """한글을 자모로 풀어 씀 (OCR이 한 획만 틀린 경우도 비슷하게)."""
    out = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_CHO[code // 588] + _JUNG[(code % 588) // 28] + _JONG[code % 28].strip())
        else:
            out.append(ch)
    return "".join(out)


def similar(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if b in a and len(a) <= len(b) + 2:
        return 1.0
    return SequenceMatcher(None, jamo(a), jamo(b)).ratio()


def best_label(text, labels, threshold=0.7):
    best, score = None, threshold
    for key, aliases in labels.items():
        for al in aliases:
            s = similar(text, al)
            if s > score or (s == score and best is None):
                best, score = key, s
    return best


def parse_number(s):
    """'13,160' / '9.212' / '0' → int. 숫자가 아니면 None."""
    s = s.strip().replace(" ", "").replace("원", "")
    s = re.sub(r"^[①-⑪]|[①-⑪]$", "", s)
    if not NUM_RE.match(s):
        return None
    v = int(re.sub(r"\D", "", s))
    return -v if s[0] in "-−" else v


def find_dates(text):
    out = []
    for y, m, d in DATE_RE.findall(text):
        try:
            out.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    return out


def split_words(tokens):
    """띄어쓰기로 이어진 덩어리('5,283 12,327')를 단어별 토큰으로. 단어 위치는 글자 수 비율로 추정.
    gi/gn: 덩어리 안 몇 번째 단어 / 단어 수, grp_x1: 덩어리 오른쪽 끝."""
    out = []
    for t in tokens:
        text = re.sub(r"[①-⑪]", " ", t["t"])
        parts = text.split()
        if len(parts) <= 1:
            out.append({**t, "t": parts[0]} if parts and parts[0] != t["t"].strip() else t)
            continue
        total, pos, w = len(text), 0, t["x1"] - t["x0"]
        for gi, p in enumerate(parts):
            start = text.index(p, pos)
            pos = start + len(p)
            x0, x1 = t["x0"] + w * start / total, t["x0"] + w * pos / total
            out.append({**t, "t": p, "x0": x0, "x1": x1, "cx": (x0 + x1) / 2,
                        "grp_x1": t["x1"], "gi": gi, "gn": len(parts), "src": t["t"]})
    return out


def label_text(s):
    """라벨 덩어리에 붙어 읽힌 금액·원문자 제거. '2·3인실', '65세 이상'의 짧은 숫자는 라벨의 일부라 남긴다."""
    return re.sub(r"[-−]?\d{1,3}(?:[,.]\d{3})+|[-−]?\d{4,}|[①-⑪]", " ", s).strip()


def text_lines(tokens, tol):
    """토큰을 줄 단위로 묶음 (보정된 y 'ry' 기준, 앞 토큰과 tol 넘게 떨어지면 새 줄). 줄마다 왼쪽부터."""
    out = []
    for t in sorted(tokens, key=lambda t: t["ry"]):
        if out and t["ry"] - out[-1][-1]["ry"] <= tol:
            out[-1].append(t)
        else:
            out.append([t])
    return [sorted(ln, key=lambda t: t["cx"]) for ln in out]


def deskew(tokens, anchor_groups):
    """같은 가로줄에 있어야 하는 라벨들로 기울기를 추정해 토큰마다 보정된 y("ry")를 붙인다."""
    slopes = []
    for group in anchor_groups:
        pts = []
        for lab in group:
            cands = [t for t in tokens if similar(t["t"], lab) >= 0.8 and len(norm(t["t"])) <= len(norm(lab)) + 3]
            if cands:
                c = min(cands, key=lambda t: t["cy"])
                pts.append((c["cx"], c["cy"]))
        if len(pts) >= 2:
            pts.sort()
            dx = pts[-1][0] - pts[0][0]
            if dx > 0.15:
                slopes.append((pts[-1][1] - pts[0][1]) / dx)
    slope = sorted(slopes)[len(slopes) // 2] if slopes else 0.0
    if abs(slope) > 0.15:
        slope = 0.0
    for t in tokens:
        t["ry"] = t["cy"] - slope * t["cx"]
    return slope


def box_px(t, size):
    """비율 좌표 토큰 → 픽셀 (x, y, w, h)."""
    W, H = size
    return (int(t["x0"] * W), int(t["y0"] * H), int((t["x1"] - t["x0"]) * W), int((t["y1"] - t["y0"]) * H))


def checkbox_text(t, word):
    """토큰 글자에서 word 앞 체크 칸 상태. 빈 칸은 OCR이 '[ ]' 그대로 읽고, 체크된 칸은 '117', 'V]', '[L]', '1J'처럼
    다른 모양으로 읽는다. → True(체크) / False(빈 칸) / None(word가 토큰 맨 앞이라 칸을 못 봄: ■가 괄호를 덮은 경우가 많음)."""
    m = re.search(r"\s*".join(map(re.escape, word)), t["t"])
    if not m:
        return None
    tail = t["t"][:m.start()].rstrip()
    if not tail:
        return None
    tail = re.sub(r"\[[1lI|]$", "[]", tail[-4:].replace(" ", ""))   # 빈 칸의 ']'를 '1'로 읽는 경우 ('[ 1 야간')
    return not tail.endswith("[]")


def checkbox_dark(gray, t, word, size):
    """토큰 안 word 바로 왼쪽의 체크 칸('[■]')이 얼마나 칠해졌나 (0~1). word가 토큰에 없으면 None.
    OCR은 ■를 '[ ]'로 읽어 버리는 경우가 많아 이미지에서 직접 잰다."""
    text = t["t"]
    m = re.search(r"\s*".join(map(re.escape, word)), text)
    if not m:
        return None
    W, H = size
    x_word = t["x0"] + (t["x1"] - t["x0"]) * m.start() / max(len(text), 1)
    h = (t["y1"] - t["y0"]) * H
    x1 = int(x_word * W) - 2
    x0 = int(x1 - 1.6 * h)
    y0, y1 = int(t["y0"] * H + 0.2 * h), int(t["y1"] * H - 0.2 * h)
    patch = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if patch.size == 0:
        return None
    return float((patch < 110).mean())
