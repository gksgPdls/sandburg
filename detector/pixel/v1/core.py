"""
탐지기 본체: detect(이미지, 의심영역) → 탐지법별 측정값·판정 + 종합 판정.

    from detector.pixel import detect          # (= detector.v1.core.detect)
    result = detect("의심문서.jpg", bbox=(893, 224, 122, 26))   # x, y, w, h
    result["verdict"]["suspicious"]   # True = 위조 의심

판정 규칙(config.json):
    rule.low  : score < low  이면 이상  (예: 노이즈가 주변보다 너무 적음)
    rule.high : score > high 이면 이상
    rule.min_inside : (선택) 안쪽 측정값이 이 값 이상일 때만 이상으로 인정
    measure   : (선택) 다른 이름의 측정값을 쓸 때 (기본: 항목 이름과 같은 측정값)
    verdict.min_votes : 켜진 탐지법 중 몇 개 이상 이상이면 '위조 의심'인지
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .methods import METHODS, Context

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
NORMAL_REFERENCE = Path(__file__).resolve().parent / "normal_reference.json"
_REF = None


def normal_reference():
    """정상 원본(튜닝용 절반)의 탐지법별 측정값 분포 — 희귀도 계산용. tune.py --write가 만든다."""
    global _REF
    if _REF is None:
        _REF = json.load(open(NORMAL_REFERENCE, encoding="utf-8")) if NORMAL_REFERENCE.exists() else {}
    return _REF


def rarity(key, score, direction):
    """정상 문서 중 이 값만큼(또는 더) 이상한 비율. 작을수록 이상. 0.03 = 정상 100장 중 3장 수준.
    low: 이 값 이하 비율 / high: 이 값 이상 비율 / two_sided: 양쪽 중 작은 쪽 × 2."""
    ref = normal_reference().get(key)
    if not ref or score is None or score != score:
        return None
    arr = np.asarray(ref)
    n = len(arr)
    lo = (np.searchsorted(arr, score, side="right") + 1) / (n + 1)
    hi = (n - np.searchsorted(arr, score, side="left") + 1) / (n + 1)
    p = {"low": lo, "high": hi}.get(direction, min(1.0, 2 * min(lo, hi)))
    return float(p)


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def bbox_to_mask(shape, bbox, padding=3):
    """(x, y, w, h) + 여백 → 사각형 마스크와 (x0, y0, x1, y1)."""
    x, y, w, h = bbox
    x0, y0 = max(x - padding, 0), max(y - padding, 0)
    x1, y1 = min(x + w + padding, shape[1]), min(y + h + padding, shape[0])
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1, y1)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_rule(score, inside, rule):
    if score is None or not np.isfinite(score) and score != float("inf"):
        return False
    low, high = rule.get("low"), rule.get("high")
    flagged = (low is not None and score < low) or (high is not None and score > high)
    if flagged and rule.get("min_inside") is not None:
        flagged = inside > rule["min_inside"]
    return bool(flagged)


def measure(image, mask=None, bbox=None, padding=3, methods=None, config=None):
    """판정 없이 탐지법별 측정값만 계산 (튜닝·평가용). methods=None이면 전부.
    측정 방식 설정(비교 패치 개수 등)은 config(기본 config.json)의 각 탐지법 comparison을 따름."""
    config = config or load_config()
    rgb = image if isinstance(image, Image.Image) else Image.open(image)
    rgb = rgb.convert("RGB")
    shape = (rgb.size[1], rgb.size[0])
    if mask is None:
        mask, box = bbox_to_mask(shape, bbox, padding)
    else:
        mask = np.asarray(mask.convert("L")) if isinstance(mask, Image.Image) else np.asarray(mask)
        box = mask_to_bbox(mask)
        if box is None:
            raise ValueError("마스크가 비어 있음")
    ctx = Context(rgb, mask, box, params=config["methods"])
    out = {}
    for key in methods or METHODS:
        out[key] = METHODS[key](ctx)
    return out


def judge(measurements, config):
    """측정값 + 설정 → 탐지법별 판정과 종합 판정."""
    per_method, votes, used = {}, 0, 0
    for key, cfg in config["methods"].items():
        m = measurements.get(cfg.get("measure", key))
        entry = {"name": cfg["name"], "enabled": cfg["enabled"]}
        if m is not None:
            entry.update({k: round(v, 4) if isinstance(v, float) and np.isfinite(v) else v
                          for k, v in m.items()})
        if not cfg["enabled"]:
            entry["flagged"] = None
            entry["why_disabled"] = cfg.get("why_disabled", "")
        else:
            used += 1
            entry["flagged"] = apply_rule(m["score"], m["inside"], cfg["rule"]) if m else False
            entry["rarity"] = rarity(key, m["score"], cfg.get("direction")) if m else None
            votes += entry["flagged"]
        per_method[key] = entry
    need = config["verdict"]["min_votes"]
    rs = [m["rarity"] for m in per_method.values() if m.get("rarity") is not None]
    min_r = min(rs) if rs else None
    return {
        "methods": per_method,
        "verdict": {"suspicious": votes >= need, "votes": votes, "enabled_methods": used,
                    "min_votes": need,
                    "rarity": min_r,  # 가장 이상한 탐지법의 희귀도 (정상 문서 중 이만큼 이상한 비율, 작을수록 의심)
                    "summary": f"{used}개 탐지법 중 {votes}개에서 이상 신호 (기준: {need}개 이상) → "
                               + ("위조 의심" if votes >= need else "정상 판정")},
    }


def detect(image, bbox=None, mask=None, padding=3, config=None):
    """이미지 1장 + 의심영역(bbox=(x,y,w,h) 또는 mask) → 판정 결과 dict."""
    config = config or load_config()
    keys = sorted({c.get("measure", k) for k, c in config["methods"].items() if c["enabled"]})
    return judge(measure(image, mask=mask, bbox=bbox, padding=padding, methods=keys, config=config), config)
