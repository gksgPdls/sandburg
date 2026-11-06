"""
의심 위치 자동 탐색 — 좌표 없이 문서 이미지만 넣으면, OCR로 금액·날짜 칸을 찾아 칸마다 탐지기를 돌린다.

    from detector.v1.locate import locate
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
from ..common.ocr import DATE_RE, MONEY_RE, ocr_candidates, rapid_candidates  # noqa: F401  (예전 import 경로 호환)

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
