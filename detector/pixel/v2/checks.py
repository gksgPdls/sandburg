"""
A 검사 모음 (픽셀·글자 쪽, AI 없이). 검사 하나 = 함수 하나.

    def check(image, cells, groups) -> [{"bbox": (x, y, w, h), "score": float, "why": str}, ...]
      image : PIL RGB, cells/groups : detector.cells 결과
      score : 클수록 의심 (검사마다 단위가 달라도 됨 — 평가에서 검사별로 기준값을 따로 정함)

CHECKS 에 등록하면 detector/pixel/scripts/v2/evaluate.py 가 평가한다.
A0 = 지금 탐지기(노이즈·FFT·픽셀통계)를 금액·날짜 칸마다 돌린 것. 새 검사의 비교 기준.
"""
import numpy as np

from ..v1.core import detect
from .digit_shape import digit_scores
from .copymove import copymove_scores
from .smooth import smooth_scores
from .texture import texture_scores
from .whitebox import whitebox_scores


def a0_existing(image, cells, groups):
    out = []
    for c in cells:
        if c["kind"] not in ("money", "date"):
            continue
        try:
            r = detect(image, bbox=c["bbox"])
        except Exception:  # noqa: BLE001
            continue
        rar = r["verdict"].get("rarity")
        out.append({"bbox": c["bbox"], "score": (1.0 - rar) if rar is not None else 0.0,
                    "why": f"투표 {r['verdict']['votes']}"})
    return out


def _gray(image):
    return np.asarray(image.convert("L"))


def a9_texture(image, cells, groups):
    """칸 질감: 같은 문서 금액·숫자 칸보다 글자 속 노이즈·경계 거칠기가 적으면(다시 그린 칸) 의심"""
    by = {c["id"]: c for c in cells}
    return [{"bbox": by[cid]["bbox"], "score": d["score"],
             "why": f"매끈 z inknoise {d['inknoise']:.1f} lapvar {d['lapvar']:.1f}"}
            for cid, d in texture_scores(_gray(image), cells).items()]


def a10_same_digit(image, cells, groups):
    """같은 숫자끼리(0은 0, 9는 9) 경계 선명도(lap) 비교: 같은 숫자보다 매끈하면 의심.
    모양 차이(shape)는 v2 24장에서 구분력이 없어(AUC 0.50) 점수에서 뺌, why에만 표시"""
    by = {c["id"]: c for c in cells}
    return [{"bbox": by[cid]["bbox"], "score": d["lap"],
             "why": f"숫자 {d['worst']} 모양 z {d['shape']:.1f}, 선명도 z {d['lap']:.1f}"}
            for cid, d in digit_scores(_gray(image), cells).items()]


def a11_redrawn(image, cells, groups):
    """다시 그린 칸 종합: 글자 속 노이즈·경계 선명도·글자 색 흔들림·같은 숫자 선명도가 주변 칸보다 낮음"""
    import cv2
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    by = {c["id"]: c for c in cells}
    return [{"bbox": by[cid]["bbox"], "score": d["score"],
             "why": " ".join(f"{k} {v:.1f}" for k, v in d.items() if k not in ("score", "n")) + f" (비교칸 {d['n']})",
             "n": d["n"], "parts": {k: round(v, 2) for k, v in d.items() if k not in ("score", "n")}}
            for cid, d in smooth_scores(bgr, cells).items()]


def a5_whitebox(image, cells, groups):
    """흰 박스 덧칠: 칸 안 종이가 바로 바깥보다 매끈하거나 밝기가 다름"""
    by = {c["id"]: c for c in cells}
    return [{"bbox": by[cid]["bbox"], "score": d["score"], "why": f"잔결 log비 {d['noise']:.2f}, 밝기차 {d['bright']:.1f}"}
            for cid, d in whitebox_scores(_gray(image), cells).items()]


def a7_copymove(image, cells, groups):
    """복사-붙여넣기: 같은 문서 안 다른 칸과 노이즈까지 똑같은 칸 (디지털 문서는 제외)"""
    by = {c["id"]: c for c in cells}
    return [{"bbox": by[cid]["bbox"], "score": d["score"],
             "why": f"'{by[d['with']]['text']}' 칸과 노이즈까지 같음 (차이 {1 - d['score']:.2f}, 문서 안 거의 같은 쌍 {d['pairs']}개)"}
            for cid, d in copymove_scores(_gray(image), cells).items()]


CHECKS = {
    "A0_기존탐지기": a0_existing,
    "A9_질감": a9_texture,
    "A10_같은숫자": a10_same_digit,
    "A11_다시그린칸": a11_redrawn,
    "A5_흰박스": a5_whitebox,
    "A7_복사붙여넣기": a7_copymove,
}
