#!/usr/bin/env python3
"""EDMS 를 거친 것 같은 변형(미세 회전 + 축소 + JPEG 재압축)을 입힌 샘플을 samples/edms/ 에 등록한다.

회의: 보험사 EDMS 를 거치면 압축·재생성된다. 실제로 엔진에 들어오는 문서는 이런 상태다.
방법론 케이스 11(재압축·미세회전 세탁). copy-move·글리프 단서가 압축 후에도 버티는지 본다.

**아직 데이터셋에 쓰지 않은 원본만 쓴다.** 기존 샘플을 변형하면 같은 문서가 두 번 들어가
평가에서 중복으로 세어지기 때문이다.
1. 새 원본에 자동 조작을 한다 (copy-move: 금액 칸 복제·숫자 조합 / 콘텐츠 로직: 합계 불일치·자릿수 /
   글리프: 날짜 치환). 변형 전 버전은 따로 등록하지 않는다
2. 원본·조작본·마스크에 같은 변형을 적용한다. 마스크는 0/255 로 다시 맞춘다
3. 정상 대조군도 새 원본으로 만들어, 압축·회전만으로 위조라고 오판하는지(오탐) 본다

사용: make_edms.py --per-axis 5 --normal 5 [--seed 0]
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np

import auto_receipt as AR
from labels import ROOT, load_all
from register_sample import register

POOL = ROOT / "_work" / "pool"
TMP = ROOT / "_work" / "tmp"
OPS_BY_AXIS = {
    "copy_move": (["copymove", "digit_combo"], ["DOC210"]),
    "content_logic": (["sum_break", "digit"], ["DOC210"]),
    "glyph": (["date_shift"], ["DOC210", "DOC205"]),
}
NORMAL_MIX = ["DOC210", "DOC210", "DOC210", "DOC205", "DOC203"]
DOC_TYPE = {"DOC210": "진료비 영수증", "DOC205": "진료비 영수증 (퇴원·세부내역 계열)", "DOC203": "진단서·소견서"}


def transform(img, angle, scale, nearest=False):
    """가운데를 축으로 angle 도 돌리고 scale 배로 줄인다 (회전으로 생긴 모서리는 가장자리 색으로)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    if scale != 1.0:
        out = cv2.resize(out, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    return out


def fresh_files(dtbs, used, rng):
    with open(POOL / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dtb"] in dtbs and r["pool_id"] not in used]
    rng.shuffle(rows)
    return [(POOL / r["file"], r["pool_id"]) for r in rows]


def main():
    p = argparse.ArgumentParser(description="EDMS 변형 샘플 (새 원본으로)")
    p.add_argument("--per-axis", type=int, default=5)
    p.add_argument("--normal", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--author", default="김동현")
    a = p.parse_args()

    used = {r.get("source", "") for r in load_all()}
    rng = random.Random(a.seed)
    TMP.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    made = []

    def save(axis, original, tampered, mask, desc, det, doc_type, notes):
        angle = rng.choice([-1, 1]) * round(rng.uniform(0.5, 2.0), 2)
        scale = rng.choice([1.0, 0.85])
        quality = rng.choice([70, 80, 90])
        params = f"회전 {angle:+.2f}도, 축소 {scale:.2f}배, JPEG 품질 {quality}"
        paths = {"original": TMP / f"eo_{pid}.jpg"}
        cv2.imwrite(str(paths["original"]), transform(original, angle, scale), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if tampered is not None:
            m = transform(mask, angle, scale, nearest=True)
            m = np.where(m > 127, 255, 0).astype(np.uint8)
            if not m.any():
                return None
            paths["tampered"], paths["mask"] = TMP / f"et_{pid}.jpg", TMP / f"em_{pid}.png"
            cv2.imwrite(str(paths["tampered"]), transform(tampered, angle, scale), [cv2.IMWRITE_JPEG_QUALITY, quality])
            cv2.imwrite(str(paths["mask"]), m)
        sid = register(method="edms", axis=axis, original=paths["original"], tampered=paths.get("tampered"),
                       mask=paths.get("mask"), description=desc.format(params=params), detection=det,
                       doc_type=doc_type, difficulty="어려움" if tampered is not None else "",
                       external_ai="N", author=a.author, source=notes["source"],
                       notes=f"make_edms.py seed={a.seed}. {notes['op']} angle={angle} scale={scale} quality={quality}")
        for f in paths.values():
            f.unlink()
        return sid

    for axis, (ops, dtbs) in OPS_BY_AXIS.items():
        got, i = 0, 0
        for f, pool_id in fresh_files(dtbs, used, rng):
            if got == a.per_axis:
                break
            op = ops[i % len(ops)]
            res = AR.tamper_file(op, f, rng)
            if not res:
                continue
            tampered, mask, info, _ = res
            sid = save(axis, cv2.imread(str(f)), tampered, mask,
                       f"EDMS 변형 ({{params}}). 변형 전 조작: {info['description']}",
                       f"변형 후에도 이 축이 잡아야 함. 회전·재압축으로 copy-move 블록 매칭과 글리프 획 비교가 약해짐. "
                       f"콘텐츠 로직은 OCR 이 읽으면 그대로. 변형 전 탐지: {info['detection']}",
                       DOC_TYPE.get(f.parent.name, ""), {"source": pool_id, "op": f"op={op}"})
            if sid:
                used.add(pool_id)
                made.append(sid)
                got += 1
                i += 1
        print(f"{axis}: {got}개")

    got = 0
    for dtb in NORMAL_MIX[:a.normal] + NORMAL_MIX * 0:
        for f, pool_id in fresh_files([dtb], used, rng):
            sid = save("none", cv2.imread(str(f)), None, None,
                       "정상 문서에 EDMS 변형 ({params}). 조작 없음",
                       "정상 문서. 압축·회전만으로 위조라고 판정하면 오탐", DOC_TYPE.get(dtb, ""),
                       {"source": pool_id, "op": "normal"})
            used.add(pool_id)
            made.append(sid)
            got += 1
            break
    print(f"normal: {got}개")
    print(f"EDMS 변형 {len(made)}개 → samples/edms/  {' '.join(made)}")


if __name__ == "__main__":
    main()
