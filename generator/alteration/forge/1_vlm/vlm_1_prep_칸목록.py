"""VLM 논리 맞추기 1단계: 사진 폴더 -> 사진마다 OCR 칸 목록 (VLM에게 사진과 같이 줄 것).

python generator/alteration/forge/1_vlm/vlm_1_prep_칸목록.py --src <사진 폴더> --name <이름> [--limit 10]
결과: outputs/고도화/_vlm/<이름>/
  <사진이름>_ocr.txt   VLM에게 줄 칸 목록 ([번호] "글자" (x, y, w, h))
  <사진이름>_ocr.json  같은 내용 (vlm_2_apply_답확인.py 가 씀)
  프롬프트.txt         VLM에게 줄 프롬프트 (오늘 날짜 채워짐)
  안내.txt             손으로 할 때 순서
VLM 답은 같은 폴더에 <사진이름>_vlm.json 으로 저장 -> vlm_2_apply_답확인.py"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # generator/alteration/forge
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
OUT = ROOT / "outputs" / "고도화" / "_vlm"


def prompt_text():
    return (HERE / "vlm_prompt.txt").read_text(encoding="utf-8").replace("{TODAY}", dt.date.today().isoformat())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", default="vlm")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import logic_list as LL
    LL.OCR = "rapid"
    src = Path(a.src).expanduser()
    src = src if src.is_absolute() else ROOT / src
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "프롬프트.txt").write_text(prompt_text(), encoding="utf-8")
    (out / "안내.txt").write_text(
        "VLM 계정으로 할 때: docs/VLM_답쓰기.md 순서대로 (VLM에게 이 폴더의 VLM 답을 써달라고 요청)\n\n"
        "손으로 할 때 (사진 한 장마다)\n"
        "1. VLM 채팅에 원본 사진 + <사진이름>_ocr.txt 내용 + 프롬프트.txt 내용을 넣는다\n"
        "2. 답(JSON)을 이 폴더에 <사진이름>_vlm.json 으로 저장한다\n"
        "3. 다 되면: python generator/alteration/forge/1_vlm/vlm_2_apply_답확인.py --name " + a.name + "\n"
        "주의: VLM도 외부 서비스. 가짜로 만든 문서만 쓴다 (실제 의료 문서는 넣지 않는다)\n",
        encoding="utf-8")
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if a.limit:
        files = files[:a.limit]
    for i, p in enumerate(files, 1):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cells = [c for c in LL.read_cells(gray) if any(ch.isdigit() for ch in c.get("raw", c["text"]))]
        cells.sort(key=lambda c: (c["y"] // 8, c["x"]))
        rows, lines = [], []
        for k, c in enumerate(cells, 1):
            raw = c.get("raw", c["text"]).replace('"', "'")
            rows.append({"id": k, "raw": raw, "text": c["text"], "amount": c["amount"],
                         "x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"], "conf": round(c["conf"], 1)})
            lines.append(f'[{k}] "{raw}" ({c["x"]}, {c["y"]}, {c["w"]}, {c["h"]})')
        (out / f"{p.stem}_ocr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out / f"{p.stem}_ocr.json").write_text(json.dumps({"original": str(p.resolve()), "cells": rows},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{i}/{len(files)} {p.name} 칸 {len(rows)}", flush=True)
    print(f"완료: {out}")


if __name__ == "__main__":
    main()
