"""
탐지기 v2 결과 뷰어 만들기 (2026-11-06). 이미지는 Mac 안 파일을 그대로 가리키고 밖으로 보내지 않는다.

  python scripts/detect/v2/make_viewer.py [--check A11_다시그린칸]
결과: outputs/평가셋/뷰어/index.html + img/ (문서 작은 사본, 최대 1400px). 파일로 바로 열어도 됨. 이미지는 이 Mac 안에만 있음 (outputs/ 는 git 제외)
문서마다: 실제 정상/위조, 탐지기 판정(의심/통과), 의심 칸(빨강), 실제 고친 칸(초록 점선), 위치 맞았는지, 점수.
3단계: 위조 의심(튜닝용 정상 오탐 2% 지점 초과) / 확인 필요(10% 지점 초과) / 정상.
"""
import argparse, csv, json, os, sys
from urllib.parse import quote
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import hit, mask_boxes  # noqa: E402

OUT = ROOT / "outputs" / "평가셋"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="A11_다시그린칸")
    ap.add_argument("--max-seconds", type=float, default=1e9, help="작은 사본 만들기 시간 제한 (끊기면 다시 실행하면 이어서)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig")))
    ocr = {d["id"]: d.get("cells") or [] for d in map(json.loads, open(OUT / "ocr.jsonl", encoding="utf-8"))}
    sc = {}
    for l in open(OUT / f"scores_{a.check}.jsonl", encoding="utf-8"):
        d = json.loads(l)
        sc[d["id"]] = sorted(d["cells"], key=lambda c: -c["score"])
    text = {}
    for i, cells in ocr.items():
        text[i] = {tuple(c["bbox"]): c["text"] for c in cells}
    tune = [sc[r["id"]][0]["score"] if sc.get(r["id"]) else -99 for r in rows
            if r["split"] == "tune" and r["label"] == "normal" and r["id"] in sc]
    # 3단계 판정: 위조 의심 = 튜닝용 정상 오탐 2% 지점 이상, 확인 필요 = 10% 지점 이상, 나머지 정상
    thr = {"hi": float(np.quantile(tune, 0.98)), "lo": float(np.quantile(tune, 0.90))}
    # 값마다 참고선: 튜닝용 정상 문서 칸들의 95% 지점
    pv = {}
    for r in rows:
        if r["split"] == "tune" and r["label"] == "normal":
            for c in sc.get(r["id"], []):
                for k, v in (c.get("parts") or {}).items():
                    pv.setdefault(k, []).append(v)
    pq = {k: round(float(np.quantile(v, 0.95)), 2) for k, v in pv.items()}
    import time
    t0 = time.time()
    IMG = OUT / "뷰어" / "img"
    IMG.mkdir(parents=True, exist_ok=True)
    docs = []
    for r in rows:
        if r["id"] not in sc:
            continue
        try:
            im = Image.open(ROOT / r["path"])
            W, H = im.size
        except Exception:
            continue
        small = IMG / f"{r['id']}.jpg"      # 뷰어 폴더 안 작은 사본: 파일로 열어도(브라우저가 상위 폴더를 막아도) 보이게. outputs/는 git 제외
        if not small.exists():
            if time.time() - t0 > a.max_seconds:
                print("시간 제한: 사본 만들기 중, 다시 실행하면 이어서"); return
            im = im.convert("RGB")
            im.thumbnail((1400, 1400))
            im.save(small, quality=85)
        gts = mask_boxes(r["mask"]) if r["mask"] else []
        cells = []
        for c in sc[r["id"]][:6]:
            b = tuple(c["bbox"])
            cells.append({"b": list(b), "s": round(c["score"], 4), "t": text[r["id"]].get(b, ""),
                          "why": c.get("why", ""), "hit": bool(gts) and hit(b, gts),
                          "p": c.get("parts", {}), "n": c.get("n")})   # 위치 적중: 70% 이상 겹침 (evaluate와 같음)
        docs.append({"id": r["id"], "img": "img/" + quote(r["id"]) + ".jpg", "W": W, "H": H, "label": r["label"],
                     "group": r["group"], "method": r["method"], "split": r["split"], "src": r["source_id"],
                     "gt": [list(g) for g in gts], "cells": cells,
                     "score": cells[0]["s"] if cells else -99})
    html = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    html = html.replace("__DATA__", json.dumps({"check": a.check, "thr": thr, "pq": pq, "docs": docs}, ensure_ascii=False))
    (OUT / "뷰어").mkdir(exist_ok=True)
    (OUT / "뷰어" / "index.html").write_text(html, encoding="utf-8")
    print(f"문서 {len(docs)}장, 위조 의심 > {thr['hi']:.3f} (정상 오탐 2%), 확인 필요 > {thr['lo']:.3f} (10%) -> {OUT / '뷰어' / 'index.html'}")


if __name__ == "__main__":
    main()
