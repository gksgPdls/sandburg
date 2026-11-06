"""
복사-붙여넣기(카피앤무브) 평가용 위조 만들기 (2026-11-06). 실제 문서 기반이라 로컬 전용.

  python scripts/generate/make_copymove.py [--n 400] [--max-seconds 150]
정상 원본마다(평가셋 manifest의 '원본') 금액 칸 A를 같은 문서의 크기가 비슷한 다른 금액 칸 B 위에 픽셀 그대로 붙인다
(오른쪽 끝·세로 가운데 맞춤, 크기 변경 없음). A가 B를 다 덮도록 A 높이 ±2px, 폭 B 이상 1.6배 이하인 쌍만.
원본이 JPEG이면 원본과 같은 압축표로 저장 (harmonize.save_like 와 같은 방식). 고친 위치 마스크 = 붙인 직사각형.
결과: outputs/고도화/copymove_평가용/AI/cm_<원본>.jpg, _mask.png, 목록.csv (원본 칸·붙인 칸·글자)
"""
import argparse, csv, json, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from detector.v2.cellfilter import money_like  # noqa: E402

OUT = ROOT / "outputs" / "고도화" / "copymove_평가용" / "AI"


def save_like(rgb, src_path, out_base):
    src = Image.open(src_path)
    im = Image.fromarray(rgb)
    if src.format == "JPEG" and getattr(src, "quantization", None):
        kw = {"qtables": src.quantization}
        try:
            sub = JpegImagePlugin.get_sampling(src)
            if sub in (0, 1, 2):
                kw["subsampling"] = sub
        except Exception:
            pass
        p = out_base.with_suffix(".jpg"); im.save(p, **kw); return p
    p = out_base.with_suffix(".png"); im.save(p); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = ROOT / "outputs" / "평가셋"
    man = [r for r in csv.DictReader(open(ev / "manifest.csv", encoding="utf-8-sig")) if r["method"] == "원본"]
    ocr = {d["id"]: d.get("cells") or [] for d in map(json.loads, open(ev / "ocr.jsonl", encoding="utf-8"))}
    lst = OUT / "목록.csv"
    done = {r["id"] for r in csv.DictReader(open(lst, encoding="utf-8-sig"))} if lst.exists() else set()
    rng = random.Random(11)
    rng.shuffle(man)
    t0, made = time.time(), len(done)
    new = not lst.exists()
    with open(lst, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "source_id", "split", "src_bbox", "src_text", "dst_bbox", "dst_text", "paste_bbox"])
        if new:
            w.writeheader()
        for r in man:
            if made >= a.n:
                break
            sid = f"cm_{r['source_id']}"
            if sid in done:
                continue
            if time.time() - t0 > a.max_seconds:
                print("시간 제한 - 다시 실행하면 이어서"); break
            C = [c for c in ocr.get(r["id"], []) if money_like(c) and c["kind"] == "money"]
            pairs = [(s, d) for s in C for d in C if s is not d and s["text"] != d["text"]
                     and abs(s["bbox"][3] - d["bbox"][3]) <= 2 and d["bbox"][2] <= s["bbox"][2] <= 1.6 * d["bbox"][2]]
            if not pairs:
                continue
            s, d = rng.choice(pairs)
            img = np.asarray(Image.open(ROOT / r["path"]).convert("RGB")).copy()
            H, W = img.shape[:2]
            sx, sy, sw, sh = s["bbox"]; dx, dy, dw, dh = d["bbox"]
            px = dx + dw - sw; py = dy + dh // 2 - sh // 2
            if px < 0 or py < 0 or px + sw > W or py + sh > H:
                continue
            # 붙일 자리가 원래 칸(A)과 겹치면 건너뜀
            if not (px + sw <= sx or sx + sw <= px or py + sh <= sy or sy + sh <= py):
                continue
            img[py:py + sh, px:px + sw] = img[sy:sy + sh, sx:sx + sw].copy()
            p = save_like(img, ROOT / r["path"], OUT / sid)
            m = np.zeros((H, W), np.uint8); m[py:py + sh, px:px + sw] = 255
            cv2.imwrite(str(OUT / f"{sid}_mask.png"), m)
            w.writerow({"id": sid, "source_id": r["source_id"], "split": r["split"], "src_bbox": list(s["bbox"]),
                        "src_text": s["text"], "dst_bbox": list(d["bbox"]), "dst_text": d["text"], "paste_bbox": [px, py, sw, sh]})
            fh.flush(); made += 1
    print(f"복사-붙여넣기 위조 {made}장 (목표 {a.n})")


if __name__ == "__main__":
    main()
