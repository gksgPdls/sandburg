"""한국어 OCR (EasyOCR). 진료비 영수증의 칸 이름(본인부담금, 합계, 진료비 총액 등)을 읽으려고 씀.
결과는 사진마다 outputs/고도화/_ocr_ko/<파일이름>.json 에 저장해 두고 다시 읽음 (한 장에 몇 초 걸려서).
맥에서만 돌림 (torch 필요). 모델은 처음 실행 때 GitHub에서 자동으로 받음 (로컬 실행, 사진은 밖으로 안 나감).

python generator/alteration/forge/3_logic/ocr_ko.py --src dataset/original/bankbook --limit 60   # 미리 읽어 두기"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "outputs" / "고도화" / "_ocr_ko"
_READER = None


def _reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _READER


def read(path, allow_run=True):
    """[{text, conf, x, y, w, h}] (원본 좌표). 캐시가 없고 allow_run=False면 None."""
    path = Path(path)
    c = CACHE / f"{path.stem}.json"
    if c.exists():
        return json.loads(c.read_text(encoding="utf-8"))
    if not allow_run:
        return None
    res = _reader().readtext(str(path), detail=1, paragraph=False)
    out = []
    for quad, text, conf in res:
        xs = [float(p[0]) for p in quad]; ys = [float(p[1]) for p in quad]
        out.append({"text": text, "conf": round(float(conf), 3), "x": int(min(xs)), "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys))})
    CACHE.mkdir(parents=True, exist_ok=True)
    c.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 몇 장 (0이면 전부)")
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    src = src if src.is_absolute() else ROOT / src
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if a.limit:
        files = files[:a.limit]
    t0 = time.time()
    for i, p in enumerate(files):
        read(p)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"{i + 1}/{len(files)}  {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
