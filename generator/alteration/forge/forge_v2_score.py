"""폴더 몇 개를 같은 점수로 빠르게 비교 (비교 이미지 없이). 이름이 _로 시작하는 시험 폴더도 됨.
python scripts/generate/forge_v2_score.py 목록.csv 폴더1 폴더2 ...
점수: 배경결·노이즈(원본대비, 1에 가까울수록 좋음), 탐지기 의심 수, 복사 탐지(동현님 copymove) 수"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataset" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge_v2_compare import ROOT, _file, _mask_box, background_texture  # noqa: E402
from detector import detect  # noqa: E402


def main():
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
    try:
        from eval_baseline import copymove_detect
    except Exception:
        copymove_detect = None
    print(f"{'폴더':28s} 배경결  노이즈  탐지기  복사탐지  장수")
    for d in sys.argv[2:]:
        p = Path(d)
        bgs, ns, sus, cm, n = [], [], 0, 0, 0
        for r in rows:
            f, mp = _file(p, r["id"]), p / f"{r['id']}_mask.png"
            if f is None or not mp.exists():
                continue
            n += 1
            im = Image.open(f).convert("RGB"); orig = Image.open(ROOT / r["original"]).convert("RGB")
            box = _mask_box(mp)
            res, ref = detect(im, mask=Image.open(mp)), detect(orig, mask=Image.open(mp))
            a, b = res["methods"]["noise"]["score"], ref["methods"]["noise"]["score"]
            if b:
                ns.append(a / b)
            g0 = background_texture(np.asarray(orig.convert("L")), box)
            if g0:
                bgs.append(background_texture(np.asarray(im.convert("L")), box) / g0)
            sus += bool(res["verdict"]["suspicious"])
            if copymove_detect is not None:
                try:
                    cm += bool(copymove_detect(np.asarray(im.convert("L")))[0])
                except Exception:
                    pass
        print(f"{p.name:28s} {np.nanmean(bgs):.3f}  {np.nanmean(ns):.3f}  {sus:3d}/{n}  {cm:3d}/{n}")


if __name__ == "__main__":
    main()
