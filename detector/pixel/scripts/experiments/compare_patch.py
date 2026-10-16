"""
실험: 비교 패치 고르는 방식 바꾸기.
  기존(k1_f0)  : 밝기+엣지밀도가 가장 비슷한 패치 1개
  f05          : 엣지밀도가 의심영역의 50% 미만인 패치(빈 종이 등)는 후보에서 제외
  k3, k5       : 가장 비슷한 패치 k개를 골라 각각 측정 후 중앙값과 비교
측정값을 변형별로 저장 → outputs/reports/experiments/compare_patch.jsonl (끊겨도 이어서)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector import features as F  # noqa: E402
from detector.core import mask_to_bbox  # noqa: E402

DATA = ROOT / "dataset"
OUT = ROOT / "outputs/reports/experiments/compare_patch.jsonl"
VARIANTS = {"k1_f0": (1, 0.0), "k1_f05": (1, 0.5), "k3_f05": (3, 0.5), "k5_f05": (5, 0.5), "k5_f0": (5, 0.0)}


def ranked_candidates(gray, bbox, search_radius=80, num_distances=4):
    """sample_outside_patch와 같은 후보·거리 계산, 대신 순위 전체와 엣지밀도를 돌려줌."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    inside = gray[y0:y1, x0:x1]
    cands = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for step in range(1, num_distances + 1):
            dist = step * max(bw, bh, 10)
            if dist > search_radius + max(bw, bh):
                continue
            ox0, oy0 = x0 + dx * dist, y0 + dy * dist
            if ox0 < 0 or oy0 < 0 or ox0 + bw > gray.shape[1] or oy0 + bh > gray.shape[0]:
                continue
            p = gray[oy0:oy0 + bh, ox0:ox0 + bw]
            if p.shape == inside.shape:
                cands.append(p)
    if not cands:
        ox0 = max(min(x1 + 5, gray.shape[1] - bw), 0)
        return inside, [gray[y0:y1, ox0:ox0 + bw]], [None]
    means = np.array([float(p.mean()) for p in cands])
    edges = np.array([F._edge_density(p) for p in cands])
    ms = means.std() if means.std() > 1e-6 else 1.0
    es = edges.std() if edges.std() > 1e-6 else 1.0
    d = np.abs(means - inside.mean()) / ms + np.abs(edges - F._edge_density(inside)) / es
    order = np.argsort(d, kind="stable")
    return inside, [cands[i] for i in order], [float(edges[i]) for i in order]


def scores(img_path, mask):
    gray = np.asarray(Image.open(img_path).convert("L"))
    inside, cands, edges = ranked_candidates(gray, mask_to_bbox(mask))
    ie = F._edge_density(inside)
    cache = {}

    def feats(i):
        if i not in cache:
            p = cands[i]
            cache[i] = (F.wavelet_noise_std(p), F.high_freq_energy_ratio(p), float(p.std()))
        return cache[i]
    n_in, h_in, s_in = F.wavelet_noise_std(inside), F.high_freq_energy_ratio(inside), float(inside.std())
    out = {}
    for name, (k, f) in VARIANTS.items():
        idx = [i for i, e in enumerate(edges) if e is None or e >= f * ie] or list(range(len(cands)))
        idx = idx[:k]
        fs = np.array([feats(i) for i in idx])
        n_o, h_o, s_o = np.median(fs[:, 0]), np.median(fs[:, 1]), np.median(fs[:, 2])
        out[name] = {"noise": n_in / n_o if n_o > 1e-6 else None, "fft": h_in - h_o,
                     "pixel_std": s_in / s_o if s_o > 1e-6 else None}
    return out


def work(base):
    try:
        mask = np.asarray(Image.open(DATA / f"labels/masks/{base}_ai_inpaint_mask.png").convert("L"))
        orig = next(p for p in (DATA / "original/bankbook").glob(base + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        return {"base": base, "tamp": scores(DATA / f"tampered/ai_inpainting/{base}_ai_inpaint.png", mask),
                "orig": scores(orig, mask)}
    except Exception as e:  # noqa: BLE001
        return {"base": base, "skip": str(e)}


if __name__ == "__main__":
    max_s = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    t0 = time.time()
    done = {json.loads(l)["base"] for l in open(OUT)} if OUT.exists() else set()
    todo = [p.stem.replace("_ai_inpaint", "") for p in sorted((DATA / "labels/meta").glob("*_ai_inpaint.json"))]
    todo = [b for b in todo if b not in done]
    with open(OUT, "a") as f, Pool(4) as pool:
        for r in pool.imap(work, todo, chunksize=4):
            f.write(json.dumps(r) + "\n"); f.flush(); done.add(r["base"])
            if time.time() - t0 > max_s:
                pool.terminate(); break
    print(f"{len(done)}/{len(done) + len([b for b in todo if b not in done])} 완료")
