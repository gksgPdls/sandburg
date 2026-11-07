"""compare_patch.jsonl 분석: 변형별 탐지법 AUC + 튜닝/검증 분리 종합 판정 성능."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "detector/pixel/scripts/v1"))
from tune import FP_GRID, thresholds, rates  # noqa: E402
sys.path.insert(0, str(ROOT))
from detector.pixel.v1.core import apply_rule  # noqa: E402

recs = [r for r in map(json.loads, open(ROOT / "outputs/reports/experiments/compare_patch.jsonl")) if "skip" not in r]
DIR = {"noise": "low", "fft": "two_sided", "pixel_std": "two_sided"}
inf = lambda v: float("inf") if v is None or v != v else v
def auc(p, n):
    p, n = np.asarray(p), np.asarray(n); return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean())
def tr(k, v):
    v = inf(v)
    if DIR[k] == "low": return -v
    if k == "fft": return abs(v)
    return abs(np.log(max(min(v, 1e9), 1e-9)))
idx = np.random.default_rng(0).permutation(len(recs)); tune = [recs[i] for i in idx[:len(recs)//2]]; test = [recs[i] for i in idx[len(recs)//2:]]
variants = list(recs[0]["tamp"])
print(f"{'변형':<8}{'노이즈AUC':>10}{'FFT AUC':>9}{'픽셀AUC':>9}   종합(검증): 정탐  오탐  정확도  [오탐목표,N]")
for v in variants:
    a = [auc([tr(k, r["tamp"][v][k]) for r in recs], [tr(k, r["orig"][v][k]) for r in recs]) for k in DIR]
    best = None
    for fp in FP_GRID:
        rules = {k: thresholds([inf(r["orig"][v][k]) for r in tune], DIR[k], fp) for k in DIR}
        vt = sum(np.array([apply_rule(inf(r["tamp"][v][k]), 0, rules[k]) for r in tune]).astype(int) for k in DIR)
        vo = sum(np.array([apply_rule(inf(r["orig"][v][k]), 0, rules[k]) for r in tune]).astype(int) for k in DIR)
        for mv in (1, 2, 3):
            rr = rates(vt >= mv, vo >= mv)
            if rr["fp"] <= .10 and (best is None or rr["acc"] > best[0]["acc"]): best = (rr, fp, mv, rules)
    _, fp, mv, rules = best
    vt = sum(np.array([apply_rule(inf(r["tamp"][v][k]), 0, rules[k]) for r in test]).astype(int) for k in DIR)
    vo = sum(np.array([apply_rule(inf(r["orig"][v][k]), 0, rules[k]) for r in test]).astype(int) for k in DIR)
    t = rates(vt >= mv, vo >= mv)
    print(f"{v:<8}{a[0]:>10.3f}{a[1]:>9.3f}{a[2]:>9.3f}   {t['tp']*100:>9.1f}%{t['fp']*100:>5.1f}%{t['acc']*100:>6.1f}%  [{fp},{mv}]")
