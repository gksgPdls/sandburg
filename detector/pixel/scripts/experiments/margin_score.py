"""
실험: 설명 가능한 연속 점수 = 탐지법별로 '정상 범위를 얼마나 벗어났나'를 정상 분포 폭 단위로 재서 가장 큰 값.
  정상 범위 = 튜닝용 정상 원본 분포의 중앙값 기준, 폭 = 정상 원본 IQR (log 값 기준, FFT는 원값)
  점수 0 이하 = 정상 범위 안, 클수록 이상. 기준점은 튜닝용 정상 원본 오탐 목표 지점.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.core import load_config, judge  # noqa: E402
recs = [r for r in map(json.loads, open(ROOT / "outputs/reports/detector_measurements.jsonl")) if "skip" not in r]
cfg = load_config()
DIR = {"noise": "low", "fft": "two_sided", "pixel_std": "two_sided"}


def tf(k, v):
    if k == "fft":
        return 0.0 if v is None else float(v)
    v = 1e6 if v is None or v != v or v == float("inf") else v
    return float(np.log(np.clip(v, 1e-6, 1e6)))


def fit(tune):
    st = {}
    for k in DIR:
        o = np.array([tf(k, r["orig"][k]) for r in tune])
        q1, med, q3 = np.percentile(o, [25, 50, 75])
        st[k] = (med, max(q3 - q1, 1e-6))
    return st


def score(s, st):
    m = []
    for k, d in DIR.items():
        med, w = st[k]
        z = (tf(k, s[k]) - med) / w
        m.append(-z if d == "low" else abs(z))
    return max(m)


def vote(s):
    w = {k: (None if v is None else {"score": float("inf") if v != v else v, "inside": 0.0, "outside": 0.0}) for k, v in s.items()}
    return judge(w, cfg)["verdict"]["suspicious"]


for seed in range(4):
    idx = np.random.default_rng(seed).permutation(len(recs))
    tune, test = [recs[i] for i in idx[: len(recs) // 2]], [recs[i] for i in idx[len(recs) // 2:]]
    st = fit(tune)
    thr = float(np.percentile([score(r["orig"], st) for r in tune], 92))
    t = np.array([score(r["tamp"], st) for r in test]); o = np.array([score(r["orig"], st) for r in test])
    tp, fp = float(np.mean(t > thr)), float(np.mean(o > thr))
    auc = float(((t[:, None] > o[None]) + .5 * (t[:, None] == o[None])).mean())
    vt = np.mean([vote(r["tamp"]) for r in test]); vo = np.mean([vote(r["orig"]) for r in test])
    print(f"seed{seed} 투표: 정탐 {vt*100:.1f}% 오탐 {vo*100:.1f}% 정확도 {(vt+1-vo)*50:.1f}%  |  "
          f"이탈점수: 정탐 {tp*100:.1f}% 오탐 {fp*100:.1f}% 정확도 {(tp+1-fp)*50:.1f}%  AUC {auc:.3f}")
