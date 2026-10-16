"""
실험: 종합 판정을 '몇 개 걸렸나(투표)' 대신 측정값 가중 점수(로지스틱 회귀)로.
튜닝용 절반으로 가중치와 기준점을 정하고, 검증용 절반에서 투표 방식과 같은 오탐률일 때 정탐을 비교.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
MODEL = sys.argv[1] if len(sys.argv) > 1 else "lr"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.core import load_config, judge  # noqa: E402

recs = [r for r in map(json.loads, open(ROOT / "outputs/reports/detector_measurements.jsonl")) if "skip" not in r]
cfg = load_config()
FEATS = ["noise", "fft", "pixel_std", "boundary"]


def feat(s):
    def lg(v):
        v = 1e6 if v is None or v != v or v == float("inf") else v
        return float(np.log(np.clip(v, 1e-6, 1e6)))
    n, p, b = lg(s["noise"]), lg(s["pixel_std"]), lg(s["boundary"])
    f = s["fft"] or 0.0
    return [n, n * n, f, f * f, p, p * p, b]


def vote(s):
    w = {k: (None if v is None else {"score": float("inf") if v != v else v, "inside": 0.0, "outside": 0.0}) for k, v in s.items()}
    return judge(w, cfg)["verdict"]["suspicious"]


res = []
for seed in range(4):
    idx = np.random.default_rng(seed).permutation(len(recs))
    tune, test = [recs[i] for i in idx[: len(recs) // 2]], [recs[i] for i in idx[len(recs) // 2:]]
    X = np.array([feat(r["tamp"]) for r in tune] + [feat(r["orig"]) for r in tune])
    y = np.array([1] * len(tune) + [0] * len(tune))
    model = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)) if MODEL == "lr" else HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0)).fit(X, y)
    st = model.predict_proba(np.array([feat(r["tamp"]) for r in test]))[:, 1]
    so = model.predict_proba(np.array([feat(r["orig"]) for r in test]))[:, 1]
    vt = np.mean([vote(r["tamp"]) for r in test]); vo = np.mean([vote(r["orig"]) for r in test])
    # 기준점: 튜닝용 정상 원본에서 오탐 8% 지점 (투표 방식과 비슷한 오탐 수준)
    # 기준점은 튜닝용 데이터의 교차검증 점수로 정함 (학습에 쓴 데이터 점수는 과하게 자신만만해서 오탐 기준이 틀어짐)
    from sklearn.model_selection import cross_val_predict
    from sklearn.base import clone
    oof = cross_val_predict(clone(model), X, y, cv=5, method="predict_proba")[:, 1]
    so_tune = oof[len(tune):]
    thr = float(np.percentile(so_tune, 92))
    tp, fp = float(np.mean(st > thr)), float(np.mean(so > thr))
    auc = float(((st[:, None] > so[None]) + .5 * (st[:, None] == so[None])).mean())
    res.append((seed, vt, vo, tp, fp, auc))
    print(f"seed{seed}  투표: 정탐 {vt*100:.1f}% 오탐 {vo*100:.1f}% 정확도 {(vt+1-vo)*50:.1f}%   |  "
          f"가중점수: 정탐 {tp*100:.1f}% 오탐 {fp*100:.1f}% 정확도 {(tp+1-fp)*50:.1f}%  AUC {auc:.3f}")
if MODEL != "lr": sys.exit()
coef = model[-1].coef_[0]
print("가중치(표준화 후):", dict(zip(["log노이즈", "log노이즈^2", "fft", "fft^2", "log픽셀", "log픽셀^2", "log경계"], np.round(coef, 2))))
