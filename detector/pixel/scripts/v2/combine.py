"""
A 검사 합치기 (2026-11-06): 검사마다 문서 점수(가장 의심 칸)를 '튜닝용 정상 문서 중 이보다 높은 비율'(p)로 바꾸고
문서 p = 검사들 중 가장 작은 p. 기준값은 다시 튜닝용 정상 문서에서 오탐 --fpr 지점 (검사 수만큼의 다중비교가 자동 반영됨).

  python detector/pixel/scripts/v2/combine.py --checks A11_다시그린칸,A5_흰박스 [--fpr 0.10]
먼저 evaluate.py 로 각 검사의 scores_<검사>.jsonl 을 만들어 둔다. 결과는 화면 표 (평가 방식은 evaluate.py와 같음)
"""
import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import auc, hit, mask_boxes, wilson  # noqa: E402

OUT = Path(os.environ.get("EVALSET_DIR", ROOT / "outputs" / "평가셋"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", required=True)
    ap.add_argument("--fpr", type=float, default=0.10)
    a = ap.parse_args()
    names = a.checks.split(",")
    rows = {r["id"]: r for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8-sig"))}
    sc = {}
    for n in names:
        sc[n] = {}
        for l in open(OUT / f"scores_{n}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            sc[n][d["id"]] = sorted(d["cells"], key=lambda c: -c["score"])
    ids = [i for i in rows if all(i in sc[n] for n in names)]
    tune_norm = [i for i in ids if rows[i]["split"] == "tune" and rows[i]["label"] == "normal"]
    ref = {n: np.sort([sc[n][i][0]["score"] if sc[n][i] else -1e9 for i in tune_norm]) for n in names}

    def pval(n, s):
        return (len(ref[n]) - np.searchsorted(ref[n], s, side="left") + 1) / (len(ref[n]) + 1)

    D, top = {}, {}
    for i in ids:
        best = (1.0, None)
        for n in names:
            if sc[n][i]:
                p = pval(n, sc[n][i][0]["score"])
                if p < best[0]:
                    best = (p, sc[n][i][0]["bbox"])
        D[i], top[i] = -np.log(best[0]), best[1]
    thr = float(np.quantile([D[i] for i in tune_norm], 1 - a.fpr))
    test = [i for i in ids if rows[i]["split"] == "test"]
    norm = [i for i in test if rows[i]["label"] == "normal"]
    fp = sum(D[i] > thr for i in norm); lo, hi = wilson(fp, len(norm))
    print(f"[합침 {'+'.join(names)}]  튜닝용 정상 {len(tune_norm)}장으로 기준")
    print(f"  정상 문서: {len(norm)}장, 오탐 {fp}, 오탐률 {fp / max(len(norm), 1):.1%} ({lo:.1%}~{hi:.1%})")
    by = defaultdict(list)
    for i in test:
        if rows[i]["label"] == "forged":
            by[(rows[i]["group"], rows[i]["method"])].append(i)
    for (g, m), rs in sorted(by.items()):
        tp = [i for i in rs if D[i] > thr]
        lo, hi = wilson(len(tp), len(rs))
        loc = sum(1 for i in tp if top[i] and hit(tuple(top[i]), mask_boxes(rows[i]["mask"])))
        print(f"  {g}/{m}: {len(rs)}장, 정탐 {len(tp)}, 미탐 {len(rs) - len(tp)}, 재현율 {len(tp) / len(rs):.1%} ({lo:.1%}~{hi:.1%}), "
              f"AUC {auc([D[i] for i in rs], [D[i] for i in norm]):.3f}, 위치적중 {loc}/{len(tp)}")


if __name__ == "__main__":
    main()
