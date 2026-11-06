"""
기준값 튜닝 — 저장된 측정값(1005쌍)으로 탐지법별 기준값과 종합 판정 기준을 정한다.
이미지를 다시 돌리지 않아서 몇 초면 끝난다. (측정값이 없으면 evaluate.py --collect 먼저)

방법 (숫자를 믿을 수 있게 튜닝용/검증용 데이터를 분리):
  1) 샘플을 무작위로 반반 나눔 (튜닝용 / 검증용)
  2) 튜닝용 '정상 원본' 분포에서 오탐 목표 지점을 기준값으로 잡음
       direction=low       → 하위 X% 지점 아래면 이상
       direction=high      → 상위 X% 지점 위면 이상
       direction=two_sided → 하위 X/2%, 상위 X/2% 바깥이면 이상
  3) 오탐 목표(X) × 종합 판정 최소 개수(min_votes) 조합 중
     튜닝용에서 '전체 오탐 ≤ --max-fp'를 지키면서 정확도가 가장 높은 조합 선택
  4) 처음 보는 검증용 절반으로 최종 성능 보고

사용: python tune.py              # 결과만 출력
      python tune.py --write      # detector/v1/config.json에 반영
"""
import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from detector.v1.core import DEFAULT_CONFIG, apply_rule, load_config  # noqa: E402

CACHE = ROOT / "outputs/reports/detector_measurements.jsonl"
REPORT = ROOT / "outputs/reports/detector_tuning.json"
FP_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15)


def load_scores():
    recs = []
    for line in open(CACHE, encoding="utf-8"):
        r = json.loads(line)
        if "skip" not in r:
            recs.append(r)
    return recs


def score(v):
    return float("inf") if v is None or v != v else float(v)  # nan(분모≈0) → inf, 탐지기와 동일


def thresholds(orig_scores, direction, fp):
    o = np.array([s for s in orig_scores if np.isfinite(s)])
    if direction == "low":
        return {"low": float(np.percentile(o, fp * 100)), "high": None}
    if direction == "high":
        return {"low": None, "high": float(np.percentile(o, 100 - fp * 100))}
    return {"low": float(np.percentile(o, fp * 50)), "high": float(np.percentile(o, 100 - fp * 50))}


def flags(recs, kind, key, rule):
    return np.array([apply_rule(score(r[kind][key]), 0.0, rule) for r in recs])


def rates(tp_flags, fp_flags):
    tp, fp = float(tp_flags.mean()), float(fp_flags.mean())
    return {"tp": tp, "fp": fp, "acc": (tp + 1 - fp) / 2,
            "precision": tp / (tp + fp) if tp + fp else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fp", type=float, default=0.10, help="종합 판정 오탐 상한 (기본 10%%)")
    ap.add_argument("--write", action="store_true", help="결과를 detector/v1/config.json에 반영")
    args = ap.parse_args()

    cfg = load_config()
    recs = load_scores()
    idx = np.random.default_rng(args.seed).permutation(len(recs))
    tune = [recs[i] for i in idx[: len(recs) // 2]]
    test = [recs[i] for i in idx[len(recs) // 2:]]
    active = {k: c for k, c in cfg["methods"].items() if c["enabled"]}
    print(f"샘플 {len(recs)}쌍 → 튜닝 {len(tune)} / 검증 {len(test)},  사용 탐지법: {', '.join(active)}")

    best = None
    for fp in FP_GRID:
        rules = {k: thresholds([score(r["orig"][k]) for r in tune], c["direction"], fp) for k, c in active.items()}
        votes_t = sum(flags(tune, "tamp", k, rules[k]).astype(int) for k in active)
        votes_o = sum(flags(tune, "orig", k, rules[k]).astype(int) for k in active)
        for mv in range(1, len(active) + 1):
            r = rates(votes_t >= mv, votes_o >= mv)
            if r["fp"] <= args.max_fp and (best is None or r["acc"] > best[2]["acc"]):
                best = (fp, mv, r, rules)
    fp, mv, tune_rates, rules = best

    print(f"\n선택: 탐지법별 오탐 목표 {fp*100:.1f}%, 종합 판정 = {mv}개 이상이면 위조 의심")
    print(f"\n[검증용 {len(test)}쌍 — 처음 보는 데이터]")
    print(f"{'탐지법':<20}{'기준값':>22}{'정탐':>8}{'오탐':>8}{'정확도':>8}")
    evidence = {}
    for k in active:
        r = rates(flags(test, "tamp", k, rules[k]), flags(test, "orig", k, rules[k]))
        lo, hi = rules[k]["low"], rules[k]["high"]
        rng = (f"<{lo:.3f}" if lo is not None else "") + (" 또는 " if lo is not None and hi is not None else "") + (f">{hi:.3f}" if hi is not None else "")
        print(f"{cfg['methods'][k]['name']:<20}{rng:>22}{r['tp']*100:>7.1f}%{r['fp']*100:>7.1f}%{r['acc']*100:>7.1f}%")
        evidence[k] = {"검증_정탐률": round(r["tp"], 4), "검증_오탐률": round(r["fp"], 4),
                       "검증_정확도": round(r["acc"], 4), "오탐목표": fp}
    vt = sum(flags(test, "tamp", k, rules[k]).astype(int) for k in active)
    vo = sum(flags(test, "orig", k, rules[k]).astype(int) for k in active)
    final = rates(vt >= mv, vo >= mv)
    print(f"{'▶ 종합 판정':<20}{f'{mv}개 이상':>22}{final['tp']*100:>7.1f}%{final['fp']*100:>7.1f}%{final['acc']*100:>7.1f}%"
          f"   (정밀도 {final['precision']*100:.1f}%)")

    report = {"date": str(date.today()), "n_pairs": len(recs), "n_tune": len(tune), "n_test": len(test),
              "seed": args.seed, "fp_target_per_method": fp, "min_votes": mv, "rules": rules,
              "per_method_test": evidence, "verdict_test": final, "verdict_tune": tune_rates}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {REPORT}")

    if args.write:
        for k in active:
            cfg["methods"][k]["rule"] = {kk: (round(v, 4) if v is not None else None)
                                         for kk, v in rules[k].items()}
            cfg["methods"][k]["evidence"] = evidence[k]
        cfg["verdict"] = {"min_votes": mv, "evidence": {
            "검증_정탐률": round(final["tp"], 4), "검증_오탐률": round(final["fp"], 4),
            "검증_정확도": round(final["acc"], 4), "검증_정밀도": round(final["precision"], 4),
            "튜닝일": str(date.today()), "데이터": f"{len(recs)}쌍 (튜닝 {len(tune)} / 검증 {len(test)})"}}
        json.dump(cfg, open(DEFAULT_CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        # 희귀도 계산용 정상 분포 (튜닝용 절반의 정상 원본 측정값, 정렬)
        ref = {k: sorted(v for v in (score(r["orig"][k]) for r in tune) if np.isfinite(v)) for k in active}
        json.dump(ref, open(DEFAULT_CONFIG.parent / "normal_reference.json", "w", encoding="utf-8"))
        print(f"반영: {DEFAULT_CONFIG}")


if __name__ == "__main__":
    main()
