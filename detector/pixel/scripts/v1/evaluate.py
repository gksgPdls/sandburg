"""
탐지기 평가 — 위조본/정상 원본 쌍으로 탐지법별·종합 성능표를 만든다.

  python evaluate.py --collect     # 1) 이미지에 탐지기를 돌려 측정값 저장 (오래 걸림, 끊겨도 이어서)
  python evaluate.py               # 2) 저장된 측정값 + 현재 config.json으로 성능표 (몇 초)

같은 마스크로 위조본 → TP/FN, 정상 원본 → FP/TN. 측정값은 tune.py도 같이 쓴다.
※ 전체 1005쌍 기준이라 튜닝에 쓴 절반도 포함됨 — 처음 보는 데이터 기준 성능은 tune.py 출력 참고.
출력: outputs/reports/detector_evaluation.csv / .json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from detector.pixel.v1.core import judge, load_config, measure  # noqa: E402

DATA = ROOT / "dataset"
META_DIR, MASK_DIR = DATA / "labels/meta", DATA / "labels/masks"
TAMPERED_DIR, ORIGINAL_DIR = DATA / "tampered/ai_inpainting", DATA / "original/bankbook"
REPORTS = ROOT / "outputs/reports"
CACHE = REPORTS / "detector_measurements.jsonl"


def find_original(base):
    for ext in (".jpg", ".jpeg", ".png"):
        if (ORIGINAL_DIR / f"{base}{ext}").exists():
            return ORIGINAL_DIR / f"{base}{ext}"
    return None


def plain(ms):
    return {k: (None if v is None else v["score"]) for k, v in ms.items()}


def _collect_one(lp):
    base = lp.stem.replace("_ai_inpaint", "")
    orig, tamp = find_original(base), TAMPERED_DIR / f"{base}_ai_inpaint.png"
    mask = MASK_DIR / f"{base}_ai_inpaint_mask.png"
    try:
        m = Image.open(mask)
        return {"base": base, "tamp": plain(measure(tamp, mask=m)), "orig": plain(measure(orig, mask=m))}
    except Exception as exc:  # noqa: BLE001
        return {"base": base, "skip": str(exc)}


def collect(fresh, max_seconds, workers=4):
    """이미지에 탐지기를 돌려 측정값 저장 (CPU 코어 workers개 병렬, 끊겨도 이어서)."""
    from multiprocessing import Pool
    t0 = time.time()
    if fresh and CACHE.exists():
        open(CACHE, "w").close()
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["base"] for l in open(CACHE, encoding="utf-8")}
    labels = [lp for lp in sorted(META_DIR.glob("*_ai_inpaint.json"))
              if lp.stem.replace("_ai_inpaint", "") not in done]
    total = len(done) + len(labels)
    with open(CACHE, "a", encoding="utf-8") as out, Pool(workers) as pool:
        for rec in pool.imap(_collect_one, labels, chunksize=4):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            done.add(rec["base"])
            if len(done) % 100 == 0:
                print(f"  {len(done)}/{total}", flush=True)
            if max_seconds and time.time() - t0 > max_seconds and len(done) < total:
                pool.terminate()
                print(f"시간 제한 — {len(done)}/{total} 완료, 다시 실행하면 이어서 함")
                return False
    return True


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def wrap(scores):
    return {k: (None if v is None else {"score": float("inf") if v != v else v, "inside": 0.0, "outside": 0.0})
            for k, v in scores.items()}


def table():
    cfg = load_config()
    recs = [r for r in map(json.loads, open(CACHE, encoding="utf-8")) if "skip" not in r]
    n = len(recs)
    cols = list(cfg["methods"]) + ["_verdict"]
    cm = {c: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for c in cols}
    for r in recs:
        for kind, (hit, miss) in (("tamp", ("TP", "FN")), ("orig", ("FP", "TN"))):
            res = judge(wrap(r[kind]), cfg)
            for k, m in res["methods"].items():
                if m["enabled"]:
                    cm[k][hit if m["flagged"] else miss] += 1
            cm["_verdict"][hit if res["verdict"]["suspicious"] else miss] += 1

    pct = lambda a, b: a / b * 100 if b else 0.0
    hdr = ["탐지법", "사용", "TP(정탐)", "FN(미탐)", "FP(오탐)", "TN", "정밀도", "정확도", "AUC"]
    rows = []
    print(f"\n평가 {n}쌍 (위조본 {n} + 정상 원본 {n}), 기준: detector/pixel/v1/config.json")
    print("\t".join(hdr))
    for c in cols:
        x = cm[c]
        if c == "_verdict":
            name, used = f"▶ 종합 판정({cfg['verdict']['min_votes']}개 이상)", "O"
            a = ""
        else:
            mc = cfg["methods"][c]
            name, used = mc["name"], "O" if mc["enabled"] else "X"
            t = [r["tamp"][c] for r in recs if r["tamp"][c] is not None and r["tamp"][c] == r["tamp"][c]]
            o = [r["orig"][c] for r in recs if r["orig"][c] is not None and r["orig"][c] == r["orig"][c]]
            if mc["direction"] == "two_sided":
                f = (lambda v: abs(v)) if c == "fft" else (lambda v: abs(np.log(max(v, 1e-9))))
                t, o = [f(v) for v in t], [f(v) for v in o]
            elif mc["direction"] == "low":
                t, o = [-v for v in t], [-v for v in o]
            a = f"{auc(t, o):.3f}"
        if used == "X":
            row = [name, used, "-", "-", "-", "-", "-", "-", a]
        else:
            row = [name, used, f"{x['TP']} ({pct(x['TP'], n):.1f}%)", f"{x['FN']} ({pct(x['FN'], n):.1f}%)",
                   f"{x['FP']} ({pct(x['FP'], n):.1f}%)", f"{x['TN']} ({pct(x['TN'], n):.1f}%)",
                   f"{pct(x['TP'], x['TP'] + x['FP']):.1f}%", f"{pct(x['TP'] + x['TN'], 2 * n):.1f}%", a]
        rows.append(row)
        print("\t".join(row))
    with open(REPORTS / "detector_evaluation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    json.dump({"n_pairs": n, "config": cfg, "confusion": cm}, open(REPORTS / "detector_evaluation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved: {REPORTS / 'detector_evaluation.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="이미지에 탐지기를 돌려 측정값 저장")
    ap.add_argument("--fresh", action="store_true", help="저장된 측정값 지우고 처음부터")
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    if a.collect and not collect(a.fresh, a.max_seconds):
        return
    table()


if __name__ == "__main__":
    main()
