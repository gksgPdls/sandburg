"""
탐지법별 "구분력"(AUC) 측정 — 임계치와 무관하게, 측정값 자체가 위조본과
정상 원본을 얼마나 잘 갈라놓는지 본다.

- 같은 마스크로 위조본/정상 원본 각각에서 탐지법의 원래 측정값(비율·차이)을 구한다.
- AUC: 위조본 하나, 정상 하나를 무작위로 뽑았을 때 위조본 점수가 더 "이상한" 확률.
  0.5 = 구분 못 함(동전 던지기), 1.0 = 완벽. 방향 3가지를 모두 재서 가장 좋은 걸 보고한다:
    "높을수록 위조" / "낮을수록 위조" / "1에서 멀수록 위조(양방향)"

사용: python measure_auc.py [--max-seconds S] [--fresh]
  (중간에 끊겨도 다시 실행하면 이어서 함 — 캐시: outputs/reports/detector_raw_scores.jsonl)
출력: outputs/reports/detector_auc.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
from config import INPUT_DIR, MASK_DIR, META_DIR, REPORT_DIR, TAMPERED_DIR, ensure_dirs
import detectors as det

METRICS = {  # key: (탐지법 이름, 설명)
    "ela_ratio": ("압축흔적_ELA", "ELA 안쪽/바깥쪽"),
    "noise_ratio": ("신호패턴_노이즈분석", "웨이블릿 노이즈 안쪽/바깥쪽"),
    "fft_diff": ("신호패턴_FFT주파수", "고주파비율 안쪽-바깥쪽"),
    "boundary_ratio": ("영역불일치_합성경계분석", "테두리 그래디언트/배경"),
    "std_ratio": ("AI탐지_노이즈지문·패치통계", "픽셀 표준편차 안쪽/바깥쪽 (두 탐지법이 같은 값 사용)"),
}


def safe_ratio(a, b):
    return a / b if b > 1e-6 else float("nan")


def raw_scores(image_path, mask_arr):
    gray = np.asarray(Image.open(image_path).convert("L"))
    bbox = det.mask_bbox_from_mask(mask_arr)
    _, e_in, e_out = det.compute_ela_detect(image_path, mask_arr)
    _, n_in, n_out = det.compute_noise_detect(gray, mask_arr, bbox)
    _, f_in, f_out = det.compute_fft_detect(gray, bbox)
    _, r_in, r_bg = det.compute_boundary_detect(gray, mask_arr)
    _, s_in, s_out = det.compute_noise_fingerprint_detect(gray, bbox)
    return {"ela_ratio": safe_ratio(e_in, e_out), "noise_ratio": safe_ratio(n_in, n_out),
            "fft_diff": f_in - f_out, "boundary_ratio": safe_ratio(r_in, r_bg),
            "std_ratio": safe_ratio(s_in, s_out)}


def auc(pos, neg):
    """Mann-Whitney AUC (P(pos > neg), 동점은 0.5)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # 동점 평균 순위
        idx = allv == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    cache_path = REPORT_DIR / "detector_raw_scores.jsonl"
    if args.fresh and cache_path.exists():
        cache_path.unlink()
    done = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line); done[r["base"]] = r

    labels = sorted(META_DIR.glob("*_ai_inpaint.json"))
    with open(cache_path, "a", encoding="utf-8") as cache:
        for i, lp in enumerate(labels, 1):
            base = lp.stem.replace("_ai_inpaint", "")
            if base in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print(f"시간 제한 — {len(done)}/{len(labels)} 완료, 다시 실행하면 이어서 함")
                return
            orig = next((INPUT_DIR / f"{base}{e}" for e in (".jpg", ".jpeg", ".png")
                         if (INPUT_DIR / f"{base}{e}").exists()), None)
            tamp = TAMPERED_DIR / f"{base}_ai_inpaint.png"
            mask = MASK_DIR / f"{base}_ai_inpaint_mask.png"
            try:
                m = np.asarray(Image.open(mask).convert("L"))
                rec = {"base": base, "tamp": raw_scores(tamp, m), "orig": raw_scores(orig, m)}
            except Exception as exc:  # noqa: BLE001
                rec = {"base": base, "skip": str(exc)}
            done[base] = rec
            cache.write(json.dumps(rec, ensure_ascii=False) + "\n"); cache.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(labels)}", flush=True)

    recs = [r for r in done.values() if "skip" not in r]
    report = {"n": len(recs), "metrics": {}}
    print(f"\n{len(recs)}쌍 기준 구분력 (AUC: 0.5=동전던지기, 0.7=쓸만, 0.8+=좋음)")
    print(f"{'탐지법':<26}{'AUC':>6}  가장 잘 갈리는 방향")
    for key, (name, desc) in METRICS.items():
        t = np.array([r["tamp"][key] for r in recs], float)
        o = np.array([r["orig"][key] for r in recs], float)
        t, o = t[np.isfinite(t)], o[np.isfinite(o)]
        cands = {"높을수록 위조": auc(t, o), "낮을수록 위조": auc(-t, -o)}
        if key != "fft_diff":
            lt, lo = np.abs(np.log(np.clip(t, 1e-9, None))), np.abs(np.log(np.clip(o, 1e-9, None)))
            cands["1에서 멀수록 위조(양방향)"] = auc(lt, lo)
        else:
            cands["0에서 멀수록 위조(양방향, 지금 방식)"] = auc(np.abs(t), np.abs(o))
        best = max(cands, key=cands.get)
        print(f"{name:<26}{cands[best]:>6.3f}  {best}")
        report["metrics"][name] = {"metric": desc, "auc_by_direction": cands, "best": best,
                                   "tamp_median": float(np.median(t)), "orig_median": float(np.median(o))}
    json.dump(report, open(REPORT_DIR / "detector_auc.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved: {REPORT_DIR / 'detector_auc.json'}")


if __name__ == "__main__":
    main()
