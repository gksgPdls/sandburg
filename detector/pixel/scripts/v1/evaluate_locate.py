"""
의심 위치 자동 탐색 평가 (좌표 입력 없이 문서 이미지만으로).

  python evaluate_locate.py --collect [--max-seconds 150]   # OCR + 칸별 탐지 결과 저장 (끊겨도 이어서)
  python evaluate_locate.py                                  # 문서 기준값 정하고 성능표

데이터: tune.py와 같은 분할(seed 0). 문서 기준값은 튜닝용 절반 일부(정상 원본)로 정하고, 성능은 검증용 절반으로 잰다.
측정 항목
  - OCR 후보 포함률: 위조된 칸이 OCR 후보 목록에 들어갔나 (못 들어가면 검사조차 안 됨)
  - 1등 적중률: 가장 의심스러운 칸이 실제 위조 칸인가
  - 문서 정탐/오탐: 위조 문서를 위조 의심으로, 정상 문서를 정상으로 판정했나
"""
import argparse
import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # tesseract 병렬 실행 시 스레드 과다로 느려지는 것 방지
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from detector.pixel.v1.core import load_config  # noqa: E402
from detector.pixel.v1.locate import locate  # noqa: E402

DATA = ROOT / "dataset"
OCR = os.environ.get("LOCATE_OCR", "tesseract")   # rapid: RapidOCR 모든 글자 칸 (2026-10-30)
_SUF = "" if OCR == "tesseract" else f"_{OCR}"
CACHE = ROOT / f"outputs/reports/locate_cache{_SUF}.jsonl"
REPORT = ROOT / f"outputs/reports/locate_evaluation{_SUF}.json"
ONLY = "all"   # 보고할 후보 범위 (rapid): all / digits / money
N_TUNE_DOCS = 200


def split():
    recs = [json.loads(l) for l in open(ROOT / "outputs/reports/detector_measurements.jsonl", encoding="utf-8")]
    bases = [r["base"] for r in recs if "skip" not in r]
    idx = np.random.default_rng(0).permutation(len(bases))
    return [bases[i] for i in idx[: len(bases) // 2]], [bases[i] for i in idx[len(bases) // 2:]]


def label_box(base):
    b = json.load(open(DATA / f"labels/meta/{base}_ai_inpaint.json", encoding="utf-8"))["forged_regions"][0]["bbox"]
    return b["x"], b["y"], b["w"], b["h"]


def hits(c, lb, tol=6):
    x, y, w, h = c
    cx, cy = x + w / 2, y + h / 2
    lx, ly, lw, lh = lb
    return lx - tol <= cx <= lx + lw + tol and ly - tol <= cy <= ly + lh + tol


def work(job):
    base, kind = job
    path = (DATA / f"tampered/ai_inpainting/{base}_ai_inpaint.png" if kind == "tamp" else
            next(p for p in (DATA / "original/bankbook").glob(base + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
    try:
        r = locate(path, ocr=OCR)
        lb = label_box(base)
        cands = [{"text": c["text"], "kind": c.get("kind"), "bbox": list(c["bbox"]), "rarity": c["rarity"],
                  "votes": c["votes"], "is_target": hits(c["bbox"], lb)} for c in r["candidates"]]
        return {"base": base, "kind": kind, "cands": cands}
    except Exception as e:  # noqa: BLE001
        return {"base": base, "kind": kind, "skip": str(e)}


def collect(max_seconds):
    t0 = time.time()
    tune, test = split()
    jobs = [(b, k) for b in test for k in ("tamp", "orig")] + [(b, "orig") for b in tune[:N_TUNE_DOCS]]
    done = set()
    if CACHE.exists():
        done = {(r["base"], r["kind"]) for r in map(json.loads, open(CACHE, encoding="utf-8"))}
    todo = [j for j in jobs if j not in done]
    with open(CACHE, "a", encoding="utf-8") as f, Pool(4) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=2):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); done.add((r["base"], r["kind"]))
            if time.time() - t0 > max_seconds and len(done) < len(jobs):
                pool.terminate()
                print(f"시간 제한 — {len(done)}/{len(jobs)} 완료, 다시 실행하면 이어서 함")
                return False
    return True


def _keep(c):
    if ONLY == "digits":
        return c.get("kind") != "text"
    if ONLY == "money":
        return c.get("kind") in ("money", "date")
    return True


def _filter(r):
    cs = sorted([c for c in r["cands"] if _keep(c)], key=lambda c: c["rarity"] if c["rarity"] is not None else 1.0)
    return {**r, "cands": cs}


def doc_rarity(r):
    rs = [c["rarity"] for c in r["cands"] if c["rarity"] is not None]
    return min(rs) if rs else 1.0


def report(max_fp):
    tune, test = split()
    recs = {(r["base"], r["kind"]): _filter(r) for r in map(json.loads, open(CACHE, encoding="utf-8")) if "skip" not in r}
    tune_o = [doc_rarity(recs[(b, "orig")]) for b in tune[:N_TUNE_DOCS] if (b, "orig") in recs]
    thr = float(np.percentile(tune_o, max_fp * 100))   # 정상 문서 오탐 max_fp 지점
    T = [recs[(b, "tamp")] for b in test if (b, "tamp") in recs]
    O = [recs[(b, "orig")] for b in test if (b, "orig") in recs]
    in_cands = np.mean([any(c["is_target"] for c in r["cands"]) for r in T])
    top1 = np.mean([bool(r["cands"]) and r["cands"][0]["is_target"] for r in T])
    tp = np.mean([doc_rarity(r) < thr for r in T])
    tp_loc = np.mean([doc_rarity(r) < thr and r["cands"][0]["is_target"] for r in T])
    fp = np.mean([doc_rarity(r) < thr for r in O])
    n_c = np.mean([len(r["cands"]) for r in O])
    # OCR이 위조 칸을 찾은 문서만 따로
    T_found = [r for r in T if any(c["is_target"] for c in r["cands"])]
    tp_found = np.mean([doc_rarity(r) < thr and r["cands"][0]["is_target"] for r in T_found]) if T_found else 0
    rep = {"문서_기준_희귀도": thr, "검증_위조문서": len(T), "검증_정상문서": len(O), "문서당_평균_후보칸": float(n_c),
           "OCR_후보포함률": float(in_cands), "1등_적중률": float(top1),
           "문서_정탐률": float(tp), "문서_정탐률_위치까지정확": float(tp_loc), "문서_오탐률": float(fp),
           "OCR이_찾은_문서중_위치까지_정탐": float(tp_found)}
    print(f"\n[검증용 위조 문서 {len(T)}장 / 정상 문서 {len(O)}장, 좌표 입력 없음]")
    print(f"문서당 평균 금액·날짜 후보 칸      {n_c:.1f}칸")
    print(f"OCR 후보 포함률 (위조 칸을 찾았나)   {in_cands*100:.1f}%")
    print(f"1등 적중률 (가장 의심 칸 = 위조 칸)  {top1*100:.1f}%")
    print(f"문서 기준: 가장 이상한 칸의 희귀도 < {thr:.4f}  (튜닝용 정상 문서 오탐 {max_fp*100:.0f}% 지점)")
    print(f"문서 정탐률                         {tp*100:.1f}%   (위치까지 맞힌 것 {tp_loc*100:.1f}%)")
    print(f"문서 오탐률                         {fp*100:.1f}%")
    print(f"OCR이 위조 칸을 찾은 문서 중 위치까지 정탐  {tp_found*100:.1f}%")
    json.dump(rep, open(REPORT if ONLY == "all" else REPORT.with_name(REPORT.stem + f"_{ONLY}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return thr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="저장된 결과 비우고 처음부터 (후보 규칙·탐지기를 바꿨을 때)")
    ap.add_argument("--max-seconds", type=float, default=150)
    ap.add_argument("--max-fp", type=float, default=0.10)
    ap.add_argument("--only", default="all", choices=["all", "digits", "money"], help="rapid 후보 범위")
    ap.add_argument("--write", action="store_true", help="문서 기준값을 detector/pixel/v1/config.json locate에 반영")
    a = ap.parse_args()
    ONLY = a.only
    if a.fresh and CACHE.exists():
        import shutil
        shutil.copy(CACHE, CACHE.with_name("locate_cache_strict_rule.jsonl"))  # 이전 규칙 결과 보관
        shutil.copy(REPORT, REPORT.with_name("locate_evaluation_strict_rule.json")) if REPORT.exists() else None
        open(CACHE, "w").close()
    if a.collect and not collect(a.max_seconds):
        sys.exit()
    thr = report(a.max_fp)
    if a.write:
        cfg = load_config()
        cfg["locate"] = {"doc_max_rarity": round(thr, 5), "근거": f"튜닝용 정상 문서 {N_TUNE_DOCS}장 중 오탐 {a.max_fp*100:.0f}% 지점, outputs/reports/locate_evaluation.json"}
        json.dump(cfg, open(ROOT / "detector/pixel/v1/config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("반영: detector/pixel/v1/config.json")
