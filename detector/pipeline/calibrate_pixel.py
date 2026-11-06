"""픽셀 엔진 기준값 정하기: 정상 서류만 돌려 문서 점수 분포에서 두 기준을 잡는다 (호출 없음, 로컬).

    .venv/bin/python combine/calibrate_pixel.py testsets/개발지_v1            # 정답.csv 의 '정상' 서류만 씀
    .venv/bin/python combine/calibrate_pixel.py <정상 사진만 있는 폴더> --hi 0.02 --lo 0.10

  hi = 정상 오탐 2% 지점 → 넘으면 픽셀 '의심' (strong)
  lo = 정상 오탐 10% 지점 → 넘으면 픽셀 '확인필요'
결과: combine/pixel_thresholds.json. **평가셋으로 정하지 않는다** (평가셋으로 학습 금지) — 개발지나 따로 둔 정상 서류로.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from combine import engines as E  # noqa: E402
from combine.run import IMG_EXT, rows  # noqa: E402


def normals(base):
    if base.suffix == ".json":                                       # 정상 사진 경로 목록 (JSON 배열)
        return [Path(p) for p in json.loads(base.read_text(encoding="utf-8"))]
    key = base / "정답_채점용" / "정답.csv"
    if key.exists() and (base / "문제목록.csv").exists():
        ok = {r["문제_id"] for r in rows(key) if r["정답"] == "정상"}
        return [base / q["파일"] for q in rows(base / "문제목록.csv") if q["문제_id"] in ok]
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in IMG_EXT and "_mask" not in p.stem)


def score(p):
    try:
        return str(p), E.pixel_score(p)[0]
    except Exception:  # noqa: BLE001
        return str(p), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("폴더")
    ap.add_argument("--hi", type=float, default=0.02, help="'의심' 기준의 정상 오탐률")
    ap.add_argument("--lo", type=float, default=0.10, help="'확인필요' 기준의 정상 오탐률")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    base = (Path(a.폴더) if Path(a.폴더).exists() else REPO / a.폴더).resolve()
    if "평가셋" in base.name:
        raise SystemExit("평가셋으로 기준값을 정하면 안 됩니다 (평가셋으로 학습 금지). 개발지나 따로 둔 정상 서류를 주세요.")
    files = normals(base)
    print(f"정상 서류 {len(files)}장 픽셀 점수 계산 중…")
    with ProcessPoolExecutor(a.j) as ex:
        res = list(ex.map(score, files))
    s = np.array([v for _, v in res if v is not None])
    if len(s) < 30:
        print(f"주의: 점수를 낸 정상 서류가 {len(s)}장뿐이라 기준값이 흔들립니다 (100장 이상 권장)")
    thr = {"hi": round(float(np.quantile(s, 1 - a.hi)), 4), "lo": round(float(np.quantile(s, 1 - a.lo)), 4),
           "hi_fpr": a.hi, "lo_fpr": a.lo, "n": int(len(s)), "from": base.name, "check": "A11_다시그린칸"}
    E.THRESHOLDS.write_text(json.dumps(thr, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"기준값: 의심 > {thr['hi']} (정상 {a.hi:.0%}), 확인필요 > {thr['lo']} (정상 {a.lo:.0%}) → {E.THRESHOLDS}")


if __name__ == "__main__":
    main()
