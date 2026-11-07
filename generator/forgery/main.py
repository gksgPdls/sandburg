"""[역할] 실행 시작점. 시작 점검 후 사건을 N건 만들어 진단서·계산서 PDF/PNG 와 data.json 을 저장
[쓰는 곳] 사용자가 직접 실행
[입력 → 출력] config/*.yaml → out/case_xxxxx/

가상 진단서(별지 제5호의2) + 진료비 계산서·영수증(별지 제6호, 외래·입원) 대량 생성기.

사용법 (공식 서식 레이아웃을 시스템 출력처럼 한 글씨체로 다시 그려서 생성):
  .venv/bin/python main.py                 # 10건 생성 → out/
  .venv/bin/python main.py --n 100         # 100건 생성
  .venv/bin/python main.py --n 5 --seed 7  # 같은 seed면 같은 결과 재현
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

import yaml

from docgen import ids, kcd, pricing
from docgen.generate import make_case, pick, pick_variant
from docgen.render_rebuild import FAMILY_WEIGHTS, render

# 결과 폴더
DEFAULT_OUT = "out"

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def to_json(o):
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="생성할 사건(case) 수")
    ap.add_argument("--seed", type=int, default=None, help="난수 시드 (재현용)")
    ap.add_argument("--out", type=Path, default=ROOT / DEFAULT_OUT, help="출력 폴더 (기본: out/)")
    ap.add_argument("--dpi", type=int, default=200, help="PNG 해상도")
    args = ap.parse_args()

    patient, codebook, variants = load("patient.yaml"), load("codebook.yaml"), load("variants.yaml")
    if not ids.rrn_is_valid(patient["rrn"]):
        raise SystemExit(f"patient.yaml 주민번호 검증번호 오류: {patient['rrn']}")
    kcd.check_diagnoses(codebook)       # 진단: 상병마스터 완전코드·공식 병명
    pricing.check_codebook(codebook)    # 단가: 진료기간의 모든 단가표에 공식 단가

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    args.out.mkdir(parents=True, exist_ok=True)
    start = len(list(args.out.glob("case_*")))
    for i in range(args.n):
        case_id = f"case_{start + i + 1:05d}"
        rng = random.Random(f"{seed}-{i}")
        var = {k: pick_variant(rng, spec) for k, spec in variants.items()}
        case = make_case(rng, case_id, patient, codebook, var)
        # 글씨체: 데이터용 난수와 분리된 별도 난수로 뽑음 → 글씨체 후보를 바꿔도 같은 seed의 데이터는 그대로
        case["font"] = pick(random.Random(f"{seed}-{i}-font"), FAMILY_WEIGHTS)

        d = args.out / case_id
        d.mkdir(exist_ok=True)
        render("diagnosis", case, d / "진단서", args.dpi)
        render("receipt", case, d / "진료비계산서영수증", args.dpi)
        (d / "data.json").write_text(
            json.dumps({"seed": seed, "index": i, **case}, ensure_ascii=False, indent=2, default=to_json),
            encoding="utf-8")
        r = case["receipt"]
        print(f"{case_id}  {case['hospital']['name']:<16} {case['department']:<5} "
              f"{case['dates']['visit']} {'입원' if case['admitted'] else '외래'}  ⑥{r['grand_total']:>9,}  ⑧{r['patient_total']:>8,}  [{case['font']}]")
    print(f"\n완료: {args.n}건 → {args.out}  (seed={seed})")


if __name__ == "__main__":
    main()
