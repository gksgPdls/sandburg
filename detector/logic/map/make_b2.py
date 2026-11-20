"""B2(진찰료 단가)·B6(진찰료 가산 ↔ 진료일) 예시용 수가 변조본. 외부 호출 없음.
회사 실제 문서에서 만든 것이므로 b2_fee/ 안에서만 쓴다 (git·외부 게시 금지).

진찰료 행의 공단부담 금액만 고친다 (글자 지우고 다시 쓰기는 팀 생성기 dataset/scripts/edit_ops.replace_text).
  inflate   공식 진찰료로 나올 수 없는 금액으로 부풀림        → B2 '공식 진찰료로 나올 수 없는 금액'
  upcode    평일 진찰료를 야간·공휴 가산 진찰료 금액으로 바꿈  → B6 '야간·공휴 가산 금액인데 평일이고 야간 표시 없음'
B2·진찰료 B6(b2_b6_consult)는 메인에서 꺼져 있어(단가 데이터 부족) 이 스크립트 안에서만 켠다.

    python make_b2.py [장수=10]
"""
import csv
import sys
from datetime import date
from pathlib import Path

import cv2

ROOT = Path.home() / "Desktop/SCRYN/3_탐지기"
sys.path.insert(0, str(ROOT / "sandburg-scryn-forgery/dataset/scripts"))
sys.path.insert(0, str(ROOT / "scryn-combine-web"))
import edit_ops  # noqa: E402
import auto_receipt as AR  # noqa: E402
import _compat  # noqa: E402,F401  (logic_detector 찾기)
from logic_detector.logic import inspect, refdata as R, rules  # noqa: E402

if rules.b2_b6_consult not in rules.RECEIPT_RULES:          # 이 실행에서만 켬
    rules.RECEIPT_RULES.append(rules.b2_b6_consult)

HERE = Path(__file__).resolve().parent
OUT = HERE / "b2_fee"
OUT.mkdir(exist_ok=True)
SAMPLES = Path.home() / "Desktop/SCRYN/2_데이터셋/변조_실서류_samples"
TYPES = ["clinic", "hospital", "general"]


def val(rd, k):
    v = rd.get(k)
    return v.get("value") if isinstance(v, dict) else v


def bx(rd, k):
    v = rd.get(k)
    return tuple(v["box"]) if isinstance(v, dict) and v.get("box") else None


def originals():
    seen = set()
    for lab in sorted(SAMPLES.glob("*/labels.csv")):
        for r in csv.DictReader(open(lab, encoding="utf-8-sig")):
            if "진단" in r["doc_type"] or not r.get("original"):
                continue
            p = lab.parent / r["original"]
            if p.exists() and p not in seen:
                seen.add(p)
                yield r["id"], p


def plan(rd):
    """원본 판독값 → (방식, 새 공단부담) 후보들. 진찰료가 공식 기본 단가와 맞는 외래 영수증만."""
    if val(rd, "visit.inpatient"):
        return []
    try:
        day = date.fromisoformat(str(val(rd, "visit.start")))
    except ValueError:
        return []
    s, n = val(rd, "rows.consult.self"), val(rd, "rows.consult.nhis")
    if not isinstance(s, int) or not isinstance(n, int) or n <= 0 or not bx(rd, "rows.consult.nhis") or R.price_table(day) is None:
        return []
    if day.weekday() >= 5 or R.is_holiday(day) is not False:
        return []
    opts = rules._consult_options(day, TYPES)
    labs = opts.get(s + n)
    if not labs or {k for *_, k in labs} != {"base"}:
        return []
    out = []
    t, kind, _ = labs[0]
    night = R.price(day, f"CONSULT_{kind}_{t}_NIGHT", t)
    if night and night - s > 0 and not val(rd, "visit.night"):
        out.append(("upcode", night - s, f"평일 {day} 진찰료 {s + n:,}원(기본) → {night:,}원(야간·공휴 가산 단가)"))
    for bump in (500, 700, 300, 3000, 2000, 5000):
        nn = n + bump
        if s + nn not in opts and len(f"{nn:,}") == len(f"{n:,}"):
            out.append(("inflate", nn, f"진찰료 공단부담 {n:,} → {nn:,} (진찰료 {s + n:,} → {s + nn:,}원, 공식 단가에 없는 금액)"))
            break
    return out


if __name__ == "__main__":
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows, got = [], {"upcode": 0, "inflate": 0}
    for oid, p in originals():
        if sum(got.values()) >= want:
            break
        r = inspect(str(p), "receipt")
        if any(f["rule"] in ("B2", "B6") and "진찰료" in f["title"] for f in r["findings"]):
            continue                                                 # 원본에서 이미 걸리면 예시로 못 씀
        rd = r.get("reading", {})
        for how, new_n, desc in plan(rd):
            if got[how] >= (want + 1) // 2:
                continue
            img0 = cv2.imread(str(p))
            up, table, k = AR.orient(img0)
            if k != 0:
                continue                                                # 돌린 사진은 좌표가 달라 건너뜀
            x, y, w, h = bx(rd, "rows.consult.nhis")
            cx, cy = x + w / 2, y + h / 2
            cells = [c for c in table if c.get("amount") == val(rd, "rows.consult.nhis") and c["x"] <= cx <= c["x"] + c["w"] and c["y"] <= cy <= c["y"] + c["h"]]
            if not cells:
                print("표에서 진찰료 공단부담 칸을 못 찾음", oid, flush=True)
                continue
            img = up
            before = img.copy()
            if edit_ops.replace_text(img, AR.box(cells[0]), f"{new_n:,}", seed=3) is None:
                print("글자 다시 쓰기 실패", oid, how, flush=True)
                continue
            bid = f"B2_{how}_{oid}"
            f_img = OUT / f"{bid}.jpg"
            cv2.imwrite(str(f_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            chk = inspect(str(f_img), "receipt")
            hit = [f for f in chk["findings"] if f["rule"] in ("B2", "B6") and ("진찰료" in f["title"])]
            if not hit:
                f_img.unlink()
                print("안 걸림, 버림", bid)
                continue
            import numpy as np
            diff = (np.abs(before.astype(int) - img.astype(int)).max(axis=2) > 0).astype("uint8") * 255
            cv2.imwrite(str(OUT / f"{bid}_mask.png"), diff)
            got[how] += 1
            rows.append({"id": bid, "doc_type": "진료비 영수증", "method": f"fee_{how}", "tampered": f_img.name,
                         "mask": f"{bid}_mask.png", "original": str(p), "description": desc,
                         "hit": " / ".join(f'{f["rule"]} {f["title"]}' for f in hit)})
            print(bid, desc, "→", rows[-1]["hit"], flush=True)
            break
    with open(OUT / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "doc_type", "method", "tampered", "mask", "original", "description", "hit"])
        w.writeheader(); w.writerows(rows)
    print(got, "→", OUT)
