"""합계까지 다 맞춘 진찰료 뻥튀기 변조본 (B2 만 남는지 보기). 외부 호출 없음. b2_full/ 안에서만 쓴다.

진찰료 본인부담에 D 원을 더하고, 그 돈이 이어지는 칸을 모두 같은 만큼 고친다:
  진찰료 본인부담 · ① 본인부담 합계 · ⑥ 진료비 총액 · ⑧ 환자부담 총액 · ⑩ 납부할 금액 · ⑪ 카드(또는 현금) · ⑪ 합계
글자 고치기는 make_b6_dates._rewrite (팀 edit_ops 의 erase · write).

    python make_b2_full.py <원본 문서 번호> ...
"""
import csv
import sys
from pathlib import Path

import cv2

import _compat  # noqa: E402,F401  (logic_detector 찾기)
import make_b2 as M
from make_b6_dates import _rewrite

HERE = Path(__file__).resolve().parent
OUT = HERE / "b2_full"
OUT.mkdir(exist_ok=True)
D = 20000
FIELDS = ["rows.consult.self", "totals.t1", "grand_total", "patient_total", "to_pay", "paid.card", "paid.total"]
NAMES = {"rows.consult.self": "진찰료 본인부담", "totals.t1": "① 본인부담 합계", "grand_total": "⑥ 진료비 총액",
         "patient_total": "⑧ 환자부담 총액", "to_pay": "⑩ 납부할 금액", "paid.card": "⑪ 카드", "paid.total": "⑪ 합계"}
PICK = sys.argv[1:]          # 원본 문서 번호들 (예: python make_b2_full.py 문서번호1 문서번호2)

rows = []
for oid, p in M.originals():
    if oid not in PICK:
        continue
    r = M.inspect(str(p), "receipt")
    rd = r["reading"]
    img = cv2.imread(str(p))
    before = img.copy()
    changes = []
    import edit_ops, statistics
    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hs = [edit_ops.ink_bbox(g0, (b[0] - 2, b[1] + 1, b[2] + 4, b[3] - 2)) for b in (M.bx(rd, k) for k in FIELDS) if b]
    H = statistics.median([h[3] for h in hs if h])                     # 이 영수증 금액 글자의 보통 높이
    for k in FIELDS:
        v, b = M.val(rd, k), M.bx(rd, k)
        if not isinstance(v, int) or not b:
            print(oid, "칸 없음", k); break
        x, y, w, h = b
        if k == "totals.t1":                      # 칸 양옆의 ①·② 동그라미 숫자는 건드리지 않게 금액 쪽만
            x, w = x + int(0.2 * w), int(0.68 * w)
        if _rewrite(img, (x - 2, y - 2, w + 4, h + 4), f"{v + D:,}", 31, height=H) is None:
            print(oid, "다시 쓰기 실패", k); break
        changes.append(f"{NAMES[k]} {v:,} → {v + D:,}")
    else:
        bid = f"B2F_{oid}"
        cv2.imwrite(str(OUT / f"{bid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        import numpy as np
        diff = (np.abs(before.astype(int) - img.astype(int)).max(axis=2) > 0).astype("uint8") * 255
        cv2.imwrite(str(OUT / f"{bid}_mask.png"), diff)
        s, n = M.val(rd, "rows.consult.self"), M.val(rd, "rows.consult.nhis")
        rows.append({"id": bid, "doc_type": "진료비 영수증", "method": "fee_inflate_full", "tampered": f"{bid}.jpg", "mask": f"{bid}_mask.png",
                     "original": str(p), "description": f"진찰료 본인부담 {D:,}원 부풀리고 이어지는 합계 칸을 모두 같이 고침 "
                     f"(진찰료 {s + n:,} → {s + n + D:,}원): " + " / ".join(changes)})
        print(bid, rows[-1]["description"])
with open(OUT / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "doc_type", "method", "tampered", "mask", "original", "description"])
    w.writeheader(); w.writerows(rows)
