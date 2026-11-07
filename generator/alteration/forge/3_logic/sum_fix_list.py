"""C3: 합계 맞추기 목록 만들기. 원본마다 표를 읽어(동현님 receipt.read_table, auto_receipt.total_of 그대로 사용)
'맨 아래 금액 = 위 금액들의 합'이 성립하는 열을 찾고, 항목 하나와 합계를 같이 바꾸는 목록을 만든다.
항목을 바꾼 만큼 합계도 바꿔서 열 합계는 맞는다 (내용 검사로는 이 열에서 못 잡는 위조).

python generator/alteration/forge/3_logic/sum_fix_list.py [--limit 초]   # 시간 제한마다 끊어서 여러 번 실행 (이어서 함)
결과: outputs/고도화/원본목록_합계맞춤.csv (forge_v2 run 에 FORGE_LIST 로 넣음), 검사 기록 outputs/고도화/_합계스캔.csv"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # generator/alteration/forge
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
OUT = ROOT / "outputs" / "고도화"
SCAN = OUT / "_합계스캔.csv"
LIST = OUT / "원본목록_합계맞춤.csv"
FIELDS = ["id", "original", "original_value", "forged_value", "x", "y", "w", "h", "kind",
          "original_value2", "forged_value2", "x2", "y2", "w2", "h2", "열항목수"]


def value_box(gray, c, ink_fn, m=2, value=None):
    """표 칸 안쪽에서 숫자만 감싸는 상자. value를 주면 오른쪽부터 그 값과 글자 구성이 맞는 덩어리만
    (앞에 '⑤' 같은 번호가 붙어 있어도 빼고 값만)."""
    x, y, w, h = c["x"] + m, c["y"] + m, c["w"] - 2 * m, c["h"] - 2 * m
    if w < 4 or h < 4:
        return None
    if value:
        import layout
        segs = sorted(layout._segments(gray, (x, y, w, h), ink_fn), key=lambda g: g["x0"])
        # 칸 안에서 값과 맞는 연속 덩어리 찾기. 앞뒤에 '②' 같은 번호가 붙어 있을 수 있어서 모든 구간을 본다.
        # 덩어리 수가 글자 수와 딱 맞는 구간을 먼저, 그다음 붙은 숫자 때문에 덩어리가 적은 구간
        # (덩어리가 더 많은 구간은 옆 번호까지 들어간 것이라 쓰지 않음, 10/30 '6,000②')
        n = len(value)
        hs = [g["y1"] - g["y0"] for g in segs if g["kind"] == "digit"]
        dh = float(np.median(hs)) if hs else 10.0
        # 숫자 줄 아래·위로 떨어진 점·선 부스러기는 뺌 (쉼표는 아래로 조금만 내려옴, 10/30 auto00031 '13,160' 밑 점)
        dg = [g for g in segs if g["kind"] == "digit" and g["y1"] - g["y0"] >= 0.7 * dh]
        if dg:
            top, bot = float(np.median([g["y0"] for g in dg])), float(np.median([g["y1"] for g in dg]))
            segs = [g for g in segs if g["kind"] == "digit" or (g["y0"] < bot + 0.25 * dh and g["y1"] > top - 0.25 * dh)]

        def tight(k, e):   # 값 안 글자 사이 틈은 좁음. 넓은 틈이 있으면 옆 번호('②')가 섞인 구간 (10/30 '6,000②')
            return all(segs[i + 1]["x0"] - segs[i]["x1"] <= 0.8 * dh for i in range(k, e - 1))
        wins = [(k, e) for k in range(len(segs)) for e in range(len(segs), k, -1) if tight(k, e)]
        found = None
        for exact in (True, False):
            for k, e in wins:
                L = e - k
                if (exact and L != n) or (not exact and L >= n):
                    continue
                if layout._fit_value(segs[k:e], value, strict=exact) is not None:
                    found = (k, e)
                    break
            if found:
                break
        if found:
            keep = segs[found[0]:found[1]]
            x0 = min(g["x0"] for g in keep); x1 = max(g["x1"] for g in keep)
            y0 = min(g["y0"] for g in keep); y1 = max(g["y1"] for g in keep)
            return int(x0), int(y0), int(np.ceil(x1 - x0)), int(y1 - y0)
        return None   # 값과 맞는 구간이 없으면 상자를 만들지 않음 (원래 숫자가 덜 지워지거나 옆 글자가 지워짐)
    return trim_box(gray, (x, y, w, h), ink_fn)


def trim_box(gray, box, ink_fn):
    """칸 안 획 중 오른쪽 끝부터 이어지는 글자 덩어리만 (금액은 오른쪽 정렬). 왼쪽에 큰 틈을 두고 떨어진
    '②' 같은 번호나 다른 글자는 뺀다 (10/23: 번호까지 값으로 잡혀 '5 ,250,784'처럼 나옴)."""
    import layout
    segs = layout._segments(gray, box, ink_fn)
    if not segs:
        return None
    segs.sort(key=lambda g: g["x0"])
    hs = [g["y1"] - g["y0"] for g in segs if g["kind"] == "digit"]
    dh = float(np.median(hs)) if hs else 10.0
    keep = [segs[-1]]
    for g in reversed(segs[:-1]):
        # 숫자 높이보다 넓은 빈칸이면 다른 글자 (숫자 사이, 쉼표 앞뒤 틈은 이보다 좁음)
        if keep[-1]["x0"] - g["x1"] > 1.0 * dh:
            break
        keep.append(g)
    x0 = min(g["x0"] for g in keep); x1 = max(g["x1"] for g in keep)
    y0 = min(g["y0"] for g in keep); y1 = max(g["y1"] for g in keep)
    return int(x0), int(y0), int(np.ceil(x1 - x0)), int(y1 - y0)


def scan_one(path, ink_fn):
    import auto_receipt as AR
    from receipt import read_table
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    table = read_table(gray)
    rng = random.Random(path.stem)
    cands = []
    for col in AR.columns(table):
        t = AR.total_of(col)
        if not t:
            continue
        total, items = t
        for it in items:
            if it["amount"] >= 1000 and it["conf"] >= AR.MIN_CONF:
                cands.append((total, it, len(items)))
    rng.shuffle(cands)
    for total, it, n in cands:
        new = AR._new_amount(it["amount"], rng)
        if not new:
            continue
        new_total = total["amount"] + new - it["amount"]
        if new_total <= 0 or len(str(new_total)) != len(str(total["amount"])):
            continue
        b1, b2 = value_box(gray, it, ink_fn), value_box(gray, total, ink_fn)
        if not b1 or not b2:
            continue
        # 상자 안 글자 덩어리가 값과 맞을 때만 (상자가 값 일부만 잡으면 원래 숫자가 덜 지워짐)
        import layout
        import forge_v2 as F
        ok = True
        for b, v in ((b1, AR.fmt(it["amount"])), (b2, AR.fmt(total["amount"]))):
            bx = F.label_box({"x": b[0], "y": b[1], "w": b[2], "h": b[3]}, gray.shape)
            if layout._fit_value(layout._segments(gray, bx, ink_fn), v) is None:
                ok = False
        if not ok:
            continue
        return {"id": f"sum_{path.stem}", "original": _rel(path),
                "original_value": AR.fmt(it["amount"]), "forged_value": AR.fmt(new),
                "x": b1[0], "y": b1[1], "w": b1[2], "h": b1[3], "kind": "금액",
                "original_value2": AR.fmt(total["amount"]), "forged_value2": AR.fmt(new_total),
                "x2": b2[0], "y2": b2[1], "w2": b2[2], "h2": b2[3], "열항목수": n}
    return None


def retrim():
    """이미 찾은 문서만 다시 검사해서 목록을 새로 씀 (상자 다듬는 규칙을 바꿨을 때)."""
    import forge_v2 as F
    old = list(csv.DictReader(open(LIST, encoding="utf-8-sig")))
    rows = []
    for r in old:
        n = scan_one(ROOT / r["original"], F.stroke_only)
        if n:
            rows.append(n)
    with open(LIST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("상자 다시 다듬음", len(rows))


def _rel(path):
    """저장소 안이면 상대 경로, 밖(처음 보는 사진 폴더)이면 절대 경로."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=0, help="이 초가 지나면 멈춤 (다음에 이어서)")
    ap.add_argument("--src", default="dataset/original/bankbook", help="원본 사진 폴더 (저장소 밖 절대 경로도 됨)")
    ap.add_argument("--name", default=None, help="결과 목록 이름 (다른 폴더를 검사할 때 섞이지 않게)")
    ap.add_argument("--retrim", action="store_true", help="목록 상자만 다시 다듬기")
    a = ap.parse_args()
    global SCAN, LIST
    if a.name:
        SCAN, LIST = OUT / f"_스캔_{a.name}.csv", OUT / f"원본목록_{a.name}.csv"
    if a.retrim:
        return retrim()
    import forge_v2 as F
    done = set()
    if SCAN.exists():
        done = {r["file"] for r in csv.DictReader(open(SCAN, encoding="utf-8-sig"))}
    files = sorted(p for p in (ROOT / a.src).expanduser().iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    t0 = time.time()
    new_scan = not SCAN.exists()
    new_list = not LIST.exists()
    with open(SCAN, "a", encoding="utf-8-sig", newline="") as fs, open(LIST, "a", encoding="utf-8-sig", newline="") as fl:
        ws = csv.DictWriter(fs, fieldnames=["file", "found"])
        wl = csv.DictWriter(fl, fieldnames=FIELDS)
        if new_scan:
            ws.writeheader()
        if new_list:
            wl.writeheader()
        n = 0
        for p in files:
            if p.name in done:
                continue
            if a.limit and time.time() - t0 > a.limit:
                break
            try:
                r = scan_one(p, F.stroke_only)
            except Exception as e:
                r = None
            if r:
                wl.writerow(r)
            ws.writerow({"file": p.name, "found": "Y" if r else ""})
            fs.flush(); fl.flush()
            n += 1
    total = len(done) + n
    found = sum(1 for _ in csv.DictReader(open(LIST, encoding="utf-8-sig")))
    print(f"검사 {total}/{len(files)}, 합계 맞추기 가능 {found}")


if __name__ == "__main__":
    main()
