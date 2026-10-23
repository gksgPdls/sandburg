"""동현님 조작 방식별로 1단계(배경) 개선 전후 비교.

동현님 샘플의 원본에 같은 조작을 두 번 한다. 고르는 칸과 새 값은 같게(같은 난수), 지우기만 다르게.
- 기존: 동현님 edit_ops.erase 그대로
- 1단계: forge_v2.erase_reuse (표 선 보호, 잡음 크기를 주변 잔결에 맞춤)
동현님 원래 샘플과는 고른 칸이 다를 수 있다(원래는 여러 원본을 이어서 같은 난수로 돌렸기 때문).
이 환경에 맥 폰트가 없으면 두 결과 모두 비슷한 고딕체로 쓴다(비교는 공정).

    python scripts/generate/forge_v2_teammate.py run [--max-seconds 160]   (이어서 실행 가능)
    python scripts/generate/forge_v2_teammate.py report

결과: outputs/고도화/01_배경_팀원조작별/<조작>/{기존,1단계,비교이미지}, 점수.csv, 요약.csv
"""
import argparse
import copy
import csv
import os
import random
import re
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_v2 as F  # noqa: E402  (edit_ops 경로와 폰트 설정 포함)
import edit_ops  # noqa: E402
import auto_receipt as AR  # noqa: E402
import pick_session  # noqa: E402
import auto_diag  # noqa: E402

SAMPLES = F.ROOT / "dataset" / "samples"
OUT = F.OUT_ROOT / "01_배경_팀원조작별"
# 지우기(erase)를 쓰는 조작만. copymove, add_diag, add_text, edms, normal은 지우기가 없어 1단계와 무관
TARGET = {"amount_sum_break": "sum_break", "amount_digit": "digit", "amount_sum_fixed": "sum_fixed",
          "date_shift": "date_shift", "date_conflict": "date_conflict", "date_copy": "date_copy",
          "digit_combo": "digit_combo", "kcd_swap": "kcd"}
ORIG_ERASE = edit_ops.erase


def erase_v2(img, box, seed=0):
    _, m = F.erase_reuse(img, box, seed)
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    x, y = box[0], box[1]
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def set_erase(fn):
    edit_ops.erase = fn
    pick_session.erase = fn


def samples():
    with open(F.ROOT / "dataset" / "labels_all.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] in TARGET]
    return rows


def one(row):
    sid, method = row["id"], row["method"]
    op = TARGET[method]
    d = OUT / method
    if (d / "1단계" / f"{sid}.jpg").exists() or (d / "실패" / sid).exists():
        return sid, "skip"
    img = cv2.imread(str(SAMPLES / row["original"]))
    seed = zlib.crc32(sid.encode())
    if op == "kcd":
        up, codes, k = auto_diag.orient(img)
    else:
        up, table, k = AR.orient(img, dates=op in AR.DATE_OPS)
    results = {}
    for name, fn in (("기존", ORIG_ERASE), ("1단계", erase_v2)):
        set_erase(fn)
        im = up.copy()
        if op == "kcd":
            info = auto_diag.try_swap(im, list(codes), random.Random(seed)) if codes else None
        else:
            info = AR.FUNCS[op](im, copy.deepcopy(table), random.Random(seed))
        if not info:
            break
        mask = AR.changed_mask(up, im)
        results[name] = (np.ascontiguousarray(np.rot90(im, -k)), np.ascontiguousarray(np.rot90(mask, -k)),
                         info["description"])
    set_erase(ORIG_ERASE)
    if len(results) < 2:
        (d / "실패").mkdir(parents=True, exist_ok=True)
        (d / "실패" / sid).write_text("조작 실패 (칸을 못 찾음 등)", encoding="utf-8")
        return sid, "fail"
    for name, (im, mask, desc) in results.items():
        (d / name).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / name / f"{sid}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(d / name / f"{sid}_mask.png"), mask)
        (d / name / f"{sid}.txt").write_text(desc, encoding="utf-8")
    return sid, "ok"


def cmd_run(a):
    rows = samples()
    t0 = time.time()
    done = 0
    with Pool(4) as pool:
        for sid, st in pool.imap_unordered(one, rows):
            done += 1
            print(f"{done}/{len(rows)} {sid} {st}", flush=True)
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                pool.terminate()
                print("시간 제한으로 중단 (다시 실행하면 이어서 함)")
                return


def cmd_report(a):
    import forge_v2_compare as C
    from detector import detect
    rows = samples()
    scores = []
    for row in rows:
        sid, method = row["id"], row["method"]
        d = OUT / method
        if not (d / "1단계" / f"{sid}.jpg").exists():
            continue
        orig = Image.open(SAMPLES / row["original"]).convert("RGB")
        og = np.asarray(orig.convert("L"))
        tiles = [("원본", orig)]
        boxes = []
        for name in ("기존", "1단계"):
            im = Image.open(d / name / f"{sid}.jpg").convert("RGB")
            mpath = d / name / f"{sid}_mask.png"
            m = np.asarray(Image.open(mpath).convert("L")) > 0
            if not m.any():
                continue
            box = C._mask_box(mpath)
            boxes.append(box)
            tiles.append((f"{name} 팀원 방식" if name == "기존" else "1단계 적용 팀원 방식", im))
            res = detect(im, mask=Image.open(mpath))
            ref = detect(orig, mask=Image.open(mpath))
            bg, bg0 = C.background_texture(np.asarray(im.convert("L")), box), C.background_texture(og, box)
            nz, nz0 = res["methods"]["noise"]["score"], ref["methods"]["noise"]["score"]
            scores.append({"method": method, "id": sid, "방식": name,
                           "배경결_원본대비": round(bg / bg0, 3) if bg0 else "",
                           "노이즈_원본대비": round(nz / nz0, 3) if nz0 else "",
                           "탐지기_의심": "Y" if res["verdict"]["suspicious"] else ""})
        if not boxes:
            continue
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        mg = 30
        crop = (max(x0 - mg, 0), max(y0 - mg, 0), min(x1 + mg, orig.width), min(y1 + mg, orig.height))
        scale = max(2, min(6, 110 // max(crop[3] - crop[1], 1) + 1))
        imgs = []
        for name, im in tiles:
            c = im.crop(crop)
            c = c.resize((c.width * scale, c.height * scale), Image.LANCZOS)
            imgs.append(C._label(c, name))
        sheet = Image.new("RGB", (max(i.width for i in imgs), sum(i.height + 6 for i in imgs)), (200, 200, 200))
        yy = 0
        for i in imgs:
            sheet.paste(i, (0, yy)); yy += i.height + 6
        (d / "비교이미지").mkdir(exist_ok=True)
        sheet.save(d / "비교이미지" / f"{sid}.jpg", quality=92)
    with open(OUT / "점수.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scores[0])); w.writeheader(); w.writerows(scores)
    summary = []
    print(f"{'조작':<18}{'장수':>4}  {'배경결(기존→1단계)':>18}  {'노이즈(기존→1단계)':>18}  {'탐지기 의심(기존→1단계)':>20}")
    for method in TARGET:
        s = [r for r in scores if r["method"] == method]
        if not s:
            continue
        def med(name, key):
            v = [r[key] for r in s if r["방식"] == name and r[key] != ""]
            return float(np.nanmedian(v)) if v else float("nan")
        def sus(name):
            return sum(r["탐지기_의심"] == "Y" for r in s if r["방식"] == name)
        n = len([r for r in s if r["방식"] == "1단계"])
        row = {"조작": method, "장수": n,
               "배경결_기존": round(med("기존", "배경결_원본대비"), 3), "배경결_1단계": round(med("1단계", "배경결_원본대비"), 3),
               "노이즈_기존": round(med("기존", "노이즈_원본대비"), 3), "노이즈_1단계": round(med("1단계", "노이즈_원본대비"), 3),
               "탐지기의심_기존": sus("기존"), "탐지기의심_1단계": sus("1단계")}
        summary.append(row)
        print(f"{method:<18}{n:>4}  {row['배경결_기존']:>8.3f} → {row['배경결_1단계']:<8.3f}  "
              f"{row['노이즈_기존']:>8.3f} → {row['노이즈_1단계']:<8.3f}  {row['탐지기의심_기존']:>9} → {row['탐지기의심_1단계']}")
    with open(OUT / "요약.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--max-seconds", type=float, default=None)
    sp.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
