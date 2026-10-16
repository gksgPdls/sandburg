#!/usr/bin/env python3
"""데이터셋에 로컬 기본 탐지기를 돌려 얼마나 잡히는지 잰다. 외부 AI·API 는 쓰지 않는다.

탐지기:
    copymove  블록 매칭. 8x8 블록(질감 있는 것만)의 특징을 정렬해 거의 같은 블록 쌍을 찾고,
              같은 이동 벡터(dx, dy)를 가진 쌍이 MIN_PAIRS 개 이상 모이면 복사로 판정한다.
              판정한 영역이 정답 마스크와 겹치면 '위치까지 맞춤'
    date      로컬 OCR 로 날짜 칸을 읽어, 한 칸 안에서 끝 날짜 < 시작 날짜면 위조로 판정

조작본(tampered)과 정상(정상 대조군의 original)에 돌린다.
결과: _work/eval/{탐지기}.csv  (이미지는 저장하지 않는다)

사용: eval_baseline.py copymove | date
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from labels import ROOT, load_all, path_of

OUT = ROOT / "_work" / "eval"
B = 8            # 블록 크기
MIN_STD = 15     # 이보다 밋밋한 블록(빈 종이)은 뺀다
MAX_DIFF = 3     # 특징(2x2 평균 16개) 최대 차이. 다시 촬영된 반복 글자는 이보다 크게 다르다
MIN_SHIFT = 6    # 이보다 가까운 쌍은 같은 자리로 본다
MIN_PAIRS = 8    # 같은 이동 벡터가 이만큼 모여야 복사로 본다


def copymove_detect(gray):
    """(판정, 예측 마스크, 최다 이동 벡터의 쌍 수)."""
    g = gray.astype(np.float32)
    H, W = g.shape
    mean2 = cv2.boxFilter(g, -1, (2, 2), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    m = cv2.boxFilter(g, -1, (B, B), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(g * g, -1, (B, B), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq - m * m, 0))[:H - B, :W - B]
    # 표 괘선 위의 블록은 선을 따라 밀어도 똑같아서 가짜 쌍을 만든다.
    # 가로·세로 양쪽으로 변화가 있는(글자 같은) 블록만 쓴다
    gx = cv2.boxFilter(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2, -1, (B, B), anchor=(0, 0))[:H - B, :W - B]
    gy = cv2.boxFilter(cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2, -1, (B, B), anchor=(0, 0))[:H - B, :W - B]
    two_d = np.minimum(gx, gy) > 0.35 * np.maximum(gx, gy)
    ys, xs = np.nonzero((std > MIN_STD) & two_d)
    if len(ys) == 0:
        return False, np.zeros_like(gray), 0
    feats = np.stack([mean2[ys + dy, xs + dx] for dy in range(0, B, 2) for dx in range(0, B, 2)], axis=1)
    q = np.round(feats).astype(np.int16)
    order = np.lexsort(q.T[::-1])
    q, ys, xs = q[order], ys[order], xs[order]
    shifts = {}
    for k in range(1, 4):
        d = np.abs(q[k:] - q[:-k]).max(axis=1)
        idx = np.nonzero(d <= MAX_DIFF)[0]
        dy, dx = ys[idx + k] - ys[idx], xs[idx + k] - xs[idx]
        far = (np.abs(dy) + np.abs(dx)) >= MIN_SHIFT
        for i, a, b in zip(idx[far], dy[far], dx[far]):
            key = (int(a), int(b)) if (a, b) > (0, 0) else (int(-a), int(-b))
            shifts.setdefault(key, []).append(i if (a, b) > (0, 0) else i + k)
            shifts[key].append(i + k if (a, b) > (0, 0) else i)
    pred = np.zeros_like(gray)
    best = 0
    for key, members in shifts.items():
        n = len(members) // 2
        best = max(best, n)
        if n >= MIN_PAIRS:
            for j in members:
                pred[ys[j]:ys[j] + B, xs[j]:xs[j] + B] = 255
    return best >= MIN_PAIRS, pred, best


def date_detect(img):
    from auto_receipt import orient
    from receipt import find_dates
    _, table, _ = orient(img, dates=True)
    for c in table:
        ds = find_dates(c["text"])
        if len(ds) >= 2:
            a, b = (ds[0]["y"], ds[0]["m"], ds[0]["d"]), (ds[1]["y"], ds[1]["m"], ds[1]["d"])
            if b < a:
                return True
    return False


def main():
    which = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(load_all(), key=lambda r: r["id"])
    out = []
    for i, r in enumerate(rows, 1):
        is_fake = r["target_axis"] != "none"
        img_path = path_of(r, "tampered") if is_fake else path_of(r, "original")
        img = cv2.imread(str(img_path))
        rec = {"id": r["id"], "method": r["method"], "axis": r["target_axis"], "fake": int(is_fake)}
        if which == "copymove":
            hit, pred, best = copymove_detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            rec.update(pred=int(hit), pairs=best, located="")
            if hit and is_fake and r.get("mask"):
                gt = cv2.imread(str(path_of(r, "mask")), 0) > 0
                rec["located"] = int(bool((gt & (pred > 0)).any()))
        else:
            rec.update(pred=int(date_detect(img)))
        out.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    path = OUT / f"{which}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{which}: {len(out)}장 → {path}")


if __name__ == "__main__":
    main()
