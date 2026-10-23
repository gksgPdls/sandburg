"""3단계: 복사 흔적 지우기 (동현님 글자 복사 결과의 후처리).

동현님 copymove, digit_combo, date_copy 샘플(dataset/samples, 동현님 코드로 이미 만든 결과)을 그대로 두고,
붙여 넣은 자리만 다시 다듬어 새 파일로 저장한다. 목적: 같은 이미지 안에 똑같은 픽셀 블록이 두 군데 있지 않게.

붙인 블록에 하는 일 (강도 s, 0~1)
1) 미세 변형: 0.5*s px 이동, 0.3*s도 회전, 0.6*s% 크기 변화 (정수 픽셀 단위로 같은 블록이 안 나오게)
2) 노이즈 교체: 원래 잔결을 s만큼 걷어내고, 붙인 자리 주변 종이 잔결 크기의 새 잡음으로 채움
3) 밝기 맞춤: 블록의 종이 밝기를 붙인 자리 주변 종이에 s만큼 맞춤 (다른 색 칸에서 가져온 블록 대비)
가장자리 2px은 서서히 섞어서 경계가 생기지 않게 한다.

평가 (강도별)
- 동현님 복사 탐지기(eval_baseline.copymove_detect): 같은 이동 벡터의 블록 쌍이 8개 이상이면 복사로 판정
- 같은 픽셀 비율: 붙인 블록과 원래 블록을 ±1px 범위에서 가장 잘 맞춰 놓고, 밝기 차이 2 이하인 픽셀 비율
- 내 탐지기(detector) 종합 판정

    python scripts/generate/copy_trace.py run       (강도 0, 0.25, 0.5, 0.75, 1.0 모두)
    python scripts/generate/copy_trace.py report    (튜닝표와 비교 이미지)
결과: outputs/고도화/03_복사흔적/
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
sys.path.insert(0, str(ROOT))
import edit_ops  # noqa: E402
from labels import load_all, path_of  # noqa: E402

OUT = ROOT / "outputs" / "고도화" / "03_복사흔적"
METHODS = ("copymove", "digit_combo", "date_copy")
STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0)
DEFAULT_STRENGTH = 0.15   # 10/23 확정: 복사 탐지 20->0, 내 탐지기 +3, 눈으로 티 거의 없음 (튜닝결과.md)


def read(p):
    return cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)


def write(img, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(p, quality=95)


def blocks(row):
    """노트에서 (원래 블록, 붙인 블록) 좌표 (x, y, w, h). 동현님 코드의 붙이는 규칙 그대로 계산."""
    n = row.get("notes", "")
    m = re.search(r"src=(\d+),(\d+),(\d+),(\d+) dst=(\d+),(\d+)(?:,(\d+),(\d+))?", n)
    if m:
        sx, sy, sw, sh, dx, dy = map(int, m.groups()[:6])
        if row["method"] == "copymove":          # 칸 안쪽 블록을 통째로 복사
            return (sx, sy, sw, sh), (dx, dy, sw, sh)
        dw = int(m.group(7))                     # copy_block: 획 외곽 + 1px, 오른쪽 끝과 윗선 맞춤
        return (sx - 1, sy - 1, sw + 2, sh + 2), (dx + dw - sw - 1, dy - 1, sw + 2, sh + 2)
    m = re.search(r"--src-box (\d+) (\d+) (\d+) (\d+) --dst (\d+) (\d+)", n)
    if m:
        sx, sy, sw, sh, dx, dy = map(int, m.groups())
        return (sx, sy, sw, sh), (dx, dy, sw, sh)
    return None


def paper_ring(gray, box, r=6):
    x, y, w, h = box
    y0, x0 = max(y - r, 0), max(x - r, 0)
    sub = gray[y0:y + h + r, x0:x + w + r].astype(np.float32)
    keep = edit_ops.ink_mask(sub.astype(np.uint8)) == 0
    keep[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    resid = sub - cv2.GaussianBlur(sub, (0, 0), 2)
    if keep.sum() < 10:
        return float(np.median(sub)), 2.0
    return float(np.median(sub[keep])), float(resid[keep].std())


def harmonize(img, box, s, seed):
    """붙인 블록(box)을 강도 s로 다듬는다. img를 바꾼다."""
    if s <= 0:
        return
    rng = np.random.default_rng(seed)
    x, y, w, h = box
    pad = 3
    X0, Y0 = max(x - pad, 0), max(y - pad, 0)
    X1, Y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    sub = img[Y0:Y1, X0:X1].astype(np.float32)
    H, W = sub.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper_out, grain = paper_ring(gray, box)

    # 1) 미세 변형 (블록 가운데 기준)
    ang = rng.uniform(-0.3, 0.3) * s
    sc = 1 + rng.uniform(-0.006, 0.006) * s
    tx, ty = (rng.choice([-1, 1]) * rng.uniform(0.3, 0.5) * s, rng.choice([-1, 1]) * rng.uniform(0.3, 0.5) * s)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    M[:, 2] += (tx, ty)
    new = cv2.warpAffine(sub, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # 2) 노이즈 교체: 잔결을 s만큼 걷어내고 주변 종이 잔결 크기의 새 잡음
    base = cv2.GaussianBlur(new, (0, 0), 0.8)
    resid = new - base
    noise = rng.normal(0, 1, (H, W)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    noise *= grain / (noise.std() + 1e-6)
    new = base + (1 - s) * resid + s * noise[..., None]

    # 3) 밝기 맞춤: 블록 안 종이 밝기를 주변 종이 쪽으로 s만큼 (글자 획은 같은 비율로 함께)
    g = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    keep = edit_ops.ink_mask(g) == 0
    if keep.sum() > 10:
        paper_in = np.median(new[keep], axis=0)
        ring_bgr = np.median(img[max(y - 6, 0):y + h + 6, max(x - 6, 0):x + w + 6].reshape(-1, 3), axis=0)
        gain = 1 + s * (ring_bgr / np.maximum(paper_in, 1) - 1)
        new = new * gain

    # 가장자리 2px은 서서히 섞기
    f = np.zeros((H, W), np.float32)
    f[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = 1
    f = cv2.GaussianBlur(f, (0, 0), 1.0)[..., None]
    img[Y0:Y1, X0:X1] = np.clip(sub + f * (new - sub), 0, 255).astype(np.uint8)


def same_pixel_ratio(img, src, dst):
    """붙인 블록과 원래 블록을 ±1px에서 가장 잘 맞춘 뒤 밝기 차이 2 이하 픽셀 비율, 평균 차이."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sx, sy, w, h = src
    dx, dy = dst[0], dst[1]
    a = g[sy:sy + h, sx:sx + w]
    best = None
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            b = g[dy + oy:dy + oy + h, dx + ox:dx + ox + w]
            if b.shape != a.shape:
                continue
            d = np.abs(a - b)
            cand = (float((d <= 2).mean()), float(d.mean()))
            if best is None or cand[1] < best[1]:
                best = cand
    return best or (float("nan"), float("nan"))


def rows():
    return [r for r in sorted(load_all(), key=lambda r: r["id"]) if r.get("method") in METHODS]


def cmd_run(a):
    from eval_baseline import copymove_detect
    from detector import detect
    recs = []
    for i, r in enumerate(rows()):
        bl = blocks(r)
        if bl is None:
            print(r["id"], "좌표를 노트에서 못 읽음, 건너뜀")
            continue
        src, dst = bl
        tam = read(path_of(r, "tampered"))
        mask = Image.open(path_of(r, "mask"))
        for s in a.strengths:
            img = tam.copy()
            harmonize(img, dst, s, seed=i * 10 + int(s * 4))
            p = OUT / f"강도_{s:.2f}" / f"{r['id']}.jpg"
            write(img, p)
            img = read(p)   # 저장(jpg 95)까지 거친 결과로 평가
            hit, _, pairs = copymove_detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            same, diff = same_pixel_ratio(img, src, dst)
            res = detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), mask=mask)
            recs.append({"id": r["id"], "method": r["method"], "강도": s, "복사탐지": int(hit), "블록쌍": pairs,
                         "같은픽셀비율": round(same, 3), "평균차이": round(diff, 2),
                         "내탐지기": int(res["verdict"]["suspicious"])})
        print(f"{i + 1} {r['id']} {r['method']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "강도별결과.csv"
    old = []
    if path.exists():   # 이번에 돌린 강도만 바꾸고 나머지는 유지
        with open(path, encoding="utf-8-sig") as f:
            old = [r for r in csv.DictReader(f) if float(r["강도"]) not in a.strengths]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(old + recs)


def cmd_report(a):
    with open(OUT / "강도별결과.csv", encoding="utf-8-sig") as f:
        recs = list(csv.DictReader(f))
    lines = ["| 강도 | 장수 | 동현님 복사 탐지기에 걸림 | 같은 픽셀 비율(중앙값) | 평균 밝기 차이 | 내 탐지기에 걸림 |", "|---|---|---|---|---|---|"]
    summary = []
    for s in sorted({float(r["강도"]) for r in recs}):
        rs = [r for r in recs if float(r["강도"]) == s]
        n = len(rs)
        hit = sum(int(r["복사탐지"]) for r in rs)
        same = np.median([float(r["같은픽셀비율"]) for r in rs])
        diff = np.median([float(r["평균차이"]) for r in rs])
        mine = sum(int(r["내탐지기"]) for r in rs)
        lines.append(f"| {s} | {n} | {hit} | {same:.2f} | {diff:.1f} | {mine} |")
        for m in METHODS:
            mr = [r for r in rs if r["method"] == m]
            summary.append({"강도": s, "method": m, "장수": len(mr), "복사탐지": sum(int(r["복사탐지"]) for r in mr),
                            "같은픽셀비율": round(float(np.median([float(r["같은픽셀비율"]) for r in mr])), 3) if mr else "",
                            "내탐지기": sum(int(r["내탐지기"]) for r in mr)})
    print("\n".join(lines))
    with open(OUT / "튜닝표.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (OUT / "튜닝표.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 비교 이미지: 원본 | 기존(강도 0) | 0.25 | 0.5 | 1.0, 붙인 자리 확대
    from PIL import ImageDraw, ImageFont
    font_path = next((p for p in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, 14) if font_path else None
    (OUT / "비교이미지").mkdir(parents=True, exist_ok=True)
    for r in rows():
        bl = blocks(r)
        if bl is None:
            continue
        x, y, w, h = bl[1]
        mg = 20
        crop = (max(x - mg - 20, 0), max(y - mg, 0), x + w + mg + 20, y + h + mg)
        tiles = [("원본", Image.open(path_of(r, "original")).convert("RGB"))]
        for s in (0.0, 0.1, 0.15, 0.25, 0.5):
            tiles.append(("기존 (동현님)" if s == 0 else f"강도 {s}", Image.open(OUT / f"강도_{s:.2f}" / f"{r['id']}.jpg").convert("RGB")))
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        sc = max(2, min(6, 120 // max(ch, 1)))
        row_imgs = []
        for name, im in tiles:
            c = im.crop(crop).resize((cw * sc, ch * sc), Image.LANCZOS)
            canvas = Image.new("RGB", (c.width, c.height + 20), "white")
            canvas.paste(c, (0, 20))
            ImageDraw.Draw(canvas).text((4, 2), name, fill="black", font=font)
            row_imgs.append(canvas)
        sheet = Image.new("RGB", (sum(i.width + 6 for i in row_imgs), max(i.height for i in row_imgs)), (200, 200, 200))
        xx = 0
        for i in row_imgs:
            sheet.paste(i, (xx, 0)); xx += i.width + 6
        sheet.save(OUT / "비교이미지" / f"{r['id']}_{r['method']}.jpg", quality=92)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--strengths", type=float, nargs="+", default=list(STRENGTHS))
    sp.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
