"""forge_v2 결과 비교: 확대 비교 이미지와 점수.csv.

비교 이미지: 원본 | 이전 단계 폴더들 | 이번 단계 폴더들 을 같은 위치로 잘라 나란히 놓는다.
점수
- 배경결_원본대비: 위조 칸 안 종이(글자 아닌 픽셀)의 잔결 / 바로 바깥 종이의 잔결을, 원본 같은 자리 값으로 나눈 것.
  1이면 원본과 같은 결, 1보다 작으면 단색 박스처럼 매끈함
- 노이즈_원본대비: 내 탐지기 노이즈 비율을 원본 같은 자리 값으로 나눈 것. 1이면 원본과 같음
- 탐지기: 내 탐지기(detector) 종합 판정
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import edit_ops
from detector import detect

LABEL_FONT = next((p for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                               "/System/Library/Fonts/AppleSDGothicNeo.ttc") if Path(p).exists()), None)
ROOT = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/generate
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)


NICE = {"내방식": "내 방식", "동현님_write": "팀원 방식(폰트로 쓰기)", "동현님_copy": "팀원 방식(글자 복사)",
        "AI": "내 방식(AI)", "재사용_write": "팀원 방식(폰트로 쓰기)", "재사용_copy": "팀원 방식(글자 복사)",
        "AI_블렌딩있음": "내 방식(AI) + SD 블렌딩", "AI_윈도우폰트": "내 방식(AI, 윈도우 글꼴)",
        "AI_최종": "내 방식 최종(1+4단계)",
        "AI_반듯": "반듯하게 쓰기 (지금)", "AI_기울기": "기울기 맞춰 쓰기 (B2)", "AI_원근": "원근까지 맞춰 쓰기 (B2-2)",
        "AI_B없음": "B3~B6 없음 (원근까지)", "AI_B전부": "B3~B6 적용 (질감, 그림자, 휘어짐, jpg 저장)"}


def _folders(root, name):
    d = root / name
    if not d.exists():
        return []
    stage = "기존" if name.startswith("00") else ("최종" if name.startswith("최종") else name.split("_", 1)[0].lstrip("0") + "단계 적용")
    return [(f"{stage} {NICE.get(p.name, p.name)}", p) for p in sorted(d.iterdir()) if p.is_dir() and not p.name.startswith(("비교", "_"))]


def _file(folder, sid):
    for ext in (".png", ".jpg"):
        p = folder / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def _mask_box(path):
    m = np.asarray(Image.open(path).convert("L")) > 0
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _mask_boxes(path, gap=6):
    """마스크의 떨어진 덩어리마다 상자 (가까운 것은 합침)."""
    m = (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)
    m = cv2.dilate(m, np.ones((gap, gap), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = [(int(x + gap // 2), int(y + gap // 2), int(w - gap), int(h - gap)) for x, y, w, h, a in st[1:]]
    return sorted(out, key=lambda b: (b[1], b[0])) or [_mask_box(path)]


def background_texture(gray, box):
    x, y, w, h = box
    inside = gray[y:y + h, x:x + w]
    m_in = edit_ops.ink_mask(inside) == 0
    r = 8
    y0, y1, x0, x1 = max(y - r, 0), y + h + r, max(x - r, 0), x + w + r
    ring = gray[y0:y1, x0:x1].copy().astype(np.float32)
    ring_mask = edit_ops.ink_mask(gray[y0:y1, x0:x1]) == 0
    ring_mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    # 밝기 기울기(조명)를 빼고 잔결만 본다
    def resid(a):
        a = a.astype(np.float32)
        return a - cv2.GaussianBlur(a, (0, 0), 3)
    s_in = resid(inside)[m_in].std() if m_in.sum() > 10 else np.nan
    s_out = resid(gray[y0:y1, x0:x1])[ring_mask].std() if ring_mask.sum() > 10 else np.nan
    return float(s_in / s_out) if s_out and s_out > 1e-6 else float("nan")


def _label(img, text):
    pad = 22
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, pad))
    f = ImageFont.truetype(LABEL_FONT, 14) if LABEL_FONT else None
    ImageDraw.Draw(canvas).text((4, 2), text, fill="black", font=f)
    return canvas


def _changes(folder):
    p = folder / "생성기록.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["id"]: r.get("변경", "") for r in csv.DictReader(f)}


def _pair(orig, im, box, label):
    """같은 위치의 원본 | 결과를 나란히 놓은 한 줄."""
    x, y, w, h = box
    mg = 25
    mx = max(mg, (200 - w) // 2)  # 너무 좁게 잘리지 않게 가로는 최소 200px
    crop = (max(x - mx, 0), max(y - mg, 0), min(x + w + mx, orig.width), min(y + h + mg, orig.height))
    scale = max(2, min(5, 100 // max(crop[3] - crop[1], 1) + 1))
    a = orig.crop(crop); b = im.crop(crop)
    a = a.resize((a.width * scale, a.height * scale), Image.LANCZOS)
    b = b.resize((b.width * scale, b.height * scale), Image.LANCZOS)
    row = Image.new("RGB", (max(a.width * 2 + 8, 560), a.height), (230, 230, 230))
    row.paste(a, (0, 0)); row.paste(b, (a.width + 8, 0))
    return _label(row, label)


def compare(out_root, rows, out, prev):
    """방식마다 자기 위조 위치를 잘라 '원본 | 결과'로 나란히 놓는다 (방식마다 고친 칸이 다를 수 있음)."""
    cols = _folders(out_root, prev) + _folders(out_root, out)
    changes = {name: _changes(p) for name, p in cols}
    sheet_dir = out_root / out / "비교이미지"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for row in rows:
        sid = row["id"]
        orig = Image.open(ROOT / row["original"]).convert("RGB")
        imgs = []
        for name, p in cols:
            f, mpath = _file(p, sid), p / f"{sid}_mask.png"
            ch = changes[name].get(sid, "")
            if f is None or not mpath.exists():
                imgs.append(_label(Image.new("RGB", (400, 30), "white"), f"{name}: {ch or '없음'}"))
                continue
            im = Image.open(f).convert("RGB")
            box = _mask_box(mpath)
            short = ch.split(": ", 1)[-1][:60]
            parts = _mask_boxes(mpath)
            for k, pb in enumerate(parts):   # 한 이미지에서 여러 칸을 고친 경우(합계 맞추기 등) 칸마다 한 줄씩
                tag = f" [{k + 1}/{len(parts)}칸]" if len(parts) > 1 else ""
                imgs.append(_pair(orig, im, pb, f"{name}{tag}  ({short})   왼쪽 원본 | 오른쪽 결과"))
            gray = np.asarray(im.convert("L"))
            # 탐지기는 고친 칸마다 따로 (10/30: 여러 칸을 상자 하나로 묶으면 문서 전체가 한 상자가 되어
            #  고치지 않은 원본도 '의심'이 나오고, 칸 하나가 이상해도 안 보임)
            ratios, votes, bad = [], 0, []
            for k, pb in enumerate(parts):
                res = detect(im, bbox=pb, padding=0)
                ref = detect(orig, bbox=pb, padding=0)
                n1, n0 = res["methods"]["noise"]["score"], ref["methods"]["noise"]["score"]
                if n1 is not None and n0:
                    ratios.append(n1 / n0)
                votes = max(votes, res["verdict"]["votes"])
                if res["verdict"]["suspicious"]:
                    why = [m["name"].split("_")[-1] for m in res["methods"].values() if m.get("flagged")]
                    bad.append(f"{k + 1}칸({','.join(why)})")
            bg, bg0 = background_texture(gray, box), background_texture(np.asarray(orig.convert("L")), box)
            scores.append({"id": sid, "kind": row["kind"], "방식": name, "변경": ch,
                           "배경결_원본대비": round(bg / bg0, 3) if bg0 else "",
                           "노이즈_원본대비": round(float(min(ratios)), 3) if ratios else "",
                           "탐지기_투표": votes,
                           "탐지기_의심": "Y" if bad else "",
                           "의심칸": " ".join(bad)})
        W = max(i.width for i in imgs)
        sheet = Image.new("RGB", (W, sum(i.height + 10 for i in imgs)), (230, 230, 230))
        yy = 0
        for i in imgs:
            sheet.paste(i, (0, yy)); yy += i.height + 10
        sheet.save(sheet_dir / f"{sid}.jpg", quality=92)
    with open(out_root / out / "점수.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(scores[0]))
        wr.writeheader(); wr.writerows(scores)

    print("원본대비 값은 1에 가까울수록 원본과 같음")
    print(f"{'방식':<22}{'배경결 원본대비':>14}{'노이즈 원본대비':>14}{'탐지기 의심':>12}")
    summary = []
    for name, _ in cols:
        s = [r for r in scores if r["방식"] == name]
        if not s:
            continue
        bg = np.nanmedian([r["배경결_원본대비"] for r in s if r["배경결_원본대비"] != ""])
        nz = np.nanmedian([r["노이즈_원본대비"] for r in s if r["노이즈_원본대비"] != ""])
        sus = sum(r["탐지기_의심"] == "Y" for r in s)
        print(f"{name:<22}{bg:>14.3f}{nz:>14.3f}{sus:>9}/{len(s)}")
        summary.append({"단계폴더": out, "방식": name, "배경결_원본대비_중앙값": round(float(bg), 3),
                        "노이즈_원본대비_중앙값": round(float(nz), 3), "탐지기_의심": f"{sus}/{len(s)}"})
    path = out_root / "단계별점수.csv"
    old = []
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            old = [r for r in csv.DictReader(f) if r["단계폴더"] != out]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader(); wr.writerows(old + summary)
