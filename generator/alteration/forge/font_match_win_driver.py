

# ===================== 여기부터 윈도우 실행용 =====================
# 이 파일은 forge_v2.py export-win 이 font_match.py 내용 + 아래 코드를 합쳐 만든다.
# 실행: 04_윈도우입력 폴더에서  python font_match_win.py
# 결과: 옆 폴더 04_윈도우결과/AI_윈도우폰트/ ({id}.png, {id}_mask.png, 생성기록.csv)
import csv as _csv
import json as _json
import sys as _sys

_WIN = Path(r"C:\Windows\Fonts")
_LOCAL = Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts"   # 사용자가 따로 설치한 글꼴
CANDIDATE_LIST = [
    (str(_WIN / "gulim.ttc"), range(0, 4)),      # 굴림, 굴림체, 돋움, 돋움체
    (str(_WIN / "batang.ttc"), range(0, 4)),     # 바탕, 바탕체, 궁서, 궁서체
    (str(_WIN / "malgun.ttf"), [0]), (str(_WIN / "malgunbd.ttf"), [0]), (str(_WIN / "malgunsl.ttf"), [0]),
    (str(_WIN / "arial.ttf"), [0]), (str(_WIN / "arialbd.ttf"), [0]), (str(_WIN / "ARIALN.TTF"), [0]),
    (str(_WIN / "times.ttf"), [0]), (str(_WIN / "timesbd.ttf"), [0]),
    (str(_WIN / "cour.ttf"), [0]), (str(_WIN / "courbd.ttf"), [0]),
    (str(_WIN / "tahoma.ttf"), [0]), (str(_WIN / "verdana.ttf"), [0]), (str(_WIN / "consola.ttf"), [0]),
    # 따로 설치해야 하는 무료 글꼴 (설치돼 있으면 후보에 들어감)
    (str(_WIN / "NanumGothic.ttf"), [0]), (str(_LOCAL / "NanumGothic.ttf"), [0]),
    (str(_WIN / "NanumGothicBold.ttf"), [0]), (str(_LOCAL / "NanumGothicBold.ttf"), [0]),
    (str(_WIN / "NanumMyeongjo.ttf"), [0]), (str(_LOCAL / "NanumMyeongjo.ttf"), [0]),
] + [(str(d / f), [0]) for d in (_WIN, _LOCAL) for f in (
    "NotoSansKR-Regular.ttf", "NotoSansKR-Medium.ttf", "NotoSansKR-Bold.ttf", "NotoSansKR-Light.ttf",
    "NotoSansKR-VariableFont_wght.ttf", "NotoSansKR-Regular.otf", "NotoSansKR-Bold.otf")]


def _ink_mask(gray_box):
    paper = float(np.median(gray_box))
    return (gray_box < paper - 35).astype(np.uint8) * 255


def stroke_only(gray_box):
    """표 선을 뺀 글자 획 마스크 (맥 forge_v2.stroke_mask와 같은 규칙)."""
    h, w = gray_box.shape
    m = _ink_mask(gray_box)
    lines = np.zeros_like(m)
    if w >= 8:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.7), 8), 1))
        lines |= cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if h >= 8:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.9), 8)))
        lines |= cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    m[lines > 0] = 0
    return m


def _read(p):   # 한글 경로 때문에 cv2.imread 대신 PIL
    return cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)


def _write(img, p):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(p)


def _main():
    here = Path(__file__).resolve().parent
    out = here.parent / "04_윈도우결과" / "AI_윈도우폰트"
    out.mkdir(parents=True, exist_ok=True)
    found = candidates()
    print(f"후보 글꼴 {len(found)}개: " + ", ".join(sorted({n for _, _, n in found})))
    if not found:
        _sys.exit("C:\\Windows\\Fonts 에서 후보 글꼴을 하나도 못 찾음")
    samples = _json.loads((here / "samples.json").read_text(encoding="utf-8"))
    log = []
    for i, s in enumerate(samples):
        sid = s["id"]
        orig = _read(here / "original" / f"{sid}.png")
        img = _read(here / "erased" / f"{sid}.png")
        bg = _read(here / "scoring_bg" / f"{sid}.png")
        og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        box = tuple(s["box"])
        gm = {k: s[k] for k in ("right", "left", "top", "height", "ink")}
        written, best, top = write_matched(img, bg, og, box, s["original_value"], s["forged_value"], gm,
                                           stroke_only, orig_bgr=orig, glyphs=[g[1:] for g in s["glyphs"]] or None)
        _write(img, out / f"{sid}.png")
        m = np.zeros(og.shape, np.uint8)
        for x, y, w, h in (box, written):
            m[max(y, 0):y + h, max(x, 0):x + w] = 255
        Image.fromarray(m).save(out / f"{sid}_mask.png")
        log.append({"id": sid, "steps": "1,4", "font": "windows", "변경": f'{s["original_value"]} → {s["forged_value"]}',
                    "폰트": best["name"], "굵기": best["weight"], "흐림": best["blur"], "글자색": best.get("글자색", ""),
                    "자간": round(best["tracking"], 2), "차이": round(best["score"], 2),
                    "후보2_3": " / ".join(f'{t["name"]}({t["score"]:.2f})' for t in top[1:3])})
        print(f"{i + 1}/{len(samples)} {sid}: {best['name']} (굵기 {best['weight']}, 흐림 {best['blur']})", flush=True)
    with open(out / "생성기록.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"완료: {out}")


if __name__ == "__main__":
    _main()
