"""[역할] 글꼴 파일 만들기: 서식·값을 그릴 글꼴 4종의 서브셋
[쓰는 곳] 사람이 글꼴을 바꿀 때 실행
[입력 → 출력] assets/fonts/src → assets/fonts/rebuild

방식 2(rebuild)용 글꼴 파일 생성: config/fonts.yaml 의 글꼴마다 Regular/Bold 서브셋 TTF를 만듦.

원본: assets/fonts/src/ (Google Fonts 배포 TTF, 모두 SIL Open Font License)
  1. 가변 폰트는 굵기 400(Regular), 700(Bold) 고정 인스턴스를 뽑고
  2. KS X 1001(EUC-KR) 완성형 한글 2,350자 + 특수문자 + ASCII + 서식에 쓰인 글자만 남김
     → PDF에 넣는 폰트 용량을 줄임
결과: assets/fonts/rebuild/{글꼴키}-Regular.ttf, {글꼴키}-Bold.ttf
※ Noto Sans KR의 OTF(CID-keyed CFF) 판은 PyMuPDF에서 한글 글리프가 깨져 TrueType 판을 사용함.
실행: .venv/bin/python tools/make_font_subset.py
"""
from pathlib import Path

import fitz
import yaml
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "fonts" / "src"
DST = ROOT / "assets" / "fonts" / "rebuild"
DST.mkdir(parents=True, exist_ok=True)

# 남길 글자: ASCII + 완성형(한자 제외) + 두 서식에 실제로 쓰인 글자 + 체크 표시 기호
chars = set(chr(c) for c in range(0x20, 0x7F))
for hi in range(0xA1, 0xFF):
    for lo in range(0xA1, 0xFF):
        try:
            ch = bytes([hi, lo]).decode("euc-kr")
        except UnicodeDecodeError:
            continue
        if not ("一" <= ch <= "鿿"):
            chars.add(ch)
for form in (ROOT / "assets" / "forms").glob("*.pdf"):
    chars |= {c for c in fitz.open(form)[0].get_text() if c.strip()}
chars |= set("√■□V")

cfg = yaml.safe_load((ROOT / "config" / "fonts.yaml").read_text(encoding="utf-8"))
for key, fam in cfg["families"].items():
    for style in ("regular", "bold"):
        src_name, wght = fam[style]
        font = TTFont(SRC / src_name)
        if wght is not None:
            font = instancer.instantiateVariableFont(font, {"wght": wght})
            # 가변 폰트는 기본 인스턴스 이름(Thin 등)이 남으므로 실제 굵기에 맞게 이름을 고침
            fam_name = font["name"].getDebugName(1).split(" Thin")[0].split(" ExtraLight")[0]
            sty = style.capitalize()
            names = {1: fam_name, 2: sty, 4: f"{fam_name} {sty}", 6: f"{fam_name.replace(' ', '')}-{sty}",
                     16: fam_name, 17: sty}
            for nid, val in names.items():
                font["name"].setName(val, nid, 3, 1, 0x409)
                font["name"].setName(val, nid, 1, 0, 0)
            font["OS/2"].usWeightClass = wght
        opts = subset.Options()
        opts.layout_features = []
        opts.name_IDs = ["*"]
        sub = subset.Subsetter(opts)
        sub.populate(unicodes=[ord(c) for c in chars])
        sub.subset(font)
        dst = DST / f"{key}-{style.capitalize()}.ttf"
        font.save(dst)
        missing = [c for c in chars if c.strip() and not fitz.Font(fontfile=str(dst)).has_glyph(ord(c))]
        print(f"{dst.name:28s} {dst.stat().st_size / 1024:5.0f} KB  없는 글자: {''.join(sorted(missing)) or '-'}")
