"""[역할] 공식 서식 PDF 를 같은 레이아웃으로 다시 그리고 값을 같은 글씨체로 기입해 PDF/PNG 저장
[쓰는 곳] main.py
[입력 → 출력] assets/forms/*.pdf, assets/fonts/rebuild → PDF/PNG

[방식 2: rebuild] 공식 서식 레이아웃을 그대로 '다시 그린' 뒤 값을 같은 글씨체로 기입.

(초기의 '공식 PDF 위에 값만 덧씌우는' 방식은 서식 글자와 값 글자의 글씨체가 섞이고
 원본 위에 레이어가 얹힌 구조가 되어 폐기함.)
실제 병원 문서는 한글(HWP)/EMR 서식에 값을 입력해 '한 번에' 출력하므로 서식 글자와 값이
같은 글씨체·같은 출력 과정에서 나옴. 이 방식은 그 형태를 재현함:

  1. 공식 PDF에서 선·칸(벡터)과 서식 문구의 글자별 위치·크기·색을 추출
  2. 빈 페이지에 선을 그대로 다시 그리고, 서식 문구를 글자 하나씩 원래 좌표에 다시 씀
  3. 값은 render.py 의 좌표표·기입 함수(draw_receipt / draw_diagnosis)로 기입
  → 칸·글자 배치는 공식 서식과 동일, 문서 전체(서식 문구 + 값)가 한 글씨체

글씨체: 사건(case)마다 config/fonts.yaml 후보 중 하나를 확률로 뽑아 그 사건의 두 문서에 공통 사용.
        글꼴에 없는 글자(예: '註')만 fallback 글꼴로 씀.
"""
import re
from pathlib import Path

import fitz  # PyMuPDF
import yaml

from .render import DOCS, FORMS, ROOT, Writer

FONT_DIR = ROOT / "assets" / "fonts" / "rebuild"     # tools/make_font_subset.py 로 생성
FONT_CFG = yaml.safe_load((ROOT / "config" / "fonts.yaml").read_text(encoding="utf-8"))
FAMILY_WEIGHTS = {k: v["weight"] for k, v in FONT_CFG["families"].items()}

# 제목 영역: 원본은 제목 글자를 벡터 외곽선으로도 그려 두었으므로 그 선은 빼고 글자만 굵게 다시 씀
TITLE_AREA = {
    "form6_receipt.pdf": fitz.Rect(110, 28, 485, 47),
    "form5-2_diagnosis.pdf": fitz.Rect(245, 86, 355, 112),
}

# 서식 상단의 파란색 개정일 표기 "<개정 2024. 7. 18.>" 등은 다시 그리지 않음
# (법령 서식 원문에만 있는 표기로, 실제 병원 시스템 출력물에는 넣지 않도록 요청)
SKIP_SPAN = re.compile(r"^\s*<\s*개정.*>\s*$")


class Family:
    """한 글씨체의 Regular/Bold + 글자가 없을 때 쓸 fallback."""

    def __init__(self, key: str, fallback=None):
        self.key = key
        self.label = FONT_CFG["families"][key]["label"]
        self.regular = fitz.Font(fontfile=str(FONT_DIR / f"{key}-Regular.ttf"))
        self.bold = fitz.Font(fontfile=str(FONT_DIR / f"{key}-Bold.ttf"))
        self.fallback = fallback

    def font_for(self, ch: str, bold: bool) -> fitz.Font:
        f = self.bold if bold else self.regular
        if not ch.strip() or f.has_glyph(ord(ch)):
            return f
        if self.fallback:
            return self.fallback.font_for(ch, bold)
        raise ValueError(f"글꼴 '{self.label}'에 '{ch}' 글자가 없음 → tools/make_font_subset.py 확인")


_families: dict = {}


def get_family(key: str) -> Family:
    if key not in _families:
        fb_key = FONT_CFG["fallback"]
        fallback = None if key == fb_key else get_family(fb_key)
        _families[key] = Family(key, fallback)
    return _families[key]


def put_text(page, pos, text, size, family: Family, bold=False, color=(0, 0, 0), x_scale=1.0):
    """글자 쓰기 (TextWriter: 한글·기호 유니코드를 정확히 매핑). 글꼴에 없는 글자만 fallback으로."""
    tw = fitz.TextWriter(page.rect, color=color)
    x, y = pos
    chunk, chunk_font = "", None
    for ch in list(text) + [None]:
        f = None if ch is None else family.font_for(ch, bold)
        if chunk and f is not chunk_font:
            tw.append((x, y), chunk, font=chunk_font, fontsize=size)
            x += chunk_font.text_length(chunk, fontsize=size)
            chunk = ""
        if f is not None:
            chunk, chunk_font = chunk + ch, f
    morph = (fitz.Point(pos), fitz.Matrix(x_scale, 1)) if x_scale < 0.98 else None
    tw.write_text(page, morph=morph)


def make_writer(family: Family):
    class FamilyWriter(Writer):
        """값 기입도 서식과 같은 글씨체로 (칸 맞춤 로직은 render.Writer 그대로)."""

        def __init__(self, page: fitz.Page):
            super().__init__(page, family.regular)

        def _put(self, pos, text, size, bold=False):
            put_text(self.page, pos, text, size, family, bold)
    return FamilyWriter


def _rgb(color_int: int) -> tuple:
    return ((color_int >> 16) & 255) / 255, ((color_int >> 8) & 255) / 255, (color_int & 255) / 255


def build_blank(form_name: str, family: Family) -> bytes:
    """공식 서식 PDF → 같은 레이아웃의 빈 서식(선 + 서식 문구)을 지정 글씨체로 새로 그려 PDF bytes로 반환."""
    src = fitz.open(FORMS / form_name)
    sp = src[0]
    out = fitz.open()
    page = out.new_page(width=sp.rect.width, height=sp.rect.height)
    title_area = TITLE_AREA[form_name]

    # 1) 선·칸·음영 다시 그리기 (원본 좌표·굵기·색 그대로)
    for d in sp.get_drawings():
        if title_area.contains(d["rect"]):
            continue
        shape = page.new_shape()
        for it in d["items"]:
            kind = it[0]
            if kind == "l":
                shape.draw_line(it[1], it[2])
            elif kind == "re":
                shape.draw_rect(it[1])
            elif kind == "qu":
                shape.draw_quad(it[1])
            elif kind == "c":
                shape.draw_bezier(it[1], it[2], it[3], it[4])
        shape.finish(color=d.get("color"), fill=d.get("fill"), width=d.get("width") or 0.5,
                     closePath=d.get("closePath", False), even_odd=d.get("even_odd", False),
                     fill_opacity=d.get("fill_opacity") or 1, stroke_opacity=d.get("stroke_opacity") or 1,
                     dashes=d.get("dashes"))
        shape.commit()

    # 2) 서식 문구를 글자 단위로 원래 위치에 다시 쓰기
    #    원본 글자 칸(bbox) 안에서 가운데 정렬, 새 글자가 더 넓으면 그 글자만 장평을 줄임
    for block in sp.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                size = span["size"]
                if size < 2 or SKIP_SPAN.match("".join(c["c"] for c in span["chars"])):
                    continue
                bold = title_area.intersects(fitz.Rect(span["bbox"]))
                color = _rgb(span["color"])
                for ch in span["chars"]:
                    c = ch["c"]
                    if not c.strip():
                        continue
                    x0, _, x1, _ = ch["bbox"]
                    y = ch["origin"][1]
                    w_old = x1 - x0
                    w_new = family.font_for(c, bold).text_length(c, fontsize=size)
                    scale = min(1.0, w_old / w_new) if w_new else 1.0
                    x = x0 + max(0.0, (w_old - w_new * scale) / 2)
                    put_text(page, (x, y), c, size, family, bold, color, scale)
    data = out.tobytes()
    src.close()
    out.close()
    return data


_blank_cache: dict = {}


def render(kind: str, case: dict, out_stem: Path, dpi: int = 200):
    form, draw, title = DOCS[kind]
    family = get_family(case["font"])
    key = (form, family.key)
    if key not in _blank_cache:
        _blank_cache[key] = build_blank(form, family)
    doc = fitz.open("pdf", _blank_cache[key])
    draw(doc[0], case, writer_cls=make_writer(family))   # 값 기입: 서식과 같은 글씨체
    doc.set_metadata({"title": title, "subject": "SYNTHETIC TEST DOCUMENT - not a real record",
                      "creator": "receipt_generator", "producer": "receipt_generator"})
    doc.subset_fonts()
    doc.save(out_stem.with_suffix(".pdf"), garbage=4, deflate=True)
    doc[0].get_pixmap(dpi=dpi).save(out_stem.with_suffix(".png"))
    doc.close()
