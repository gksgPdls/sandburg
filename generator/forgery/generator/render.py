"""서식별 값 칸 좌표표 + 값 기입 함수 (실제 그리기·저장은 render_rebuild.py).

서식 원본: assets/forms/ (법제처 국가법령정보센터 별표·서식 PDF, 수정하지 않음)
좌표 단위: PDF 포인트(pt, 1pt = 1/72인치), 원점은 페이지 왼쪽 위.
좌표는 서식 PDF 안의 글자·선 위치를 PyMuPDF로 추출해서 잡은 값.
"""
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
FORMS = ROOT / "assets" / "forms"

# ════════════════════════════════════════════════════════════════
#  진료비 계산서·영수증 [별지 제6호서식] 좌표
# ════════════════════════════════════════════════════════════════
# 항목 표의 행 순서 (서식 위→아래) — 각 행 높이 12.5pt, 첫 행 상단 y=136.3
RECEIPT_ROWS = [
    "consult", "room_1", "room_23", "room_4", "meal", "med_act", "med_drug", "inj_act", "inj_drug",
    "anesth", "proc", "lab", "imaging", "radio", "material", "rehab", "psych", "blood",
    "ct", "mri", "pet", "us", "prosth", "cert",                       # 선택항목
    "selective", "senior", "fixed_ltc", "fixed_hospice", "drg", "etc",  # 선별급여 ~ 기타
]
ZERO_FILL_ROWS = RECEIPT_ROWS[:24] + ["etc"]   # 0 표기 대상 (기본·선택항목 + 기타)
ROW_TOP, ROW_H = 136.3, 12.5
TOTAL_TOP = ROW_TOP + ROW_H * len(RECEIPT_ROWS)      # 합계 행 (512.1)
CAP_TOP = TOTAL_TOP + ROW_H                          # 상한액 초과금 행 (524.6)
COL_X = {"self": (169.1, 218.2), "nhis": (218.2, 267.2), "full": (267.2, 316.3), "nonpay": (316.3, 365.3)}
VAL_X = (475.3, 536.2)                               # 금액산정내용 값 칸

R6 = {
    # 제목 [ ]외래 [ ]입원 ([ ]퇴원[ ]중간) 체크 위치 (x 중심, 기준선 y)
    "chk_out": (130.5, 41.0), "chk_in": (184.5, 41.0), "chk_dis": (243.5, 41.0), "chk_mid": (291.5, 41.0),
    # 상단 값 칸 (x0, y0, x1, y1)
    "chart_no": (58.6, 61.1, 169.1, 73.6),
    "name": (169.1, 61.1, 316.3, 73.6),
    # 진료기간 ". . .부터 . . .까지" 의 점(.) 바로 왼쪽 x 좌표 → 숫자를 오른쪽 정렬로 찍음
    "period_start": (331.0, 346.0, 361.0), "period_end": (400.0, 415.0, 430.0), "period_y": (61.1, 73.6),
    "dept": (58.6, 86.2, 169.1, 98.7),
    "drg": (169.1, 86.2, 316.3, 98.7),
    "room": (316.3, 86.2, 365.3, 98.7),
    "ptype": (365.3, 86.2, 423.1, 98.7),
    "receipt_no": (423.1, 86.2, 536.2, 98.7),
    # 금액산정내용 (y0, y1)
    "g6": (111.2, 136.3), "g7": (136.3, 161.3), "g8": (161.3, 186.4), "g9": (186.4, 211.4), "g10": (211.4, 236.5),
    "card": (236.5, 249.0), "cash_receipt": (249.0, 261.6), "cash": (261.6, 274.0), "paid": (274.0, 286.6),
    "unpaid": (286.6, 299.1),
    # 하단
    "inst_chk": {"clinic": 185.5, "hospital": 285.5, "general": 345.5, "tertiary": 413.5}, "inst_y": 546.4,
    "bizno": (169.1, 549.6, 292.4, 562.2),
    "hosp_name": (315.4, 549.6, 410.3, 562.2),
    "phone": (466.7, 549.6, 536.2, 562.2),
    "address": (169.1, 562.2, 365.3, 574.7),
    "ceo": (437.2, 562.2, 503.0, 574.7),
    # 발행일 "년 월 일" 글자 바로 왼쪽 x
    "issue": (239.5, 291.5, 343.5), "issue_y": (574.7, 587.2),
}

# ════════════════════════════════════════════════════════════════
#  진단서 [별지 제5호의2서식] 좌표
# ════════════════════════════════════════════════════════════════
D52 = {
    "reg_no": (108.0, 114.6, 175.0, 138.2),
    "serial_no": (108.0, 138.2, 175.0, 158.9),
    "name": (175.0, 164.9, 344.9, 197.1),
    "rrn": (418.2, 164.9, 534.8, 197.1),
    "address": (175.0, 199.0, 534.8, 212.0),
    "phone": (405.0, 213.5, 522.0, 227.0),
    "chk_clinical": (93.7, 260.0), "chk_final": (93.7, 274.3),
    "dx_main": (255.0, 233.0, 416.0, 252.0), "dx_sub": (255.0, 260.5, 416.0, 279.5),
    "code_main": (418.2, 244.0, 534.8, 257.0), "code_sub": (418.2, 258.0, 534.8, 271.0),
    # 날짜: 서식에 인쇄된 (년 시작, 년 끝, 월 시작, 월 끝, 일 시작) x 좌표
    "onset": (243.9, 253.4, 271.1, 280.5, 298.0), "diag_date": (464.0, 473.5, 486.6, 496.1, 509.1),
    "date_y": (285.8, 315.5),
    "opinion": (180.0, 318.0, 530.0, 387.0),
    "admit": (263.9, 274.4, 288.6, 299.1, 313.2), "discharge": (446.4, 457.0, 471.1, 481.7, 495.8),
    "adm_y": (389.3, 419.2),
    "purpose": (180.0, 419.2, 534.8, 446.2),
    "remark": (180.0, 446.2, 534.8, 473.1),
    "issue": (450.1, 459.1, 481.4, 490.4, 512.8), "issue_y": (504.0, 517.5),
    "hosp_name": (214.0, 526.0, 534.0, 540.0),
    "hosp_addr": (214.0, 548.0, 534.0, 562.0),
    "chk_doctor": (262.0, 580.0),
    "license": (474.5, 569.5, 519.0, 583.5),
    "doctor": (302.0, 592.0, 460.0, 606.0),
}


class Writer:
    """서식 페이지에 글자를 칸 맞춰 찍는 도우미. 칸보다 길면 글자 크기를 줄임."""

    def __init__(self, page: fitz.Page, font: fitz.Font):
        self.page = page
        self.font = font          # 칸 너비 계산용 글꼴

    def _put(self, pos, text, size, bold=False):
        """글자 한 덩어리를 페이지에 씀 — 글씨체 처리는 render_rebuild.FamilyWriter 에서 구현."""
        raise NotImplementedError

    def fit(self, text: str, width: float, size: float, min_size: float = 5.0) -> float:
        while size > min_size and self.font.text_length(text, fontsize=size) > width:
            size -= 0.25
        return size

    def cell(self, text, box, align="c", size=8.0, pad=2.0, bold=False):
        if text in (None, ""):
            return
        text = str(text)
        x0, y0, x1, y1 = box
        size = self.fit(text, (x1 - x0) - 2 * pad, size)
        w = self.font.text_length(text, fontsize=size)
        x = {"l": x0 + pad, "c": (x0 + x1 - w) / 2, "r": x1 - pad - w}[align]
        y = (y0 + y1) / 2 + size * 0.36
        self._put((x, y), text, size, bold)

    def right_at(self, text, x_right, y_band, size=8.0):
        """미리 인쇄된 글자(년/월/일, 점 등) 바로 왼쪽에 오른쪽 정렬로 찍기."""
        y0, y1 = y_band
        w = self.font.text_length(str(text), fontsize=size)
        self._put((x_right - w, (y0 + y1) / 2 + size * 0.36), str(text), size)

    def center_at(self, text, x_center, y_band, size=8.0):
        y0, y1 = y_band
        w = self.font.text_length(str(text), fontsize=size)
        self._put((x_center - w / 2, (y0 + y1) / 2 + size * 0.36), str(text), size)

    def mark(self, ch, x_center, y_base, size=9.0):
        w = self.font.text_length(ch, fontsize=size)
        self._put((x_center - w / 2, y_base), ch, size, bold=True)

    def paragraph(self, text, box, size=8.5, leading=1.7):
        """칸 너비에 맞춰 줄바꿈해서 세로 가운데 정렬."""
        x0, y0, x1, y1 = box
        lines, cur = [], ""
        for ch in text:
            if self.font.text_length(cur + ch, fontsize=size) > (x1 - x0):
                lines.append(cur)
                cur = ch.lstrip()
            else:
                cur += ch
        lines.append(cur)
        top = (y0 + y1) / 2 - len(lines) * size * leading / 2
        for i, ln in enumerate(lines):
            self._put((x0, top + size * (i * leading + 1.2)), ln, size)


def money(x: int, zero_fill: bool = True) -> str:
    return f"{x:,}" if (x or zero_fill) else ""


# ── 진료비 계산서·영수증 ─────────────────────────────────────────
def draw_receipt(page: fitz.Page, case: dict, writer_cls):
    w = writer_cls(page)
    p, h, d, rc = case["patient"], case["hospital"], case["dates"], case["receipt"]
    v = case["variant"]["receipt"]
    m = v["mark"]

    # 제목 체크
    if rc["kind"] == "outpatient":
        w.mark(m, *R6["chk_out"], size=11)
    else:
        w.mark(m, *R6["chk_in"], size=11)
        w.mark(m, *R6["chk_dis"], size=11)

    # 상단
    w.cell(p["chart_no"], R6["chart_no"])
    w.cell(p["name"], R6["name"])
    for key, dt in (("period_start", d["start"]), ("period_end", d["end"])):
        xy, xm, xd = R6[key]
        w.right_at(dt.year, xy - 0.3, R6["period_y"], size=6.0)
        w.right_at(f"{dt.month:02d}", xm - 0.3, R6["period_y"], size=6.0)
        w.right_at(f"{dt.day:02d}", xd - 0.3, R6["period_y"], size=6.0)
    w.cell(case["department"], R6["dept"])
    w.cell(rc["room"], R6["room"])
    w.cell(rc["patient_type"], R6["ptype"])
    w.cell(rc["receipt_no"], R6["receipt_no"], size=7.5)

    # 항목 표
    zf = v["zero_fill"]
    for i, key in enumerate(RECEIPT_ROWS):
        y0 = ROW_TOP + ROW_H * i
        a = rc["rows"].get(key)
        if a is None and not (zf and key in ZERO_FILL_ROWS):
            continue
        for col, (x0, x1) in COL_X.items():
            val = (a or {}).get(col, 0)
            w.cell(money(val, zf), (x0, y0, x1, y0 + ROW_H), align="r", size=7.5)
    t = rc["totals"]
    for k, col in (("t1", "self"), ("t2", "nhis"), ("t3", "full"), ("t4", "nonpay")):
        x0, x1 = COL_X[col]
        w.cell(money(t[k]), (x0 + 10, TOTAL_TOP, x1, TOTAL_TOP + ROW_H), align="r", size=7.5)
    w.cell(money(t["t5"]), (179.0, CAP_TOP, 218.2, CAP_TOP + ROW_H), align="r", size=7.5)

    # 금액산정내용
    pay = rc["payment"]
    for key, val in (("g6", rc["grand_total"]), ("g7", rc["nhis_total"]), ("g8", rc["patient_total"]),
                     ("g9", rc["already_paid"]), ("g10", rc["to_pay"]), ("card", pay["card"]),
                     ("cash_receipt", pay["cash_receipt"]), ("cash", pay["cash"]), ("paid", rc["paid_total"]),
                     ("unpaid", 0)):
        y0, y1 = R6[key]
        w.cell(money(val), (VAL_X[0], y0, VAL_X[1], y1), align="r", size=8)

    # 하단 기관 정보
    w.mark(m, R6["inst_chk"][h["type"]], R6["inst_y"], size=9)
    w.cell(h["bizno"], R6["bizno"])
    w.cell(h["name"], R6["hosp_name"])
    w.cell(h["phone"], R6["phone"], size=7.5)
    w.cell(h["address"], R6["address"], size=7.5)
    w.cell(h["ceo"], R6["ceo"], align="r")
    iy, im, idd = R6["issue"]
    w.right_at(d["issue"].year, iy, R6["issue_y"])
    w.right_at(f"{d['issue'].month:02d}", im, R6["issue_y"])
    w.right_at(f"{d['issue'].day:02d}", idd, R6["issue_y"])


# ── 진단서 ──────────────────────────────────────────────────────
def draw_diagnosis(page: fitz.Page, case: dict, writer_cls):
    w = writer_cls(page)
    p, h, d, dc = case["patient"], case["hospital"], case["dates"], case["diagnosis_cert"]
    v = case["variant"]["diagnosis"]
    m = v["mark"]

    w.cell(p["chart_no"], D52["reg_no"], align="l", size=9)
    w.cell(dc["serial_no"], D52["serial_no"], align="l", size=9)
    w.cell(p["name"], D52["name"], size=10)
    w.cell(p["rrn"], D52["rrn"], size=9.5)
    w.cell(p["address"], D52["address"], align="l", size=9)
    w.cell(p["phone"], D52["phone"], align="l", size=9)

    w.mark(m, *(D52["chk_final"] if dc["final"] else D52["chk_clinical"]), size=10)
    def dx_text(dx):
        """영문 병기(변형)는 칸에 9pt로 들어갈 때만. 글자를 작게 줄여 넣지 않음."""
        text = f"{dx['name']} ({dx['en']})"
        x0, _, x1, _ = D52["dx_main"]
        if v["english_name"] and w.font.text_length(text, fontsize=9) <= (x1 - x0) - 4:
            return text
        return dx["name"]
    w.cell(dx_text(dc["main"]), D52["dx_main"], align="l", size=9)
    w.cell(dc["main"]["code"], D52["code_main"], size=9.5, bold=True)
    if dc["sub"]:
        w.cell(dx_text(dc["sub"]), D52["dx_sub"], align="l", size=9)
        w.cell(dc["sub"]["code"], D52["code_sub"], size=9.5, bold=True)

    def ymd(dt, xs, band, size=9.0):
        """연도는 '년' 왼쪽에 붙이고, 월·일은 인쇄된 글자 사이 빈칸 가운데에 찍음."""
        y_start, y_end, m_start, m_end, d_start = xs
        w.right_at(dt.year, y_start - 1.5, band, size)
        w.center_at(f"{dt.month:02d}", (y_end + m_start) / 2, band, size)
        w.center_at(f"{dt.day:02d}", (m_end + d_start) / 2, band, size)

    ymd(d["onset"], D52["onset"], D52["date_y"])
    ymd(d["diagnosis"], D52["diag_date"], D52["date_y"], size=8.2)
    w.paragraph(dc["opinion"], D52["opinion"], size=9)
    if case["admitted"]:
        ymd(d["start"], D52["admit"], D52["adm_y"])
        ymd(d["end"], D52["discharge"], D52["adm_y"])
    w.cell(dc["purpose"], D52["purpose"], align="l", size=9.5)
    w.cell(dc["remark"], D52["remark"], align="l", size=9.5)

    ymd(d["issue"], D52["issue"], D52["issue_y"], size=9)
    w.cell(h["name"], D52["hosp_name"], align="l", size=10, bold=True)
    w.cell(h["address"], D52["hosp_addr"], align="l", size=9.5)
    w.mark(m, *D52["chk_doctor"], size=10)
    w.cell(h["license_no"], D52["license"], size=9.5)
    w.cell(h["doctor"], D52["doctor"], align="l", size=10.5, bold=True)


# ── 문서 종류 → (서식 PDF, 값 기입 함수, 제목) ───────────────────
DOCS = {
    "diagnosis": ("form5-2_diagnosis.pdf", draw_diagnosis, "진단서"),
    "receipt": ("form6_receipt.pdf", draw_receipt, "진료비 계산서·영수증"),
}

