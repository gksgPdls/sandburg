"""
A9·A10 비교 대상 칸 고르기 (2026-11-06).
정상 문서 오탐 상위가 전부 '금액이 아닌 숫자'(①②, 단독 '1', 전화번호, 연도, 글자 섞인 줄)라서
  - 금액 모양 글자만 (숫자·쉼표·점·원·공백, 숫자 3개 이상, 전화번호 모양 제외)
    OCR이 옆 ①② 번호·괄호를 붙여 읽은 칸은 그걸 떼고 판단 (clean_text)
  - 비교는 글자 높이가 비슷한(±20%) 칸끼리만 (다른 글꼴·크기 줄과 섞지 않음)
"""
import re

MONEY_ONLY = re.compile(r"^[\d,\.\s]+원?$")
MONEY_FMT = re.compile(r"^\d{1,3}([,.]\d{3})+$")
PHONE = re.compile(r"\d{2,4}-\d{3,4}-\d{4}")
JUNK = re.compile(r"[\u2460-\u24FF\u3251-\u325F\u32B1-\u32BF()\[\]（）]")   # ①~⑳·㉑~ 같은 동그라미 번호, 괄호


LOOKALIKE = str.maketrans({"S": "5", "s": "5", "O": "0", "o": "0", "I": "1", "l": "1", "B": "8", "$": "5"})
LOOK_RE = re.compile(r"[SsOoIlB$]")


def clean_text(t):
    """OCR이 금액 옆 번호·괄호를 붙여 읽은 것 떼기: '493.100②' -> '493.100', '(12,000)' -> '12,000' (11-06 추가)
    숫자와 닮은 글자를 숫자로: 'S6,000' -> '56,000', '1S1200' -> '151200' (숫자가 3개 이상 섞인 칸만, 11-06 추가)"""
    t = JUNK.sub("", (t or "").strip()).strip().rstrip("원").strip()
    if LOOK_RE.search(t) and sum(ch.isdigit() for ch in t) >= 3:
        t = t.translate(LOOKALIKE)
    return t


def money_like(c):
    raw = (c.get("text") or "").strip()
    if PHONE.search(raw):
        return False
    t = clean_text(raw)
    if not MONEY_ONLY.match(t):
        return False
    if c.get("kind") != "money" and not MONEY_FMT.match(t):     # 원래 금액으로 안 잡힌 칸은 잡글자를 뗀 뒤 금액 형식일 때만
        return False
    return len(re.sub(r"\D", "", t)) >= 3


def peers(cells, c, tol=0.2):
    """c와 글자 높이가 비슷한 다른 칸들"""
    h = c["bbox"][3]
    return [o for o in cells if o is not c and abs(o["bbox"][3] - h) <= tol * h]


DATE_FMT = re.compile(r"^\d{4}[-.]\d{1,2}[-.]\d{1,2}$")


def date_like(c):
    """날짜 칸 (2021-11-10, 2021.11.10). 잡글자를 뗀 뒤 판단 (11-06 추가)"""
    return c.get("kind") == "date" or bool(DATE_FMT.match(clean_text(c.get("text"))))


def column_numeric(c, cells, tol_h=0.3):
    """금액 열에 있는데 OCR이 형식을 깨뜨려 읽은 칸 (예: 찌그러진 '3,000,000'을 '3.00.0.0'으로 읽음, 11-06 추가).
    숫자가 3개 이상이고 글자 대부분이 숫자·쉼표·점이며, 오른쪽 끝이 같은 금액 칸이 2개 이상 있는 칸."""
    if money_like(c) or date_like(c):
        return False
    t = clean_text(c.get("text"))
    if PHONE.search(t) or len(re.sub(r"\D", "", t)) < 3:
        return False
    if sum(ch.isdigit() or ch in ",." for ch in t) < 0.8 * len(t):
        return False
    x, y, w, h = c["bbox"]
    r = x + w
    same = [o for o in cells if o is not c and money_like(o) and abs(o["bbox"][3] - h) <= tol_h * h
            and abs(o["bbox"][0] + o["bbox"][2] - r) <= max(4, 0.4 * h)]
    return len(same) >= 2
