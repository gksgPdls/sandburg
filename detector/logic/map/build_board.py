"""로직 엔진 지도 (판 형식) → index.html. 저장된 결과만 읽음, 외부 호출 없음. 이 맥 안에서만 본다.

예시 출처 (먼저 있는 것부터):
  1. 합친 탐지기 VLM 결과 — demo_real20/결과.json (실제 문서 20장)
  2. 합친 탐지기 VLM 결과 — scryn-combine-web/outputs/combine/실제추가20_*/문서별.jsonl (있으면 자동으로 붙음)
  3. 위 둘에 없는 묶음만 로컬 OCR 결과 (logic_results.jsonl)
    python build_board.py
"""
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
# 예시 목록(실제 서류에서 나온 금액·날짜·코드 설명)은 공개 저장소에 올리지 않고 이 맥의 examples_local.py 에만 둔다.
try:
    import examples_local as _X
except ImportError:      # 공개 저장소: examples_sample.py 를 복사해 examples_local.py 로 만들어 채운다
    import examples_sample as _X
import sys as _sys0
_sys0.path.insert(0, str(Path.home() / "Desktop/SCRYN/3_탐지기/scryn-combine-web"))  # logic_detector
import _compat  # noqa: E402,F401  (logic_detector 찾기)
ROOT = Path.home() / "Desktop/SCRYN/3_탐지기"
V20 = Path.home() / "Desktop/SCRYN/2_데이터셋/시험_데모/demo_real20"
ADD = sorted((ROOT / "scryn-combine-web/outputs/combine").glob("실제추가20_*/문서별.jsonl"))
SAMPLES = Path.home() / "Desktop/SCRYN/2_데이터셋/변조_실서류_samples"
IMG = HERE / "img"; IMG.mkdir(exist_ok=True)
E = html.escape
GC = {"확정": "sus", "강함": "sus", "참고": "chk"}
NORM = {"의심": "위변조 의심", "확인필요": "확인 필요", "위변조의심": "위변조 의심"}
METHOD = {"amount_sum_break": "금액 하나 바꾸기", "amount_sum_fixed": "금액과 합계 같이 바꾸기", "amount_digit": "금액 자릿수 키우기",
          "digit_combo": "숫자 오려 붙이기", "copymove": "금액 칸 복사", "kcd_swap": "질병 코드만 바꾸기", "add_diag": "병명 한 줄 추가",
          "add_text": "병명 한 줄 추가", "date_conflict": "날짜 순서 뒤집기", "date_shift": "날짜 옮기기", "date_copy": "날짜 숫자 복사",
          "vlm_한칸": "AI로 한 칸 다시 그림", "sd_inpainting": "AI 인페인팅", "edms": "EDMS 변형", "holiday_check": "공휴일 칸 체크", "holiday_date_shift": "공휴일 진료일을 평일로", "fee_inflate": "진찰료 금액 부풀리기", "fee_inflate_full": "진찰료 부풀리고 합계까지 맞춤", "normal": "조작 없음", "": "조작 없음"}
OUT = {"tp": ("정탐", "ok"), "off": ("이유 다름", "part"), "fp": ("오탐", "miss"), "fn": ("미탐", "miss"), "tn": ("정답", "ok")}
OUT_LONG = {"tp": "변조 서류를 잡음", "off": "변조는 걸었지만 바뀐 곳과 무관한 기준으로", "fp": "정상 서류인데 걸림",
            "fn": "변조 서류를 놓침", "tn": "정상 서류를 정상으로"}

labels = {}
for f in SAMPLES.glob("*/labels.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        labels[r["id"]] = (f.parent, r)


def mask_of(i):
    if i not in labels or not labels[i][1].get("mask"):
        return None
    p = labels[i][0] / labels[i][1]["mask"]
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L")) > 127
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1] if len(xs) else None


def lvl(x):
    return NORM.get(x, x)


# ── 1. VLM 20장
docs = {}
saved = json.load(open(V20 / "결과.json", encoding="utf-8"))
for row in csv.DictReader(open(V20 / "정답.csv", encoding="utf-8-sig")):
    p = V20 / "samples" / row["파일"]
    x = saved.get(hashlib.sha1(p.read_bytes()).hexdigest())
    if x:
        nm = p.stem.endswith("_정상")
        docs[p.stem] = {"path": p, "chg": [int(float(v)) for v in row["바뀐 곳"].split(",")] if row["바뀐 곳"] else None,
                        "verdict": lvl(x["engines"][0]["level"]), "findings": [f for f in x["findings"] if f["rule"] != "P-A11"],
                        "via": "VLM 판독", "boxes": [], "normal": nm, "method": "" if nm else row["조작 방식"], "desc": row["바뀐 내용"],
                        "reading": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in (x.get("reading") or {}).items()},
                        "kind": "diagnosis" if "진단" in row["서류"] else "receipt"}
# ── 2. 추가로 돌린 VLM 결과
for jf in ADD:
    base = ROOT / "scryn-combine-web/testsets/실제추가20"
    num = {r["문제_id"]: r["원래_번호"] for r in csv.DictReader(open(base / "번호표.csv", encoding="utf-8-sig"))}
    for l in open(jf, encoding="utf-8"):
        r = json.loads(l)
        i = num[r["문제_id"]]
        lab = labels[i][1]
        nm = lab["method"] == "normal"
        docs[i] = {"path": Path(r["파일"]), "chg": None if nm else mask_of(i), "verdict": lvl(r["logic"]["level"]),
                   "findings": r["logic"].get("findings", []), "via": "VLM 판독", "normal": nm,
                   "boxes": [list(b) for b in r["logic"].get("boxes", []) if b and len(b) == 4],
                   "method": "" if nm else lab["method"], "desc": lab["description"], "added": True}
# ── 3. 로컬 OCR (VLM 에 없는 묶음 채우기용)
loc = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "logic_results.jsonl", encoding="utf-8")}


def local(i, rule):
    r = loc[i]
    f = next(f for f in r["findings"] if f["rule"] == rule)
    return {"path": Path(r["path"]), "chg": None if r["normal"] else mask_of(i.replace("_orig", "")), "verdict": r["verdict"],
            "findings": [f], "via": "로컬 OCR 판독", "boxes": f.get("boxes") or [], "normal": r["normal"],
            "method": "" if r["normal"] else r["method"], "desc": r["desc"]}


def fmt(v):
    if isinstance(v, (list, tuple)) and len(v) > 3 and all(isinstance(x, (int, float)) for x in v):
        return f"공식 진찰료 {min(v):,} ~ {max(v):,}원 중 하나 ({len(v)}가지)"
    return "" if v in (None, "") else (f"{v:,}" if isinstance(v, (int, float)) else str(v))


def shot(tag, d, pad=0.14, full=False):
    im = ImageOps.exif_transpose(Image.open(d["path"])).convert("RGB")
    W, H = im.size
    chgs = d.get("chgs") or ([d["chg"]] if d["chg"] else [])
    marks = chgs + d["boxes"] + d.get("fboxes", [])
    if marks and not full:
        x0 = min(b[0] for b in marks); y0 = min(b[1] for b in marks)
        x1 = max(b[0] + b[2] for b in marks); y1 = max(b[1] + b[3] for b in marks)
        p = int(pad * max(W, H))
        x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p // 2), min(W, x1 + p), min(H, y1 + p // 2)
        need = int(0.62 * (x1 - x0)) - (y1 - y0)
        if need > 0:
            y0, y1 = max(0, y0 - need // 2), min(H, y1 + need - need // 2)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    out = IMG / f"b_{tag}{'_full' if full else ''}.jpg"
    t = im.crop((x0, y0, x1, y1)); t.thumbnail((1400, 1400)); t.save(out, quality=86)
    sw = max(1.5, (x1 - x0) / (550 if full else 450))
    r = "".join(f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]}" height="{b[3]}" class="hitr" stroke-width="{sw}"/>' for b in d["boxes"])
    r += "".join(f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]}" height="{b[3]}" class="focr" stroke-width="{sw}"/>' for b in d.get("fboxes", []))
    for c in chgs:
        r += f'<rect x="{c[0]}" y="{c[1]}" width="{c[2]}" height="{c[3]}" class="chgr" stroke-width="{sw}"/>'
    return (f'<div class="shot"><svg viewBox="{x0} {y0} {x1-x0} {y1-y0}"><image href="img/{out.name}" x="{x0}" y="{y0}" '
            f'width="{x1-x0}" height="{y1-y0}" preserveAspectRatio="none"/>{r}</svg></div>')


def vlm_boxes(d, f):
    """VLM 예시: 보여 주는 기준(f, 없으면 걸린 것 전부)이 쓴 칸들을 로컬 OCR 글자 위치로 찾음."""
    import locate_local as LL
    from logic_detector.logic import focus as _fo, rules as _ru
    from logic_detector.logic.reading import Reading as _Rd
    rd = d["reading"]
    found = _ru.check(_Rd(d["kind"], rd)) + _fo.check(_Rd(d["kind"], rd))
    if f:
        found = [g for g in found if g["rule"] == f["rule"] and g["title"][:15] == f["title"][:15]] or \
                [g for g in found if g["rule"] == f["rule"]]
    fields = list(dict.fromkeys(x for g in found for x in g.get("fields", [])))
    return LL.boxes_for(d["path"], fields, rd)


def insets(tag, d):
    """확대 사진: 실제로 바뀐 곳 · 탐지기가 의심한 칸을 따로 (둘 다 있을 때)."""
    chg = d.get("chgs") or ([d["chg"]] if d.get("chg") else [])
    out = []
    if chg and d["boxes"]:
        out.append((shot(tag + "_c", {**d, "boxes": [], "fboxes": []}), "실제로 바뀐 곳 (초록)"))
        out.append((shot(tag + "_h", {**d, "chg": None, "chgs": None}), "탐지기가 의심한 칸 (빨강)"))
    elif chg:
        out.append((shot(tag, d), "실제로 바뀐 곳 (초록) — 탐지기가 짚은 칸 없음"))
    elif d["boxes"]:
        out.append((shot(tag, d), "탐지기가 의심한 칸 (빨강) — 조작 없는 원본"))
    return "".join(f'<div class="inset">{h}<p class="cap">{c}</p></div>' for h, c in out)


def _plain(t):
    """'S_2' 같은 표기(밑줄이 글꼴에서 안 보임)를 풀어 씀."""
    import re as _re
    return _re.sub(r"S_(\d)", r"S○\1", t).replace("S○2", "S52처럼 셋째 자리 2").replace("S○3", "셋째 자리 3").replace("S○0", "셋째 자리 0")


def chip(g):
    return f'<span class="v {GC.get(g, "off")}">{E(g)}</span>'


# ── 묶음: 근거(제목, 설명) · 흐름 · 기준
GROUPS = [
    ("A", "서식 산식 · 합계", "영수증",
     [("요양급여규칙 별지 제6호서식", "「국민건강보험 요양급여의 기준에 관한 규칙」이 정한 진료비 계산서·영수증 법정 서식. 금액산정내용 칸에 ⑥ = ①+②+③+④ 같은 계산식이 인쇄돼 있어, 정상 영수증이면 항상 성립합니다."),
      ("같은 서식의 항목별 표", "진찰료·입원료 같은 행의 본인부담 · 공단부담 · 전액본인 · 비급여 열을 더하면 아래 합계 칸이 됩니다 (합계의 정의)."),
      ("허용 오차 10원", "원 단위 절사로 생기는 차이만 봐줍니다. ⑩·⑪처럼 병원마다 감면·수납 방식이 다른 칸은 참고로만 씁니다.")],
     ["합계 칸 · 항목 행 읽기", "서식 산식 계산", "열마다 행 합 = 합계 칸?", "10원 넘게 차이 → 걸림"],
     [("A1", "⑥=①+②+③+④ 등 서식 산식", "확정"), ("A2", "행 합 = 열 합계", "확정"), ("A3", "⑪ 낸 금액 ≤ ⑩ 낼 금액", "참고")]),
    ("B", "수가 · 가산", "영수증",
     [("심평원 요양급여비용 고시 (진찰료)", "진찰료는 해마다 고시된 단가로 정해집니다. 의원·병원·종합병원 × 초진·재진마다 기본 금액이 있고, 야간·공휴일과 토요일엔 정해진 가산 금액이 붙습니다. 영수증 진찰료(본인부담 + 공단부담)는 이 표의 금액 중 하나여야 합니다."),
      ("관공서의 공휴일에 관한 규정", "일요일 · 국경일 · 설 · 추석 · 선거일 같은 공휴일 목록. 공휴일 가산 표시나 가산 금액이 있으면 진료일이 실제 공휴일이어야 합니다."),
      ("꺼 둔 것", "본인부담률(B1)·입원료(B4)는 환자·병원별 예외가 많고 공식 데이터가 모자라 꺼 두었습니다. 진찰료 단가(B2)도 메인에서는 꺼져 있고 이 예시에서만 켰습니다.")],
     ["진찰료 행 · 진료일 · 가산 표시 읽기", "그해 공식 진찰료 표와 비교", "표에 없는 금액 → B2", "가산 표시·가산 금액 ↔ 요일이 안 맞음 → B6"],
     [("B2", "진찰료 = 그해 공식 진찰료 표의 금액 (예시에서만 켬)", "참고"), ("B6", "공휴·야간·토요 가산 ↔ 진료일 요일", "참고"), ("B1·B4", "본인부담률 · 입원료", "꺼 둠")]),
    ("C", "상병코드", "진단서",
     [("심평원 상병마스터 2025.9", "건강보험 청구에 쓰는 질병분류기호 전체 목록. 코드마다 공식 병명, 완전코드 여부, 성별·연령 제한, 주상병으로 쓸 수 있는지가 들어 있습니다."),
      ("KCD-8 (한국표준질병사인분류)", "통계청이 고시하는 코드 체계. 하위 세분류가 있는 코드는 끝자리까지 써야 완전코드입니다. 코드와 병명이 짝지어져 있어, 병명이 다른 코드의 공식 병명이면 코드만 바꾼 흔적입니다.")],
     ["코드 · 병명 읽기", "상병마스터 조회", "병명 ↔ 코드 공식 병명 대조", "성별 · 연령 · 주상병 확인"],
     [("C1", "있는 코드 · 완전코드", "강함"), ("C2", "병명이 다른 코드 것 = 코드만 바꿈", "강함"),
      ("C3", "성별 전용 코드 (연령은 참고)", "강함"), ("C4", "주상병 불가 · 같은 코드 두 번", "참고")]),
    ("D", "날짜 · 번호", "영수증 · 진단서",
     [("날짜의 정의", "진료 시작 ≤ 끝 ≤ 발행일, 발병 ≤ 진단 ≤ 발급, 오늘보다 뒤 날짜 없음. 어기면 정의상 불가능해 확정입니다."),
      ("국세청 사업자등록번호 체계", "10자리 중 마지막 자리는 앞 9자리로 계산하는 검증번호, 앞 3자리는 관할 세무서 코드입니다."),
      ("「주민등록법」 주민등록번호", "앞 6자리는 실제로 있는 생년월일이어야 하고, 2020년 10월 전에 받은 번호는 마지막 자리가 검증번호입니다."),
      ("병원 EMR 영수증번호", "많은 병원이 번호 앞에 발행일을 넣지만 병원마다 달라 참고로만 씁니다.")],
     ["날짜 · 번호 읽기", "날짜 순서 비교", "검증번호 계산", "어긋나면 걸림"],
     [("D1", "시작 ≤ 끝, 발병 ≤ 진단 ≤ 발급", "확정"), ("D2", "영수증번호 앞 날짜 = 진료일", "참고"),
      ("D3", "사업자번호 검증번호", "확정"), ("D4", "주민번호 날짜 · 검증번호", "확정")]),
]

# ── 예시: (묶음, 문서, 결과, 보여 줄 규칙, 한 줄 설명)
EX = list(_X.EX)


def auto_examples():
    """추가 VLM 결과에서 자동으로 예시 만들기 (이미 있는 문서는 건너뜀)."""
    seen = {n for _, n, *_ in EX}
    MG = {"kcd_swap": "C", "add_diag": "C", "add_text": "C", "date_conflict": "D", "date_shift": "D", "date_copy": "D"}
    out = []
    for i, d in docs.items():
        if i in seen or not d.get("added"):
            continue
        fs = sorted(d["findings"], key=lambda f: {"확정": 0, "강함": 1, "참고": 2}.get(f["grade"], 3))
        if d["normal"]:
            g = "C" if "진단" in labels[i][1]["doc_type"] else "A"
            out.append((g, i, "fp" if fs else "tn", fs[0]["rule"] if fs else None, "정상 서류인데 걸림" if fs else "걸린 기준 없음"))
            continue
        if not fs:
            out.append((MG.get(d["method"], "A"), i, "fn", None, "걸린 기준 없음"))
            continue
        c = d["chg"]
        on = c and any(not (b[0] + b[2] < c[0] or c[0] + c[2] < b[0] or b[1] + b[3] < c[1] or c[1] + c[3] < b[1]) for b in d["boxes"])
        g = fs[0]["rule"][0] if fs[0]["rule"][0] in "ABCD" else MG.get(d["method"], "A")
        out.append((g, i, "tp" if (on or not d["boxes"]) else "off", fs[0]["rule"], fs[0]["title"]))
    return out


for e in auto_examples():
    if e[2] == "tp" and not any(x[0] == e[0] and x[2] == "tp" for x in EX):
        EX.append(e)
# ── B 묶음 예시 (로컬 OCR 결과로 먼저 채우고, 실제B20 VLM 결과가 있으면 그걸로 바꿈. B2 는 이 예시에서만 켬)
def _bbox_of(mask_path):
    m = np.array(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1] if len(xs) else None


def b_doc(key, path, mask, res, rule, desc, method):
    f = [x for x in res["findings"] if x["rule"] == rule]
    return {"path": Path(path), "chg": mask, "verdict": res["verdict"], "findings": f + [x for x in res["findings"] if x not in f],
            "via": "로컬 OCR 판독", "boxes": [list(b) for b in (f[0].get("boxes") or []) if b] if f else [],
            "normal": False, "method": method, "desc": desc}


B6R = json.load(open(HERE / "b6_dates/local_results.json", encoding="utf-8")) if (HERE / "b6_dates/local_results.json").exists() else {}
B2R = json.load(open(HERE / "b2_fee/local_results.json", encoding="utf-8")) if (HERE / "b2_fee/local_results.json").exists() else {}
B2X = json.load(open(HERE / "b2_fee/existing_results.json", encoding="utf-8")) if (HERE / "b2_fee/existing_results.json").exists() else {}
B6L = {r["id"]: r for r in csv.DictReader(open(HERE / "b6_dates/labels.csv", encoding="utf-8-sig"))} if (HERE / "b6_dates/labels.csv").exists() else {}
B_EX = list(_X.B_EX)
for key, kind, rule, why in B_EX:
    if kind == "b6" and key in B6R:
        lab = B6L[key]
        docs[key] = b_doc(key, HERE / "b6_dates" / lab["tampered"], _bbox_of(HERE / "b6_dates" / lab["mask"]), B6R[key], rule,
                          lab["description"].split(". ")[0], "holiday_date_shift")
    elif kind == "b2" and key in B2R:
        x = B2R[key]
        docs[key] = b_doc(key, HERE / "b2_fee" / f"{key}.jpg", _bbox_of(HERE / "b2_fee" / f"{key}_mask.png"), x, rule,
                          x["description"], "fee_inflate")
    elif kind == "b2f" and (HERE / "b2_full/local_results.json").exists():
        x = json.load(open(HERE / "b2_full/local_results.json", encoding="utf-8")).get(key)
        if not x:
            continue
        lab = {r["id"]: r for r in csv.DictReader(open(HERE / "b2_full/labels.csv", encoding="utf-8-sig"))}[key]
        docs[key] = b_doc(key, HERE / "b2_full" / lab["tampered"], _bbox_of(HERE / "b2_full" / lab["mask"]), x, rule,
                          "진찰료 본인부담 +2만 원, 이어지는 합계 칸 6개도 모두 맞춰 고침", "fee_inflate_full")
        import cv2 as _cv
        _m = (np.array(Image.open(HERE / "b2_full" / lab["mask"]).convert("L")) > 0).astype("uint8")
        _n, _, _st, _ = _cv.connectedComponentsWithStats(_cv.dilate(_m, np.ones((9, 9), np.uint8)))
        docs[key]["chgs"] = [list(map(int, x[:4])) for x in _st[1:] if x[4] > 30]
    elif kind == "b2x" and key in B2X:
        lab = labels[key][1]
        docs[key] = b_doc(key, labels[key][0] / lab["tampered"], mask_of(key), B2X[key], rule, lab["description"], lab["method"])
    else:
        continue
    if any(f["rule"] == rule for f in docs[key]["findings"]):
        EX.append(("B", key, "tp", rule, why))

# 실제B20 VLM 결과가 있으면: 저장된 판독값으로 B2 를 켜고 다시 판정해 같은 문서를 VLM 판독으로 바꿈 (호출 없음)
B20 = sorted((ROOT / "scryn-combine-web/outputs/combine").glob("실제B20_*/문서별.jsonl"))
B2F = sorted((ROOT / "scryn-combine-web/outputs/combine").glob("실제B2F_*/문서별.jsonl"))
if B20 or B2F:
    import sys as _s
    _s.path.insert(0, str(ROOT / "scryn-combine-web"))
    from logic_detector.logic import rules as _rules
    from logic_detector.logic.reading import Reading as _Reading
    if _rules.b2_b6_consult not in _rules.RECEIPT_RULES:
        _rules.RECEIPT_RULES.append(_rules.b2_b6_consult)
    runs = [(B20[-1], "실제B20")] if B20 else []
    runs += [(B2F[-1], "실제B2F")] if B2F else []
    recs = []
    for jf, ts in runs:
        num = {r["문제_id"]: r["원래_번호"] for r in csv.DictReader(open(ROOT / f"scryn-combine-web/testsets/{ts}/번호표.csv", encoding="utf-8-sig"))}
        recs += [(num[json.loads(l)["문제_id"]], json.loads(l)) for l in open(jf, encoding="utf-8")]
    for key, r in recs:
        fields = (r.get("판독") or {}).get("판독")
        if key.endswith("_orig") and key[:-5] in B6L and fields:      # 공휴일 진료 원본 (정상)
            docs[key] = {"path": Path(B6L[key[:-5]]["original"]), "chg": None, "verdict": "정상", "findings": [], "via": "VLM 판독",
                         "boxes": [], "normal": True, "method": "", "desc": ""}
        if key not in docs or not fields:
            continue
        fs = _rules.check(_Reading("receipt", fields))
        docs[key].update({"findings": fs, "verdict": _rules.verdict(fs), "via": "VLM 판독", "boxes": [],
                          "reading": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in fields.items()}, "kind": "receipt"})
    EX[:] = [e for e in EX if e[0] != "B" or e[2] != "tp" or any(f["rule"] == e[3] for f in docs[e[1]]["findings"])]

if not B20:   # VLM 결과가 없을 때만 로컬 OCR 예시 (예: 빈 체크 칸을 체크로 잘못 읽은 경우)
    for e in _X.LOCAL_EXTRA:
        docs[e[1]] = local(e[1], e[3])
    EX += list(_X.LOCAL_EXTRA)

frames, edges = [], []
ORDER = {"tp": 0, "off": 1, "fp": 2, "fn": 3, "tn": 4}
COLS = {g: 2 + i for i, g in enumerate(["A", "B", "C", "D", "F"])}


def ex_item(g, k, n, oc, rule, why):
    d = docs[n]
    f = next((f for f in d["findings"] if f["rule"] == rule), None) if rule else None
    if d["via"] == "VLM 판독" and not d["boxes"] and d.get("reading"):
        d["boxes"] = vlm_boxes(d, f)
    lab, cls = OUT[oc]
    truth = "정상 서류" if d["normal"] else "변조 서류"
    ea = (f'<div class="ea"><div><span>기대</span><b>{E(fmt(f.get("expected")))}</b></div><div><span>읽음</span><b>{E(fmt(f.get("actual")))}</b></div></div>'
          if f and f.get("expected") not in (None, "") else "")
    hit = (f'{chip(f["grade"])} {E(f["rule"])} {E(f["title"])}' if f else
           ("없음" if not d["findings"] else " · ".join(E(x["rule"] + " " + x["title"][:30]) for x in d["findings"][:2])))
    what = "조작 없음 (원본)" if d["normal"] else METHOD.get(d["method"], d["method"]) + (f" — {d['desc'].split('. ')[0]}" if d["desc"] else "")
    return (f'<article class="exi" data-oc="{oc}"><div class="exh"><span class="oc {cls}">{E(lab)}</span>'
            f'<b class="num">{E(n.replace("_정상", " (정상 원본)"))}</b><span class="exv">{E(truth)} → 로직 판정 <i>{E(d["verdict"])}</i></span>'
            f'<span class="tag">{E(d["via"])}</span></div>'
            f'<div class="exg"><div class="full">{shot(f"{g}{k}_{n}", d, full=True)}<p class="cap">전체 서류</p></div><div class="side">'
            f'{insets(f"{g}{k}_{n}", d)}'
            f'<div class="kv"><span>조작</span><b>{E(what[:90])}</b></div>'
            f'<div class="kv"><span>걸린 기준</span><b>{hit}</b></div>{ea}'
            f'<div class="kv"><span>설명</span><b>{E(why)}</b></div></div></div></article>')


def tabs(counts):
    b = '<button type="button" data-f="all" aria-pressed="true">전체</button>'
    b += "".join(f'<button type="button" data-f="{k}" aria-pressed="false" class="{OUT[k][1]}">{OUT[k][0]}</button>'
                 for k, v in counts.items() if v)
    return f'<div class="tabs">{b}</div>'


for g, name, kind, srcs, flow, rules in GROUPS:
    col = COLS[g]
    sh = "".join(f'<div class="srcbox"><b>{E(t)}</b><p>{E(x)}</p></div>' for t, x in srcs)
    steps = "".join(f'<li>{E(s)}</li>' for s in flow)
    rl = "".join(f'<li><b>{E(r)}</b><span>{E(t)}</span>{chip(gr)}</li>' for r, t, gr in rules)
    mine = sorted([e for e in EX if e[0] == g], key=lambda e: ORDER[e[2]])
    cnt = {k: sum(e[2] == k for e in mine) for k in ORDER}
    note = (f'<p class="nocatch">{E(_X.NOTES.get(g, ""))}</p>' if _X.NOTES.get(g) else "")
    
    frames.append(
        f'<section class="frame grp" data-stop="g{g}" data-nav="{g}" data-name="{g} {E(name)}" style="grid-column:{col};grid-row:3">'
        f'<div class="face"><span class="fn">{g}</span><b>{E(name)}</b><span class="fs">{E(kind)}</span></div>'
        f'<header><span class="pin">{g}</span><h2>{E(name)}</h2><span class="kind">{E(kind)}</span></header>'
        f'<div class="lab">근거</div><div class="srcs">{sh}</div>'
        f'<div class="lab">흐름</div><ol class="flow">{steps}</ol>'
        f'<div class="lab">기준</div><ul class="rules">{rl}</ul>{note}</section>')
    items = "".join(ex_item(g, k, *e[1:]) for k, e in enumerate(mine))
    frames.append(
        f'<section class="frame excard" data-stop="x{g}" data-name="{g} 예시" data-group="{g}" data-f="all" style="grid-column:{col};grid-row:4">'
        f'<div class="face"><span class="fn">{g} · {E(name)}</span><b>탐지 결과 예시</b><span class="fs">'
        + " · ".join(OUT[k][0] for k, v in cnt.items() if v) + '</span></div>'
        f'<header><span class="pin">{g}</span><h2>탐지 결과 예시</h2></header>{tabs(cnt)}<div class="exlist">{items}</div></section>')
    edges += [["top", f"g{g}", "v"], [f"g{g}", f"x{g}", "v"]]

# ── 주목 칸 (판정 아님 · 보험금이 걸린 칸)
FOCUS_ITEMS = [("암·유사암", "질병 코드 C(일반암) ↔ 유사암(D0x·C44·C73), 진단일 = 조직검사 결과보고일"),
               ("진단일·보장개시일", "진단 연월일 — 보장개시(90일)·감액기간 경계"),
               ("골절", "골절 코드(S52처럼 셋째 자리 2) ↔ 염좌(3)·타박(0) 코드, 소견의 손상 종류"),
               ("화상", "코드 4번째 자리(1·2·3도) ↔ 소견의 깊이"),
               ("수술", "수술명·수술일 — 수술 분류에 따라 보험금이 다름"),
               ("입원 기간", "입·퇴원일 ↔ 소견 날짜 — 입원일당"),
               ("상해·질병", "코드 첫 글자(S·T 손상인지) ↔ 사고 경위"),
               ("비급여", "주사·영상·재활·처치 비급여 합 5만 원 이상인 줄 — 실손 보험금")]
FX = list(_X.FX)
FOC = {"hit": ("바꾼 곳을 짚음", "ok"), "orig": ("조작 없는 원본", "part")}
col = COLS["F"]
lis = "".join(f'<li><b>{E(a)}</b><span>{E(b)}</span><span class="v off">주목</span></li>' for a, b in FOCUS_ITEMS)
frames.append(
    f'<section class="frame grp" data-stop="gF" data-nav="주목" data-name="주목 칸" style="grid-column:{col};grid-row:3">'
    f'<div class="face"><span class="fn">주목</span><b>보험금이 걸린 칸</b><span class="fs">판정 아님</span></div>'
    f'<header><span class="pin">!</span><h2>보험금이 걸린 칸</h2><span class="kind">판정 아님</span></header>'
    f'<div class="lab">근거</div><div class="srcs"><div class="srcbox"><b>표준약관 · KCD · 금감원 자료</b><p>변조 여부와 별개로, 보험금 액수가 이 칸 값에 따라 달라지는 곳. 고칠 동기가 큰 칸이라 사람이 먼저 봐야 할 곳으로 짚습니다. 각 항목에 약관 조항을 근거로 붙입니다.</p></div></div>'
    f'<div class="lab">흐름</div><ol class="flow"><li>VLM 판독값에서 코드·단어·칸으로 항목 찾기</li><li>항목마다 볼 칸 고르기</li><li>걸린 칸과 같이 위치 묻기 (호출 1번)</li><li>서류 위에 보라 상자 + 이유·근거</li></ol>'
    f'<div class="lab">항목</div><ul class="rules">{lis}</ul></section>')
import locate_local as LL
from logic_detector.logic import focus as _fo
from logic_detector.logic.reading import Reading as _Rd
fitems, fcnt = [], {"hit": 0, "orig": 0}
for k, (n, key, tag, why) in enumerate(FX):
    x = saved.get(hashlib.sha1((V20 / "samples" / (n + ".jpg")).read_bytes()).hexdigest()) if (V20 / "samples" / (n + ".jpg")).exists() else None
    if not x or not x.get("focus"):
        continue
    rd = docs[n]["reading"]
    fo = _fo.focus(_Rd(docs[n]["kind"], rd))
    d = {"path": V20 / "samples" / (n + ".jpg"), "chg": docs[n]["chg"], "boxes": [], "normal": True,
         "fboxes": LL.boxes_for(V20 / "samples" / (n + ".jpg"), [f for it in fo for f in it["fields"]], rd)}
    kv = "".join(f'<div class="kv"><span>{E(f["name"])}</span><b>{E(_plain(f["what"])[:120])}</b></div>' for f in x["focus"])
    fcnt[key] += 1
    fitems.append(
        f'<article class="exi" data-oc="{key}"><div class="exh"><span class="oc {FOC[key][1]}">{E(FOC[key][0])}</span>'
        f'<b class="num">{E(n.replace("_정상", " (정상 원본)"))}</b><span class="tag">VLM 판독</span></div>'
        f'<div class="exg"><div class="full">{shot(f"F{k}_{n}", d, full=True)}<p class="cap">전체 서류</p></div>'
        f'<div class="side">{kv}<div class="kv"><span>설명</span><b>{E(why)}</b></div>'
        f'<div class="kv"><span>근거</span><b class="srcq">{E(x["focus"][0]["why"][:260])}…</b></div></div></div></article>')
ftabs = ('<div class="tabs"><button type="button" data-f="all" aria-pressed="true">전체</button>'
         + "".join(f'<button type="button" data-f="{k}" aria-pressed="false" class="{FOC[k][1]}">{FOC[k][0]}</button>' for k, v in fcnt.items() if v)
         + '</div>')
frames.append(
    f'<section class="frame excard" data-stop="xF" data-name="주목 예시" data-group="F" data-f="all" style="grid-column:{col};grid-row:4">'
    f'<div class="face"><span class="fn">주목 · 보험금이 걸린 칸</span><b>탐지 결과 예시</b><span class="fs">바꾼 곳을 짚음 · 원본</span></div>'
    f'<header><span class="pin">!</span><h2>탐지 결과 예시</h2></header>{ftabs}<div class="exlist">{"".join(fitems)}</div></section>')
edges += [["top", "gF", "v"], ["gF", "xF", "v"]]

T = (HERE / "template_board.html").read_text(encoding="utf-8")
(HERE / "index.html").write_text(T.replace("{{FRAMES}}", "".join(frames)).replace("{{EDGES}}", json.dumps(edges)), encoding="utf-8")
print("index.html 예시", len(EX), {g: sum(e[0] == g for e in EX) for g in "ABCD"}, "추가 결과", [p.parent.name for p in ADD])
