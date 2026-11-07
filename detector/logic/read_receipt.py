"""
판독기: 진료비 계산서·영수증 이미지 → Reading (칸 값 + 위치 + 불확실 표시).

원리 (레포 generator/alteration/forge/3_logic/receipt_form.py, real_data_analyzer/core/receipt.py와 같음)
  1. 한글 OCR(Apple Vision)으로 칸 이름을 읽는다. 머리글(본인부담금·공단부담금·전액·비급여)로 열 위치를 정한다
  2. 표 안 숫자를 같은 높이끼리 묶어 '숫자 줄'을 만들고, 줄 왼쪽 라벨을 서식 행 순서(진찰료 → … → 합계)에
     순서를 지키며 맞춘다(동적계획법) → 라벨을 조금 틀리게 읽어도 위치·순서로 보정
  3. 오른쪽 금액산정내용(진료비 총액 등)은 라벨 오른쪽 같은 높이의 숫자
  4. 체크 칸(외래/입원, 야간/공휴일, 요양기관 종류)은 OCR이 ■를 놓치므로 이미지에서 칸이 칠해진 정도로
  5. 숫자는 RapidOCR로 읽는다 (Vision은 한글은 잘 읽지만 숫자를 빠뜨리거나 세로로 붙은 칸을 합쳐 읽음).
     같은 자리를 Vision이 다른 숫자로 읽었으면 uncertain (규칙이 그 칸을 쓰면 '참고'로 낮춤)
real_data_analyzer와 다른 점: 합계 줄이 행 합과 안 맞아도 버리지 않는다 (불일치가 곧 위조 신호).
"""
import re
from statistics import median

import numpy as np
from PIL import Image, ImageOps

from . import ocr as O
from .reading import Reading

ROW_ORDER = [
    ("consult", ["진찰료", "진찰"]), ("room", ["입원료"]), ("room_1", ["1인실"]),
    ("room_23", ["2ㆍ3인실", "2·3인실", "23인실"]), ("room_4", ["4인실이상", "4인실"]), ("meal", ["식대"]),
    ("med_act", ["투약및조제료행위료", "투약행위료", "조제료행위료", "행위료", "행위"]),
    ("med_drug", ["투약및조제료약품비", "투약약품비", "조제료약품비", "약품비", "약품"]),
    ("inj_act", ["주사료행위료", "주사행위료", "행위료", "행위"]), ("inj_drug", ["주사료약품비", "주사약품비", "약품비", "약품"]),
    ("anesth", ["마취료", "마취", "신경치료"]), ("proc", ["처치및수술료", "처치및수술", "처치"]), ("lab", ["검사료", "검사"]),
    ("imaging", ["영상진단료", "영상진단"]), ("radio", ["방사선치료료", "방사선치료"]), ("material", ["치료재료대", "치료재료"]),
    ("rehab", ["재활및물리치료료", "재활및물리치료", "물리치료"]), ("psych", ["정신요법료", "정신요법"]),
    ("blood", ["전혈및혈액성분제제료", "혈액성분제제료", "혈액성분제제"]), ("ct", ["CT진단료", "CT진단", "CT"]),
    ("mri", ["MRI진단료", "MRI진단", "MRI"]), ("pet", ["PET진단료", "PET진단", "PET"]), ("us", ["초음파진단료", "초음파진단", "초음파"]),
    ("prosth", ["보철교정료", "보철ㆍ교정료", "보철"]), ("room_diff", ["병실차액"]), ("cert", ["제증명수수료", "제증명"]),
    ("etc_early", ["기타"]), ("selective", ["선별급여", "41조의4"]), ("senior", ["65세이상등정액", "65세이상"]),
    ("fixed_ltc", ["정액수가요양병원"]), ("fixed_hospice", ["정액수가완화의료"]),
    ("drg", ["질병군포괄수가", "포괄수가진료비", "포괄수가"]), ("etc", ["기타"]), ("total", ["합계"]),
    ("cap", ["상한액초과금", "상한액"]),
]
ROW_KEYS = [k for k, _ in ROW_ORDER]
SIDE_LABELS = {
    "grand_total": ["진료비총액"], "nhis_total": ["공단부담총액"], "patient_total": ["환자부담총액"],
    "already_paid": ["이미납부한금액"], "to_pay": ["납부할금액"], "paid.card": ["카드"], "paid.cash_receipt": ["현금영수증"],
    "paid.cash": ["현금"], "paid.total": ["합계"], "unpaid": ["납부하지않은금액"],
}
PATIENT_TYPES = ["건강보험", "건보", "의료급여", "일반", "자보", "자동차보험", "산재", "보훈"]
INST_WORDS = {"clinic": "의원급", "hospital": "병원급", "general": "종합병원", "tertiary": "상급종합"}
DARK_ON = 0.3    # 글자로 판별이 안 될 때, 체크 칸이 칠해졌다고 보는 어두운 픽셀 비율 (■ 0.36~0.39, 빈 칸 0.07~0.17)


def _choose(gray, size, opts):
    """여러 선택지 중 체크된 하나. opts: {key: (토큰, 글자)}. 글자 판별(checkbox_text)을 먼저,
    안 되면 괄호를 못 읽은 선택지가 하나뿐인지, 그래도 안 되면 칸의 어두운 정도로. 못 정하면 None."""
    state = {k: O.checkbox_text(t, w) for k, (t, w) in opts.items()}
    dark = {k: O.checkbox_dark(gray, t, w, size) or 0.0 for k, (t, w) in opts.items()}
    on = [k for k, v in state.items() if v]
    if on:
        return max(on, key=lambda k: dark[k])
    unseen = [k for k, v in state.items() if v is None]
    if len(unseen) == 1 and len(opts) > 1:
        return unseen[0]
    best = max(dark, key=dark.get)
    rest = [v for k, v in dark.items() if k != best]
    return best if dark[best] > 0.12 and (not rest or dark[best] > 1.4 * max(rest)) else None


def _rapid_number(text):
    """RapidOCR 글자 → 금액. '37,830③', '⑤13,160' 처럼 원문자가 붙은 것도. 금액이 아니면 None."""
    t = re.sub(r"[①-⑪\s]", "", text.replace("，", ",").replace("．", "."))
    return int(re.sub(r"\D", "", t)) if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+|\d+", t) else None


def _nums(tokens):
    """금액 토큰. '13,200②'처럼 옆 칸 원문자가 붙어 읽히면 덩어리 가운데가 옆 열로 밀리므로,
    숫자 부분이 글자 중 차지하는 위치로 x를 다시 잡는다 (열 판정은 cx로 함)."""
    out = []
    for t in tokens:
        v = _rapid_number(t["t"])
        if v is None:
            continue
        text = t["t"]
        m = re.search(r"\d[\d,.]*", text)
        if m and len(text) > len(m.group(0)):
            w = t["x1"] - t["x0"]
            x0, x1 = t["x0"] + w * m.start() / len(text), t["x0"] + w * m.end() / len(text)
            t = {**t, "x0": x0, "x1": x1, "cx": (x0 + x1) / 2}
        out.append({**t, "v": v})
    return out


def _other_agrees(vtoks, t, v):
    """다른 OCR(Vision)이 같은 자리에서 읽은 숫자들 중 v가 있나. 같은 자리에 숫자가 없으면 None(대조 불가)."""
    xs, ys = t.get("x0"), t.get("y0")
    pad_x, pad_y = 0.01, (t["y1"] - t["y0"]) * 0.5
    found = []
    for r in vtoks:
        if r["x1"] < xs - pad_x or r["x0"] > t["x1"] + pad_x or r["y1"] < ys - pad_y or r["y0"] > t["y1"] + pad_y:
            continue
        found += [int(re.sub(r"\D", "", n)) for n in re.findall(r"\d{1,3}(?:[.,]\d{3})+|\d+", r["t"].replace(" ", ""))]
    if not found:
        return None
    return v in found


class _Out:
    def __init__(self, size, vtoks):
        self.f, self.size, self.vtoks = {}, size, vtoks

    def put(self, key, value, tok=None, number=False):
        cell = {"value": value}
        if tok is not None:
            cell["box"] = O.box_px(tok, self.size)
            if number:
                agree = _other_agrees(self.vtoks, tok, value)
                if agree is False:
                    cell["uncertain"] = True
                if tok.get("conf", 1) < 0.3:
                    cell["uncertain"] = True
        self.f[key] = cell


def read(src):
    """이미지 경로 또는 PIL → (Reading, 판독 메모)."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    gray = np.asarray(img.convert("L"))
    size = img.size
    tokens = [dict(t) for t in O.vision(img)]
    out = _Out(size, tokens)
    notes = []
    if len(tokens) < 30:
        return Reading("receipt", {}), ["글자 수 부족"]
    slope = O.deskew(tokens, [["환자등록번호", "환자성명", "진료기간"], ["진료과목", "환자구분", "영수증번호"], ["본인부담금", "공단부담금"]])
    words = O.split_words(tokens)
    rnums = _nums([dict(r, ry=r["cy"] - slope * r["cx"]) for r in O.rapid(img)])   # 숫자는 RapidOCR
    hs = [t["h"] for t in rnums]
    tol = (median(hs) if hs else 0.01) * 0.75

    # ── 머리글(열) ── 공단부담금 기준으로 왼쪽 본인부담 = 본인부담금, 오른쪽 = 전액본인부담
    upper = [t for t in words if t["cy"] < 0.45]
    nhis_c = [t for t in upper if O.similar(t["t"], "공단부담금") >= 0.8 and len(O.norm(t["t"])) <= 7]
    if not nhis_c:
        return Reading("receipt", {}), ["머리글(공단부담금) 못 찾음"]
    nhis_h = max(nhis_c, key=lambda t: O.similar(t["t"], "공단부담금"))
    band = [t for t in upper if abs(t["ry"] - nhis_h["ry"]) < tol * 2.5 and t is not nhis_h]
    bonin = [t for t in band if O.similar(t["t"], "본인부담") >= 0.8 and len(O.norm(t["t"])) <= 7
             and "일부" not in t.get("src", t["t"])]
    left = [t for t in bonin if t["cx"] < nhis_h["cx"]]
    right = [t for t in bonin + [t for t in band if O.similar(t["t"], "전액") >= 0.9] if t["cx"] > nhis_h["cx"]]
    if not left:
        return Reading("receipt", {}), ["머리글(본인부담금) 못 찾음"]
    heads = {"nhis": nhis_h, "self": max(left, key=lambda t: (O.similar(t["t"], "본인부담금"), t["cx"]))}
    if right:
        heads["full"] = min(right, key=lambda t: t["cx"])
    min_x = heads["full"]["x1"] if "full" in heads else nhis_h["x1"]
    for t in upper:
        k = O.best_label(t["t"], {"select": ["선택진료료"], "nonselect": ["선택진료료이외"], "nonpay": ["비급여"]}, 0.85)
        if k and t["cx"] > min_x and abs(t["ry"] - nhis_h["ry"]) < 0.06:
            heads.setdefault(k, t)
    old = "select" in heads or "nonselect" in heads
    out.put("form", "old" if old else "2024")
    table_top = max(heads["self"]["ry"] + heads["self"]["h"] / 2, nhis_h["ry"] + nhis_h["h"] / 2)
    label_right = heads["self"]["x0"] - 0.005
    col_keys = ["self", "nhis", "full"] + (["select", "nonselect"] if old else ["nonpay"])
    step = nhis_h["cx"] - heads["self"]["cx"]
    centers = {k: heads[k]["cx"] if k in heads else heads["self"]["cx"] + step * i for i, k in enumerate(col_keys)}
    col_list = sorted(centers.items(), key=lambda kv: kv[1])
    bounds = [(col_list[i][1] + col_list[i + 1][1]) / 2 for i in range(len(col_list) - 1)]
    table_right = col_list[-1][1] + step * 0.6

    def col_of(tok):
        def idx(x):
            return next((i for i, b in enumerate(bounds) if x < b), len(col_list) - 1)
        return col_list[idx(tok["cx"])][0]

    # ── 표 안 숫자 줄 → 서식 행 (순서 지키는 정렬) ──
    nums = sorted([n for n in rnums if n["ry"] > table_top - 0.002 and label_right < n["cx"] < table_right],
                  key=lambda n: n["ry"])
    lines = []
    for n in nums:
        if lines and abs(n["ry"] - lines[-1]["ry"]) < tol:
            lines[-1]["nums"].append(n)
            lines[-1]["ry"] = median(x["ry"] for x in lines[-1]["nums"])
        else:
            lines.append({"ry": n["ry"], "nums": [n]})
    labels = [{**t, "t": O.label_text(t["t"])} for t in tokens
              if t["x0"] < label_right and t["ry"] > table_top - 0.002 and O.label_text(t["t"])]
    for ln in lines:
        # 줄 근처 라벨 후보 전부 (거리 포함). 사진은 기울기 보정 후에도 숫자 줄이 라벨보다 반 줄쯤 어긋나기도 해서
        # 가장 가까운 라벨 하나만 쓰면 '합계' 줄이 바로 위 '기타'에 붙는다 → 후보마다 이름 점수 - 거리 벌점으로 고른다.
        # '행위료'·'약품비'는 투약과 주사에 다 있음 → 왼쪽 묶음 이름(투약 및 조제료 / 주사료)을 앞에 붙여 구분
        cands = []
        for t in labels:
            d = abs(t["ry"] - ln["ry"]) / tol
            if d >= 1.6:
                continue
            text = t["t"]
            if any(O.similar(text, w) >= 0.9 for w in ("행위료", "약품비")):
                grp = [g for g in labels if g["x1"] <= t["x0"] + 0.002 and abs(g["ry"] - t["ry"]) < tol * 2.2]
                if grp:
                    text = min(grp, key=lambda g: abs(g["ry"] - t["ry"]))["t"] + " " + text
            cands.append((text, d))
        ln["near"] = cands
        ln["label"] = min(cands, key=lambda c: c[1])[0] if cands else ""

    def score(ln, j):
        best = 0.0
        for text, d in ln["near"]:
            sim = max(O.similar(text, al) for al in ROW_ORDER[j][1])
            if ROW_KEYS[j] == "total" and O.similar(text, "합계") >= 0.9:
                sim = 1.5
            best = max(best, sim - 0.35 - 0.3 * d)
        return best

    n, m = len(lines), len(ROW_ORDER)
    dp = [[0.0] * (m + 1)] + [[-1e9] * (m + 1) for _ in range(n)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(dp[i][j - 1], "skip_row"), (dp[i - 1][j], "skip_line"), (dp[i - 1][j - 1] + score(lines[i - 1], j - 1), "match")]
            dp[i][j], back[i][j] = max(opts, key=lambda o: o[0])
    i, j, assign = n, m, {}
    while i > 0 and j > 0:
        mv = back[i][j]
        if mv == "match":
            assign[i - 1] = ROW_KEYS[j - 1]
            i, j = i - 1, j - 1
        elif mv == "skip_row":
            j -= 1
        else:
            i -= 1
    last = max((i for i, k in assign.items() if k in ("total", "cap")), default=len(lines) - 1)
    unassigned = 0
    totals_line = cap_line = None
    for idx, ln in enumerate(lines):
        key = assign.get(idx)
        if idx > last:
            continue
        if key is None:
            unassigned += any(x["v"] for x in ln["nums"])
            continue
        if key == "total":
            totals_line = ln
            continue
        if key == "cap":
            cap_line = ln
            continue
        key = "etc" if key == "etc_early" else key
        for x in ln["nums"]:
            c = col_of(x)
            path = f"rows.{key}.{c}"
            prev = out.f.get(path, {}).get("value", 0)
            out.put(path, prev + x["v"], x, number=True)
    if unassigned:
        notes.append(f"어느 행인지 모르는 숫자 줄 {unassigned}개")
    # 읽은 행은 4열을 모두 채운다(빈 칸 = 0). 서식에서 빈 칸은 0이라 합계 검산에 필요
    for row in {k.split(".")[1] for k in out.f if k.startswith("rows.")}:
        for c in col_keys:
            out.f.setdefault(f"rows.{row}.{c}", {"value": 0})
    out.put("rows_complete", unassigned == 0 and totals_line is not None)
    if totals_line:
        # 합계 줄은 표 맨 아래라 사진 원근 때문에 머리글 기준 열 위치가 한 칸씩 밀리기 쉽다 →
        # ① ② ③ ④ 표시로 정한다 ('①13,200' = ①의 값, '13,200②' = ② 앞이므로 ①의 값).
        # 표시가 없으면 숫자 개수가 열 개수와 같을 때 왼쪽부터 순서대로, 아니면 열 위치로
        circ = "①②③④⑤"
        tnums = sorted(totals_line["nums"], key=lambda x: x["cx"])
        for x in tnums:
            raw = x["t"].replace(" ", "")
            lead = circ.find(raw[0]) if raw and raw[0] in circ else -1
            tail = circ.find(raw[-1]) if raw and raw[-1] in circ else -1
            x["tcol"] = lead + 1 if lead >= 0 else (tail if tail >= 1 else None)
        if all(x["tcol"] is None for x in tnums) and len(tnums) == len(col_keys):
            for i, x in enumerate(tnums):
                x["tcol"] = i + 1
        for x in tnums:
            key = f"totals.t{x['tcol']}" if x["tcol"] and x["tcol"] <= len(col_keys) else \
                f"totals.t{col_keys.index(col_of(x)) + 1}"
            out.put(key, x["v"], x, number=True)
        tcols = {c: f"totals.t{i + 1}" for i, c in enumerate(col_keys)}
        for c in col_keys:
            out.f.setdefault(tcols[c], {"value": 0})
    else:
        notes.append("합계 줄 못 찾음")
    if cap_line:
        out.put("totals.cap", sum(x["v"] for x in cap_line["nums"]), cap_line["nums"][0], number=True)

    # ── 오른쪽 금액산정내용 ──
    side_tokens = [{**t, "t": O.label_text(t["t"])} for t in tokens if t["cx"] > table_right - step * 0.3 and O.label_text(t["t"])]
    side_nums = [x for x in rnums if x["cx"] > table_right - step * 0.3]
    for t in sorted(side_tokens, key=lambda t: t["ry"]):
        k = O.best_label(t["t"], SIDE_LABELS, threshold=0.72)
        if not k or k in out.f:
            continue
        cands = [x for x in side_nums if x["cx"] > t["x1"] and abs(x["ry"] - t["ry"]) < tol * 2.2]
        if cands:
            x = max(cands, key=lambda x: (-abs(x["ry"] - t["ry"]), x["cx"]))
            out.put(k, x["v"], x, number=True)
    out.f.pop("unpaid", None)

    # ── 머리 부분: 진료기간, 야간·공휴일, 외래·입원, 영수증번호, 환자구분 ──
    head = sorted([t for t in tokens if t["ry"] < table_top - 0.01], key=lambda t: (round(t["ry"], 2), t["cx"]))
    period = [t for t in head if "부터" in t["t"] or "까지" in t["t"] or O.DATE_RE.search(t["t"])]
    dates = O.find_dates(" ".join(t["t"] for t in period))
    if dates:
        out.put("visit.start", str(dates[0]), period[0])
        out.put("visit.end", str(dates[1] if len(dates) > 1 else dates[0]), period[-1])
    for word, key in (("야간", "visit.night"), ("공휴일", "visit.holiday")):
        t = next((t for t in head if word in t["t"].replace(" ", "") and "진료" not in t["t"]), None)
        if t is None:
            continue
        st = O.checkbox_text(t, word)
        if st is None:
            d = O.checkbox_dark(gray, t, word, size)
            st = None if d is None else d > DARK_ON
        if st is not None:
            out.put(key, st, t)
    # 외래/입원: 제목줄 체크는 OCR이 ■를 '[ ]'로, ']'를 '1'로 읽어 믿기 어려움 → 입원료·식대 금액으로 판단
    inpatient_amt = sum(out.f.get(f"rows.{k}.{c}", {}).get("value", 0)
                        for k in ("room", "room_1", "room_23", "room_4", "meal") for c in ("self", "nhis"))
    out.put("visit.inpatient", inpatient_amt > 0)
    for t in head:
        s = t["t"].replace(" ", "")
        if "receipt_no" not in out.f and re.fullmatch(r"\d{6}-\d{2,8}|\d{8,}(-\d+)?", s):
            out.put("receipt_no", s, t)
        if "patient.type" not in out.f and O.norm(s) in PATIENT_TYPES:
            out.put("patient.type", O.norm(s), t)

    # ── 아래 부분: 요양기관 종류, 사업자번호, 소재지, 발행일 ──
    bottom_y = (totals_line or cap_line or {"ry": 0.6})["ry"]
    if cap_line:
        bottom_y = max(bottom_y, cap_line["ry"])
    low = sorted([t for t in tokens if t["ry"] > bottom_y + tol], key=lambda t: (round(t["ry"], 3), t["cx"]))
    opts = {}
    for key, word in INST_WORDS.items():
        for t in low:
            s = t["t"].replace(" ", "")
            if word in s and not (key == "general" and "상급종합" in s and s.find("종합병원") == s.find("상급종합") + 2):
                opts[key] = (t, word)
                break
    best = _choose(gray, size, opts) if opts else None
    if best:
        out.put("institution.type", best, opts[best][0])
    for t in low:
        s = t["t"].replace(" ", "")
        mb = re.search(r"\d{3}-\d{2}-\d{5}", s)
        if mb and "institution.bizno" not in out.f:
            out.put("institution.bizno", mb.group(0), t)
    addr_lab = next((t for t in low if O.similar(t["t"], "소재지") >= 0.8 or O.similar(t["t"], "사업장소재지") >= 0.8), None)
    if addr_lab:
        same = [t for t in low if t["cx"] > addr_lab["x1"] and abs(t["ry"] - addr_lab["ry"]) < tol * 1.6 and re.search(r"[시도군구]", t["t"])]
        if same:
            out.put("institution.address", min(same, key=lambda t: t["cx"])["t"].strip(), min(same, key=lambda t: t["cx"]))
    expl = next((t for t in low if O.similar(t["t"], "항목별설명") >= 0.8), None)
    issue_toks = [t for t in low if (expl is None or t["ry"] < expl["ry"]) and re.search(r"\d", t["t"])
                  and not re.search(r"\d{3}-\d{2}-\d{5}|\d{2,3}-\d{3,4}-\d{4}", t["t"].replace(" ", ""))]
    for grp in O.text_lines(issue_toks, tol):
        ds = O.find_dates(" ".join(t["t"] for t in grp))
        if ds:
            out.put("issue_date", str(ds[0]), grp[0])
            break
    return Reading("receipt", out.f), notes
