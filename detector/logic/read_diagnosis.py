"""
판독기: 진단서 이미지 → Reading. 칸 이름(Apple Vision) 오른쪽 같은 줄의 값을 읽는다.

  patient.rrn      주민등록번호 (######-####### 또는 뒷자리 가린 ######-#)
  diagnoses        질병분류기호 + 병명. 위에서부터 첫 코드 = 주 질병, 다음 = 부 질병
                   병명은 '(주 질병·부상) 병명' 줄의 괄호 뒤 글자
  dates.onset / dates.diagnosis   '발병 연월일' / '진단 연월일' 오른쪽 날짜
  dates.admit / dates.discharge   '입원일:' / '퇴원일:' 오른쪽 날짜
  dates.issue      '진단합니다' 문장 아래 첫 날짜
"""
import re
from statistics import median

from PIL import Image, ImageOps

from . import ocr as O
from .reading import Reading

# KCD 첫 글자는 항상 영문. OCR이 굵은 J를 3으로 읽는 식의 혼동 → RapidOCR로 다시 보고, 그래도 숫자면 이 표로 바꾸고 불확실 표시
CODE_LIKE = re.compile(r"([A-Z0-9])(\d{2}(?:\.\d{1,3})?)")
DIGIT_AS_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "J", "5": "S", "6": "G", "7": "T", "8": "B"}
RRN_RE = re.compile(r"(\d{6})\s*-\s*(\d{7}|\d)")
DATE_LABELS = {"dates.onset": ["발병연월일", "발병일"], "dates.diagnosis": ["진단연월일", "진단일"],
               "dates.admit": ["입원일"], "dates.discharge": ["퇴원일"]}


def read(src):
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    size = img.size
    toks = [dict(t) for t in O.vision(img)]
    f, notes = {}, []
    if len(toks) < 10:
        return Reading("diagnosis", {}), ["글자 수 부족"]
    for t in toks:
        t["ry"] = t["cy"]
    tol = median(t["h"] for t in toks) * 0.8

    def put(key, value, t):
        f[key] = {"value": value, "box": O.box_px(t, size)}

    def right_of(lab, stop_words=()):
        """라벨 오른쪽 같은 줄 토큰 (다음 라벨 앞까지)."""
        row = sorted([t for t in toks if t["x0"] > lab["x1"] - 0.005 and abs(t["ry"] - lab["ry"]) < tol * 1.5],
                     key=lambda t: t["cx"])
        out = []
        for t in row:
            if any(O.similar(t["t"], w) >= 0.85 for w in stop_words):
                break
            out.append(t)
        return out

    # 주민등록번호 — 전화번호(010-…)와 헷갈리지 않게 6자리-로 시작하는 것만
    for t in sorted(toks, key=lambda t: t["ry"]):
        m = RRN_RE.search(t["t"].replace(" ", ""))
        if m and not re.search(r"0\d{1,2}-\d{3,4}-\d{4}", t["t"]):
            put("patient.rrn", f"{m.group(1)}-{m.group(2)}", t)
            break

    # 질병분류기호 + 병명
    code_lab = next((t for t in toks if O.similar(t["t"], "질병분류기호") >= 0.8), None)
    codes, code_unsure = [], False
    if code_lab:
        rtoks = O.rapid(img)
        for t in toks:
            if not (code_lab["ry"] - tol < t["ry"] < code_lab["ry"] + 0.12 and t["x0"] > code_lab["x0"] - 0.05
                    and len(O.norm(t["t"])) <= 12):
                continue
            for part in re.split(r"[,/·•]", t["t"].replace(" ", "")):   # 토큰 전체가 코드 모양일 때만 (날짜 '2024년'의 202 등 제외)
                m = CODE_LIKE.fullmatch(re.sub(r"[;:'`]", "", part).strip("()[]"))
                if not m:
                    continue
                code = m.group(0)
                if code[0].isdigit():
                    alt = [O.KCD_RE.findall(r["t"].replace(" ", "")) for r in rtoks
                           if r["x0"] < t["x1"] and r["x1"] > t["x0"] and r["y0"] < t["y1"] and r["y1"] > t["y0"]]
                    alt = [c for cs in alt for c in cs if c[1:] == code[1:]]
                    if alt:
                        code = alt[0]
                    else:
                        code, code_unsure = DIGIT_AS_LETTER.get(code[0], code[0]) + code[1:], True
                codes.append((t["ry"], code, t))
        codes.sort(key=lambda c: c[0])
    names = []
    for word in ("주질병", "부질병"):
        t = next((t for t in toks if word in O.norm(t["t"])), None)
        if t is None:
            names.append(None)
            continue
        after = re.split(r"[)）]", t["t"], maxsplit=1)
        name = after[1].strip() if len(after) > 1 else ""
        name = re.sub(r"\s*\([^)]*[A-Za-z][^)]*\)\s*$", "", name)   # 뒤에 붙은 영문 병명 '(Headache)'은 뺌
        if not name:   # 병명이 다른 토큰으로 떨어져 읽힌 경우
            nxt = [x["t"] for x in right_of(t)]
            name = " ".join(nxt).strip()
        names.append((name, t) if name else None)
    dxs = []
    for i, (_, code, t) in enumerate(codes[:2]):
        nm = names[i] if i < len(names) else None
        dxs.append({"code": code, "name": nm[0] if nm else "", "main": i == 0})
    if dxs:
        f["diagnoses"] = {"value": dxs, "box": O.box_px(codes[0][2], size), "uncertain": code_unsure}
    elif code_lab:
        notes.append("질병분류기호 칸에서 코드를 못 읽음")

    # 날짜 칸
    stops = [w for ws in DATE_LABELS.values() for w in ws]
    for key, labs in DATE_LABELS.items():
        # 라벨과 날짜가 한 토큰으로 읽힌 경우 ('입원일: 2026년 02월 02 일부터')
        joined = next((t for t in toks if any(O.norm(t["t"]).startswith(w) for w in labs) and O.find_dates(t["t"])), None)
        if joined:
            put(key, str(O.find_dates(joined["t"])[0]), joined)
            continue
        lab = next((t for t in toks if any(O.similar(O.label_text(t["t"]).rstrip(":"), w) >= 0.85 for w in labs)
                    and len(O.norm(t["t"])) <= 8), None)
        if lab is None:
            continue
        row = right_of(lab, stop_words=[w for w in stops if w not in labs])
        ds = O.find_dates(" ".join(t["t"] for t in row))
        if ds:
            put(key, str(ds[0]), row[0])
    # 치료 내용 및 향후 치료에 대한 소견 — 칸 이름(왼쪽 두 줄) 오른쪽, 다음 칸(입원·퇴원 연월일) 위까지의 글
    op_lab = next((t for t in toks if O.norm(t["t"]).startswith("치료내용")), None)
    if op_lab:
        nxt = [t["ry"] for t in toks if t["ry"] > op_lab["ry"] + tol and t["x1"] < 0.3
               and any(w in O.norm(t["t"]) for w in ("입원", "퇴원", "용도"))]
        y1 = min(nxt) - tol if nxt else op_lab["ry"] + 0.08
        body = sorted([t for t in toks if t["x0"] > op_lab["x1"] - 0.01 and op_lab["ry"] - tol < t["ry"] < y1],
                      key=lambda t: (t["ry"], t["cx"]))
        if body:
            text = " ".join(" ".join(t["t"] for t in grp) for grp in O.text_lines(body, tol))
            box = O.box_px({"x0": min(t["x0"] for t in body), "y0": min(t["y0"] for t in body),
                            "x1": max(t["x1"] for t in body), "y1": max(t["y1"] for t in body)}, size)
            f["opinion"] = {"value": text, "box": box}

    decl = next((t for t in toks if "진단합니다" in O.norm(t["t"]) or "위와같이" in O.norm(t["t"])), None)
    if decl:
        below = sorted([t for t in toks if t["ry"] > decl["ry"] + tol * 0.5], key=lambda t: (t["ry"], t["cx"]))
        for grp in O.text_lines(below, tol):
            ds = O.find_dates(" ".join(t["t"] for t in grp))
            if ds:
                put("dates.issue", str(ds[0]), grp[0])
                break
    return Reading("diagnosis", f), notes
