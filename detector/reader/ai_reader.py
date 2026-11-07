"""VLM 판독기: 서류 사진 → '문서에 찍힌 값' (OpenRouter).

AI 는 읽기는 잘하고 계산·날짜 비교는 틀린다 → AI 는 읽기만 하고, 판정은 로직 엔진(detector/logic)이 한다.
- read(): 서류 한 장의 칸을 모두 읽는다 (납부 칸·발행일시·영수증번호·내원일·사업자번호·병명·소견까지)
- to_values(): 읽은 값 → export_values(visible) 형식 (행 이름·병명 표기 맞춤, 빈 칸 0)
- reread(): 검산에서 실패한 칸만 더 큰 사진으로 다시 읽고 위치(상자)도 받는다 → 잘못 읽어 생긴 실패를 걸러 내고, 맞으면 근거 위치가 된다
- crop_check()·apply_crop(): 영수증 합계·금액산정·납부 칸을 잘라 한 칸씩 다시 읽어 전체 판독과 대조한다 (조각 대조)
합성 서류만 보낸다 (or_client 규칙).
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import or_client as oc  # noqa: E402

KCD_MASTER = Path(__file__).resolve().parents[1] / "logic" / "data" / "hira_kcd_master_2025_9.csv"   # 심평원 상병마스터

READ_PROMPT = {
    "영수증": """너는 '진료비 계산서·영수증'의 값을 그대로 옮겨 적는 판독기다. 판단하지 말고 사진에 적힌 값만 적는다.
아래 JSON 하나로만 답한다. 날짜는 YYYY-MM-DD, 금액은 쉼표 없는 정수. 칸이 비어 있으면 null, 0이 적혀 있으면 0.
{
 "요양기관": {"이름": "요양기관 명칭", "사업자등록번호": "000-00-00000", "주소": "소재지"},
 "환자등록번호": "적힌 그대로",
 "진료기간": {"시작": "YYYY-MM-DD", "끝": "YYYY-MM-DD"},
 "영수증번호": "적힌 그대로 (예: 2603-01056 또는 202603-01056)",
 "행": [{"항목": "서식의 항목 이름", "본인부담": 0, "공단부담": 0, "전액본인부담": 0, "비급여": 0}],
 "합계": {"본인부담": 0, "공단부담": 0, "전액본인부담": 0, "비급여": 0},
 "상한액초과금": 0,
 "금액산정": {"진료비총액": 0, "공단부담총액": 0, "환자부담총액": 0, "이미납부한금액": 0, "납부할금액": 0},
 "납부": {"카드": 0, "현금영수증": 0, "현금": 0, "합계": 0, "납부하지않은금액": 0},
 "발행일": "YYYY-MM-DD",
 "발행일시": "YYYY-MM-DD HH:MM:SS",
 "내원일": ["YYYY-MM-DD"]
}
- 진료기간: 진료일이 하루면 시작과 끝에 같은 날짜
- 영수증번호: 환자등록번호와 다른 번호다
- 행: 금액이 하나라도 적힌 항목 줄만, 위에서 아래 순서로. 항목 이름은 서식에 인쇄된 이름 그대로
  · 항목 열에서 '행위료'·'약품비' 작은 줄은 바로 위의 큰 항목('투약 및 조제료' 또는 '주사료')에 속한다. 금액이 적힌 작은 줄이 어느 큰 항목 아래인지 확인해 "주사료-행위료", "주사료-약품비", "투약 및 조제료-약품비"처럼 적는다
  · 한 줄의 금액을 두 줄에 나눠 적거나, 같은 금액을 두 줄에 겹쳐 적지 않는다. 두 줄을 합치지도 않는다
- 합계: 항목 표 맨 아래 '합계' 줄의 ①본인부담금 ②공단부담금 ③전액본인부담 ④비급여
- 금액산정: 오른쪽 금액산정내용의 ⑥진료비 총액 ⑦공단부담 총액 ⑧환자부담 총액 ⑨이미 납부한 금액 ⑩납부할 금액
- 납부: ⑩ 아래의 카드·현금영수증·현금·합계·납부하지 않은 금액
- 발행일: 서식 아래쪽 '년 월 일' 날짜
- 발행일시: 맨 아래 '발행일시 :' 줄이 있을 때만, 없으면 null
- 내원일: 내원일 목록이 따로 적혀 있을 때만, 없으면 null""",
    "진단서": """너는 '진단서'의 값을 그대로 옮겨 적는 판독기다. 판단하지 말고 사진에 적힌 값만 적는다.
아래 JSON 하나로만 답한다. 날짜는 YYYY-MM-DD. 칸이 비어 있으면 null.
{
 "제목": "서식 맨 위 제목 그대로 (예: 진단서, 상해진단서)",
 "요양기관": {"이름": "의료기관 명칭", "주소": "주소"},
 "병명": [{"구분": "주", "병명": "적힌 그대로", "코드": "한국질병분류번호 (예: M54.56)"}],
 "발병일": "YYYY-MM-DD",
 "진단일": "YYYY-MM-DD",
 "발행일": "YYYY-MM-DD",
 "소견": "치료 내용 및 향후 치료에 대한 소견 칸의 문장 그대로 (문장 속 날짜도 적힌 그대로)",
 "용도": "적힌 그대로"
}
- 병명: 주 진단과 부 진단(있으면)을 모두. 구분은 "주" 또는 "부"
- 진단일: 진단 연월일 칸 (발병일·발행일과 다른 칸)""",
}

# 검사 항목 → 다시 읽을 칸 (읽기 프롬프트의 이름으로)
FIELDS_BY_CHECK = {
    "CALC-02": ["행"], "CALC-03": ["행", "합계"], "CALC-04": ["행"], "CALC-05": ["행"], "RULE-04": ["행"],
    "SUM-01": ["행", "합계"], "SUM-02": ["합계", "상한액초과금", "금액산정"], "SUM-03": ["금액산정", "납부"],
    "DATE-01": ["진료기간"], "DATE-02": ["진료기간", "발행일"], "DATE-04": ["진료기간", "행"],
    "CROSS-05": ["진료기간", "내원일"], "FORM-02": ["발행일", "발행일시"], "FORM-03": ["영수증번호", "진료기간"],
    "BIZ-01": ["요양기관"], "BIZ-02": ["요양기관"], "BIZ-03": ["요양기관"],
    "DATE-03": ["발병일", "진단일", "발행일"], "CROSS-06": ["소견", "발병일", "진단일"],
    "FORM-01": ["제목", "병명"], "CODE-01": ["병명"], "CODE-02": ["병명"],
    # 확인필요(주의)에도 위치를 받는다 (11-06): 사람에게 '어디를 보라'를 주고, 그 칸에만 이미지 탐지기를 돌릴 수 있게
    "MOTIVE-01": ["진료기간", "발행일"], "KCD-01": ["병명"], "RULE-02b": ["행"],
}

ROW_CANON = [("진찰", "진찰료"), ("투약약품", "투약 및 조제료-약품비"), ("조제약품", "투약 및 조제료-약품비"),
             ("투약", "투약 및 조제료-행위료"), ("조제", "투약 및 조제료-행위료"),
             ("주사약품", "주사료-약품비"), ("주사", "주사료-행위료"), ("처치", "처치 및 수술료"), ("수술", "처치 및 수술료"),
             ("검사", "검사료"), ("영상", "영상진단료"), ("재활", "재활 및 물리치료료"), ("물리", "재활 및 물리치료료"),
             ("증명", "제증명수수료")]


def num(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"[^\d-]", "", str(v))
    try:
        return int(s)
    except ValueError:
        return None


def ymd(v):
    m = re.search(r"(\d{4})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})", "" if v is None else str(v))
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}" if 1 <= mo <= 12 and 1 <= d <= 31 else None


def canon_row(name):
    n = re.sub(r"[\s·ㆍ_()\-및]", "", str(name or ""))
    if not n or "합계" in n or "상한" in n:
        return None
    for k, v in ROW_CANON:
        if all(ch in n for ch in ([k[:2], k[2:]] if len(k) > 2 else [k])):
            return v
    return str(name)


_MASTER = None


def master():
    global _MASTER
    if _MASTER is None:
        import csv
        _MASTER = {}
        with open(KCD_MASTER, encoding="cp949", newline="") as f:   # 첫 줄 = 공식 병명, 둘째 줄부터 동의어
            for r in csv.DictReader(f):
                if r["상병기호"] in _MASTER:
                    _MASTER[r["상병기호"]]["synonyms"].append(r["한글명"])
                else:
                    _MASTER[r["상병기호"]] = {**r, "synonyms": []}
    return _MASTER


def norm_text(s):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(s or ""))


def canon_code(c):
    c = re.sub(r"[^0-9A-Za-z]", "", str(c or "")).upper()
    return f"{c[:3]}.{c[3:]}" if len(c) > 3 else c


LABEL_RE = re.compile(r"^\s*[(\[]?\s*[주부]\s*(?:질병|진단|상병)?\s*[·ㆍ.]?\s*(?:부상)?\s*[)\]:.]?\s*")


def canon_name(code, name):
    """읽은 병명을 상병마스터 공식 병명(또는 동의어)에 맞춘다. 서식 글자 '(주 질병·부상)'은 떼고,
    띄어쓰기·문장부호만 다르거나 한두 글자만 다르면(예: 충격 → 충돌, 모델이 흔한 말로 바꿔 읽음) 공식 표기로 본다.
    병명을 바꾸는 변조는 코드도 함께 바뀌므로 한두 글자 차이는 잘못 읽은 것으로 본다."""
    import difflib
    name = LABEL_RE.sub("", str(name or "")) if name else name
    r = master().get(code.replace(".", ""))
    if not r or not name:
        return name
    n = norm_text(name)
    best, best_ratio = None, 0.0
    for cand in [r["한글명"]] + r["synonyms"]:
        nc = norm_text(cand)
        if n == nc or (len(n) >= 0.8 * len(nc) and nc.startswith(n)):
            return cand
        ratio = difflib.SequenceMatcher(None, n, nc).ratio()
        if ratio > best_ratio:
            best, best_ratio = cand, ratio
    return best if best_ratio >= 0.85 else name


def inst_of(d):
    d = d or {}
    name, addr = d.get("이름") or "", d.get("주소") or ""
    cls = "clinic" if name.endswith("의원") else "general" if "종합병원" in name else "hospital" if name.endswith("병원") else "clinic"
    region = "eupmyeon" if re.search(r"\S+[읍면]\s", addr) else "dong"
    biz = d.get("사업자등록번호")
    if biz:
        dg = re.sub(r"\D", "", str(biz))
        biz = f"{dg[:3]}-{dg[3:5]}-{dg[5:]}" if len(dg) == 10 else str(biz)
    return {"class": cls, "fee_type": cls, "region": region, "name": name, "address": addr, "biz_no": biz}


OUTPATIENT_RATE = {"clinic": {"dong": 0.3, "eupmyeon": 0.3}, "hospital": {"dong": 0.4, "eupmyeon": 0.35},
                   "general": {"dong": 0.5, "eupmyeon": 0.45}}   # rules/fees_2026.json outpatient_copay 와 같은 값


def fit_class(rows, inst):
    """종별(의원·병원·종합병원)은 이름에서 정하되, 이름을 잘못 읽으면 본인부담률 검사가 통째로 틀린다
    (예: 의원 이름을 실존 병원 이름으로 지어내 읽음). 행 금액의 본인부담 비율이 더 많이 맞는 종별을 쓴다."""
    name_cls, region = inst["class"], inst["region"]
    fits = {}
    for cls, rate in OUTPATIENT_RATE.items():
        r = rate[region]
        fits[cls] = sum(abs(x["copay"] - (x["copay"] + x["nhis"]) * r) <= 10 for x in rows.values() if x["copay"] + x["nhis"])
    best = max(fits, key=lambda c: (fits[c], c == name_cls))
    return best if fits[best] > fits.get(name_cls, 0) else name_cls


def usable(r, doc):
    """읽기 답이 빈 틀만 돌려준 것이 아닌지 (값이 하나도 없으면 다시 묻는다)."""
    if not isinstance(r, dict):
        return False
    if doc == "영수증":
        return any(num(x) for x in (r.get("합계") or {}).values()) or bool(r.get("행"))
    return bool(r.get("병명")) or bool(ymd(r.get("발행일")))


SHAPE_KEYS = {"합계": ["본인부담", "공단부담", "전액본인부담", "비급여"],
              "금액산정": ["진료비총액", "공단부담총액", "환자부담총액", "이미납부한금액", "납부할금액"],
              "납부": ["카드", "현금영수증", "현금", "합계", "납부하지않은금액"]}


def shape(r):
    """모델마다 다른 답 모양을 지시문 모양으로 맞춘다 (11-06 작은 모델 시험에서 '합계'를 숫자 목록으로,
    '요양기관'을 글자 하나로, '행'을 묶음으로 준 모델이 있었음)."""
    r = dict(r) if isinstance(r, dict) else {}
    for k, names in SHAPE_KEYS.items():
        x = r.get(k)
        if isinstance(x, list):                                   # [①, ②, ③, ④] 처럼 순서대로 준 경우
            r[k] = {n: v for n, v in zip(names, x) if not isinstance(v, (dict, list))}
        elif not isinstance(x, dict):
            r[k] = {}
    if isinstance(r.get("요양기관"), str):
        r["요양기관"] = {"이름": r["요양기관"]}
    elif not isinstance(r.get("요양기관"), dict):
        r["요양기관"] = {}
    p = r.get("진료기간")
    if isinstance(p, str):                                        # "2026-08-05 ~ 2026-08-05" 처럼 글자 하나
        ds = re.findall(r"\d{4}\D{1,3}\d{1,2}\D{1,3}\d{1,2}", p)
        r["진료기간"] = {"시작": ds[0] if ds else None, "끝": ds[-1] if ds else None}
    elif isinstance(p, list):
        r["진료기간"] = {"시작": p[0] if p else None, "끝": p[-1] if p else None}
    elif not isinstance(p, dict):
        r["진료기간"] = {}
    rows_ = r.get("행")
    if isinstance(rows_, dict):                                   # {"진찰료": {...}} 처럼 묶음으로 준 경우
        r["행"] = [{"항목": k, **v} for k, v in rows_.items() if isinstance(v, dict)]
    elif not isinstance(rows_, list):
        r["행"] = []
    if not isinstance(r.get("병명"), list):
        r["병명"] = [r["병명"]] if isinstance(r.get("병명"), dict) else []
    return r


def to_values(r, doc):
    """읽은 값(JSON) → export_values(visible) 형식 한 장. 꼭 필요한 값을 못 읽었으면 missing 목록에 적는다."""
    r = shape(r)
    v = {"visible_only": True, "institution": inst_of(r.get("요양기관")), "missing": []}
    v["patient"] = {"reg_no": r.get("환자등록번호")}
    if doc == "영수증":
        p, t0, f, pay = r.get("진료기간") or {}, r.get("합계") or {}, r.get("금액산정") or {}, r.get("납부") or {}
        rows = {}
        for it in r.get("행") or []:
            if not isinstance(it, dict):
                continue
            name = canon_row(it.get("항목"))
            if not name:
                continue
            cur = rows.setdefault(name, {"copay": 0, "nhis": 0, "full": 0, "uninsured": 0})
            for src, dst in (("본인부담", "copay"), ("공단부담", "nhis"), ("전액본인부담", "full"), ("비급여", "uninsured")):
                cur[dst] += num(it.get(src)) or 0
        rows = {k: x for k, x in rows.items() if any(x.values())}
        v["institution"]["class"] = v["institution"]["fee_type"] = fit_class(rows, v["institution"])
        t = {"t1": num(t0.get("본인부담")), "t2": num(t0.get("공단부담")), "t3": num(t0.get("전액본인부담")),
             "t4": num(t0.get("비급여")), "t5": num(r.get("상한액초과금")), "t6": num(f.get("진료비총액")),
             "t7": num(f.get("공단부담총액")), "t8": num(f.get("환자부담총액")), "t9": num(f.get("이미납부한금액")),
             "t10": num(f.get("납부할금액"))}
        v["missing"] += [k for k in ("t1", "t2", "t6", "t8", "t10") if t[k] is None]
        t = {k: (x if x is not None else 0) for k, x in t.items()}
        rc = {"visit_start": ymd(p.get("시작")), "visit_end": ymd(p.get("끝")) or ymd(p.get("시작")),
              "issue_date": ymd(r.get("발행일")), "rows": rows, "totals": t}
        if any(num(pay.get(k)) is not None for k in ("카드", "현금영수증", "현금", "합계")):
            rc["payment"] = {"card": num(pay.get("카드")) or 0, "cash_receipt": num(pay.get("현금영수증")) or 0,
                             "cash": num(pay.get("현금")) or 0, "paid_sum": num(pay.get("합계")) or 0,
                             "unpaid": num(pay.get("납부하지않은금액")) or 0}
        visits = [d for d in (ymd(x) for x in (r.get("내원일") or [])) if d] if isinstance(r.get("내원일"), list) else []
        if visits:
            rc["visits_listed"] = visits
        if r.get("영수증번호"):
            rc["receipt_no"] = str(r["영수증번호"]).strip()
        if ymd(r.get("발행일시")):
            rc["issue_stamp_date"] = ymd(r["발행일시"])
        v["missing"] += [k for k in ("visit_start", "issue_date") if not rc[k]]
        v["receipt"] = rc
    else:
        names, codes = {}, []
        sick = [x for x in (r.get("병명") or []) if isinstance(x, dict)]
        sick.sort(key=lambda x: 0 if str(x.get("구분", "")).startswith("주") else 1)
        for x in sick:
            c = canon_code(x.get("코드"))
            if c:
                codes.append(c)
                names[c] = canon_name(c, x.get("병명"))
        title = str(r.get("제목") or "")
        v["diagnosis"] = {"form": "상해진단서" if "상해" in title else "일반진단서",
                          "onset_date": ymd(r.get("발병일")), "diag_date": ymd(r.get("진단일")), "issue_date": ymd(r.get("발행일")),
                          "codes": codes, "names": names, "opinion": str(r.get("소견") or "")}
        v["missing"] += [k for k in ("diag_date", "issue_date") if not v["diagnosis"][k]] + ([] if codes else ["codes"])
    return v


def ask(model, image, prompt, size, max_tokens=2500):
    url, (W, H), sent, _ = oc.image_data_url(image, size)
    r = oc.chat(model, [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                     {"type": "image_url", "image_url": {"url": url}}]}],
                reasoning="off", max_tokens=max_tokens)
    r["W"], r["H"], r["sent_w"] = W, H, sent[0]
    return r


def read(model, image, doc, size=2500):
    return ask(model, image, READ_PROMPT[doc], size)


def reread_prompt(doc, fields):
    """실패한 검사에 쓰인 칸만 다시 읽게 하는 지시문. 같은 이름·형식으로 답하고 칸마다 위치를 준다."""
    return (f"이 '{'진료비 계산서·영수증' if doc == '영수증' else '진단서'}' 사진에서 아래 칸만 아주 천천히 한 글자씩 다시 읽어라. "
            "숫자 하나, 날짜 하나도 추측하지 말고 보이는 그대로 적는다.\n"
            f"다시 읽을 칸: {', '.join(fields)}\n"
            "답은 JSON 하나: {\"값\": {칸 이름: 처음 판독과 같은 형식의 값}, "
            "\"위치\": {칸 이름 또는 '행:항목 이름': [x0, y0, x1, y1]}}\n"
            "- 형식은 처음 판독과 같다 (날짜 YYYY-MM-DD, 금액 쉼표 없는 정수, 행은 [{\"항목\", \"본인부담\", \"공단부담\", \"전액본인부담\", \"비급여\"}])\n"
            "- 행은 금액이 적힌 줄만 적는다 (0이거나 빈 줄은 빼기). '행위료'·'약품비' 작은 줄은 바로 위 큰 항목 이름을 붙인다 (예: \"주사료-약품비\")\n"
            "- 위치는 사진 전체를 가로·세로 0~1000으로 본 상자. 행은 \"행:항목 이름\" 으로, 그 줄의 금액 칸 전체를 감싼다.\n\n"
            "참고로 처음 판독의 형식은 다음과 같았다:\n" + READ_PROMPT[doc].split("\n{", 1)[1].split("\n}", 1)[0][:1500])


def fields_for(items):
    out = []
    for it in items:
        for f in FIELDS_BY_CHECK.get(it["id"], []):
            if f not in out:
                out.append(f)
    return out


def reread(model, image, doc, failed, size=3200):
    fields = fields_for(failed)
    if not fields:
        return None, []
    return ask(model, image, reread_prompt(doc, fields), size, max_tokens=3000), fields


# ---------------------------------------------------------------- 조각 대조 (11-06)
# 서류 전체를 읽을 때 AI 가 합계가 맞아 보이게 숫자를 '고쳐 읽는' 일이 있다 (평가셋 v4 L121: ⑧·⑩ 8,600 을 둘 다 ① 과 같은
# 8,500 으로 적음. 같은 칸을 잘라 따로 읽히면 8,600). 그래서 합계·금액산정·납부 칸은 자리를 받아 잘라서 한 칸씩 다시 읽고,
# 전체 판독과 다르면 조각 값을 쓴다 (자릿수가 같고, 조각 속 칸 이름이 그 칸일 때만 — 엉뚱한 칸을 자른 경우를 막음).
CROP_KEYS = [("합계", "본인부담", "①", ("①", "본인부담")), ("합계", "공단부담", "②", ("②", "공단부담")),
             ("합계", "전액본인부담", "③", ("③", "전액")), ("합계", "비급여", "④", ("④", "비급여", "선택진료")),
             ("금액산정", "진료비총액", "⑥", ("⑥", "진료비")), ("금액산정", "공단부담총액", "⑦", ("⑦", "공단부담")),
             ("금액산정", "환자부담총액", "⑧", ("⑧", "환자부담")), ("금액산정", "납부할금액", "⑩", ("⑩", "납부할")),
             ("납부", "카드", "카드", ("카드",)), ("납부", "현금영수증", "현금영수증", ("현금영수증",)),
             ("납부", "현금", "현금", ("현금",)), ("납부", "합계", "납부 합계", ("합계",))]
# 조각 값은 그 줄의 칸 이름이 맞을 때만 쓴다 (11-06 v4: 상자가 옆 칸을 잘라 26,000 을 20,000 으로 '고쳐' 정상 1장을 헛짚음)
NAME_OK = {(g, k): words for g, k, _, words in CROP_KEYS}


def crop_targets(r):
    """판독값에서 0 이 아닌 합계·금액산정·납부 칸 → [(묶음, 칸, 표시, 값)]."""
    out = []
    for grp, key, label, _ in CROP_KEYS:
        v = num(((r or {}).get(grp) or {}).get(key))
        if v:
            out.append((grp, key, label, v))
    return out


def box_prompt(targets):
    names = ", ".join(f"{g}.{k}({lab})" for g, k, lab, _ in targets)
    return ("이 '진료비 계산서·영수증' 사진에서 아래 칸마다 금액 숫자가 적힌 자리만 상자로 답하라. 값은 적지 않는다.\n"
            f"칸: {names}\n"
            "답은 JSON 하나: {\"위치\": {\"묶음.칸\": [x0, y0, x1, y1]}} — 사진 전체를 가로·세로 0~1000으로 본 좌표, "
            "숫자 글자만 꼭 맞게 감싼다 (칸 이름 글자·옆 칸 숫자는 넣지 않는다)")


CROP_PROMPT = ("사진 조각에서 빨간 상자 안의 금액 숫자를 보이는 그대로 적고, 같은 줄에서 그 숫자 왼쪽에 인쇄된 가장 가까운 칸 이름이나 "
               "기호(예: ①, ⑧ 환자부담 총액, 카드, 합계)를 적어라. 계산하거나 고치지 않는다. "
               "JSON 하나만: {\"숫자\": \"...\", \"이름\": \"...\"}")


def crop_read(model, image, box):
    """상자 왼쪽(칸 이름이 있는 쪽)까지 넓게 잘라 상자를 빨갛게 표시하고 키운 뒤, 상자 안 숫자와 칸 이름을 읽힌다. (값, 이름, 응답)"""
    import base64
    import io
    from PIL import ImageDraw
    x, y, w, h = (int(v) for v in box.split(","))
    oc.refuse_real(image)
    im = Image.open(image).convert("RGB")
    x0, y0 = max(0, x - max(4 * w, 8 * h)), max(0, y - h)
    x1, y1 = min(im.width, x + w + max(8, w // 3)), min(im.height, y + 2 * h)
    c = im.crop((x0, y0, x1, y1))
    ImageDraw.Draw(c).rectangle((x - x0 - 3, y - y0 - 3, x - x0 + w + 3, y - y0 + h + 3), outline=(230, 0, 0), width=3)
    s = max(1.0, 160 / max(1, c.height))
    c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=92)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    r = oc.chat(model, [{"role": "user", "content": [{"type": "text", "text": CROP_PROMPT},
                                                     {"type": "image_url", "image_url": {"url": url}}]}],
                reasoning="off", max_tokens=150)
    j = r.get("json") if isinstance(r.get("json"), dict) else {}
    return num(j.get("숫자")), str(j.get("이름") or ""), r


TIGHT_PROMPT = "사진 조각의 가운데에 보이는 금액 숫자 하나를 보이는 그대로 적어라. 계산하거나 고치지 않는다. JSON 하나만: {\"숫자\": \"...\"}"


def crop_read_tight(model, image, box):
    """상자 둘레만 좁게 잘라 숫자만 읽힌다 (넓게 자른 읽기와 맞춰 보는 두 번째 읽기). (값, 응답)"""
    import base64
    import io
    x, y, w, h = (int(v) for v in box.split(","))
    oc.refuse_real(image)
    im = Image.open(image).convert("RGB")
    mx, my = max(6, round(w * 0.2)), max(4, round(h * 0.45))
    c = im.crop((max(0, x - mx), max(0, y - my), min(im.width, x + w + mx), min(im.height, y + h + my)))
    s = max(1.0, 120 / max(1, c.height))
    c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=92)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    r = oc.chat(model, [{"role": "user", "content": [{"type": "text", "text": TIGHT_PROMPT},
                                                     {"type": "image_url", "image_url": {"url": url}}]}],
                reasoning="off", max_tokens=80)
    j = r.get("json") if isinstance(r.get("json"), dict) else {}
    return num(j.get("숫자")), r


def name_ok(grp, key, name):
    """조각에서 읽은 칸 이름이 그 칸의 이름인가 (현금 ↔ 현금영수증, 합계 줄 ↔ 납부 합계는 구분)."""
    n = re.sub(r"\s", "", name)
    if not n or not any(w in n for w in NAME_OK[(grp, key)]):
        return False
    if (grp, key) == ("납부", "현금") and "영수증" in n:
        return False
    if (grp, key) == ("납부", "합계") and any(m in n for m in "①②③④⑤⑥⑦⑧⑨⑩"):
        return False
    return True


def crop_check(model, image, r1):
    """영수증 판독(JSON) → 합계·금액산정·납부 칸을 잘라 다시 읽은 기록. 바꾸지는 않는다 (apply_crop 이 바꿈)."""
    targets = crop_targets(r1)
    if not targets:
        return None
    rb = ask(model, image, box_prompt(targets), 2500, max_tokens=900)
    out = {"위치_응답": rb.get("text"), "읽음": {}, "비용": rb.get("cost") or 0, "초": rb.get("seconds") or 0}
    j = rb.get("json") if isinstance(rb.get("json"), dict) else {}
    found = j.get("위치") if isinstance(j.get("위치"), dict) else next((x for x in j.values() if isinstance(x, dict)), j)
    pos = {re.sub(r"\(.*?\)\s*$", "", str(k)).strip(): x for k, x in found.items()}   # '합계.본인부담(①)' → '합계.본인부담'
    out["상자_수"] = len(pos)
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        grp, key, label, v = t
        box = to_box(pos.get(f"{grp}.{key}"), rb.get("W"), rb.get("H"), rb.get("sent_w"))
        if not box:
            return None
        try:
            c, name, rc = crop_read(model, image, box)
            c2, rc2 = crop_read_tight(model, image, box)
        except Exception as e:                                    # 조각 하나가 실패해도 나머지는 계속
            return f"{grp}.{key}", {"전체": v, "조각": None, "상자": box, "오류": f"{type(e).__name__}"}, 0, 0
        return (f"{grp}.{key}", {"전체": v, "조각": c, "조각2": c2, "이름": name, "이름_맞음": name_ok(grp, key, name), "상자": box},
                (rc.get("cost") or 0) + (rc2.get("cost") or 0), max(rc.get("seconds") or 0, rc2.get("seconds") or 0))
    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(one, targets):
            if res:
                k, rec, cost, sec = res
                out["읽음"][k] = rec
                out["비용"] += cost
                out["초"] += sec / 4
    return out


def apply_crop(r1, crop):
    """조각 값이 전체 판독과 다르고, 자릿수가 같고, 조각 속 칸 이름이 그 칸이고, 좁게 다시 자른 읽기와도 같으면 조각 값을 쓴다."""
    if not crop:
        return r1, []
    r = json.loads(json.dumps(r1))
    changed = []
    for k, x in (crop.get("읽음") or {}).items():
        c, v = x.get("조각"), x.get("전체")
        if (c is None or v is None or c == v or len(str(abs(c))) != len(str(abs(v))) or not x.get("이름_맞음")
                or x.get("조각2") != c):                          # 넓게·좁게 두 번 잘라 읽은 값이 같을 때만
            continue
        grp, key = k.split(".", 1)
        r.setdefault(grp, {})[key] = c
        changed.append((k, v, c, x.get("상자")))
    return r, changed


def to_box(pos, W, H, sent_w):
    """0~1000 좌표 → 원래 사진 픽셀 'x,y,w,h' (e2_judge 와 같은 규칙)."""
    try:
        x0, y0, x1, y1 = (float(v) for v in pos)
    except (TypeError, ValueError):
        return None
    if max(x0, y0, x1, y1) > 1000:
        s = W / sent_w
        x0, y0, x1, y1 = x0 * s, y0 * s, x1 * s, y1 * s
    else:
        x0, x1, y0, y1 = x0 / 1000 * W, x1 / 1000 * W, y0 / 1000 * H, y1 / 1000 * H
    if x1 <= x0 or y1 <= y0:
        return None
    return f"{round(x0)},{round(y0)},{round(x1 - x0)},{round(y1 - y0)}"
