"""읽기: VLM(OpenRouter) 한 번으로 서류 칸을 읽고 → 로직 엔진(logic_detector)의 판독 형식 Reading 으로 바꾼다.

- 지시문은 detector/reader/ai_reader.READ_PROMPT 를 그대로 쓰고, 로직 규칙이 쓰는 칸(요양기관 종류 · 입원/외래 · 야간/공휴 ·
  환자 구분 · 주민번호 · 입원일/퇴원일)만 덧붙여 묻는다
- 같은 사진 · 같은 모델 · 같은 지시문이면 원본 응답을 캐시에서 꺼낸다 → 같은 평가셋을 다시 돌릴 때 비용 0
- 합성 서류만 보낸다 (or_client.refuse_real 이 보내기 전에 막음)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "detector" / "reader"))
import ai_reader as ar  # noqa: E402
import or_client as oc  # noqa: E402

sys.path.insert(0, str(REPO))
from detector.logic.reading import Reading  # noqa: E402
from detector.logic.read_receipt import ROW_ORDER  # noqa: E402

CACHE = REPO / "outputs" / "combine" / "vlm_cache"
MODELS = {"qwen27b": "qwen/qwen3.8-27b", "deepseek41f": "deepseek/deepseek-v4.1-flash-1017",
          "qwen3vl30b": "qwen/qwen3-vl-30b-a3b-instruct", "qwen3vl8b": "qwen/qwen3-vl-8b-instruct"}

EXTRA = {
    "영수증": """
위 JSON 에 아래 칸도 함께 넣는다 (서식에 없거나 체크가 안 보이면 null):
 "요양기관종류": 아래쪽 요양기관 종류에서 체크(■ 또는 V)된 것 하나 — "의원급" / "병원급" / "종합병원" / "상급종합",
 "입원외래": 위쪽 환자 구분 칸에서 체크된 것 — "외래" / "입원",
 "야간": 야간 칸이 체크되어 있으면 true, 아니면 false,
 "공휴일": 공휴일 칸이 체크되어 있으면 true, 아니면 false,
 "환자구분": 적힌 그대로 (예: 건강보험, 의료급여, 자보),
 "주민등록번호": 적힌 그대로 (가려진 자리는 빼고, 예: 891203-1)""",
    "진단서": """
위 JSON 에 아래 칸도 함께 넣는다 (서식에 없으면 null):
 "주민등록번호": 적힌 그대로 (가려진 자리는 빼고, 예: 891203-1),
 "입원일": "YYYY-MM-DD", "퇴원일": "YYYY-MM-DD" (소견이 아니라 입원·퇴원 칸에 적힌 날짜만)""",
}
PROMPT_VERSION = "c1"     # 지시문을 고치면 올린다 (캐시가 섞이지 않게)

# ai_reader.canon_row 가 돌려주는 이름 → 로직 엔진 행 이름
CANON_TO_ROW = {"진찰료": "consult", "투약 및 조제료-행위료": "med_act", "투약 및 조제료-약품비": "med_drug",
                "주사료-행위료": "inj_act", "주사료-약품비": "inj_drug", "처치 및 수술료": "proc", "검사료": "lab",
                "영상진단료": "imaging", "재활 및 물리치료료": "rehab", "제증명수수료": "cert"}
INST_TYPE = {"의원": "clinic", "병원": "hospital", "종합": "general", "상급": "tertiary"}
COLS = (("본인부담", "self"), ("공단부담", "nhis"), ("전액본인부담", "full"), ("비급여", "nonpay"))


REAL_LIMIT = 20       # 실서류는 허용된 소수(20장 이내)만 외부 AI 로


def allow_real(n):
    """실서류 막기를 이번 실행에서만 끈다. 장수는 REAL_LIMIT 이하만. run.py --실서류허용 에서만 부른다."""
    if n > REAL_LIMIT:
        raise SystemExit(f"실서류는 한 번에 {REAL_LIMIT}장까지만 외부 AI 로 보낼 수 있습니다 (지금 {n}장)")
    oc.refuse_real = lambda path: None


def prompt(doc):
    return ar.READ_PROMPT[doc] + EXTRA[doc]


def _cache_key(image, model, doc, size):
    h = hashlib.sha1(Path(image).read_bytes()).hexdigest()
    return f"{h[:20]}_{doc}_{model.replace('/', '_')}_{size}_{PROMPT_VERSION}.json"


def read(image, doc, model="qwen27b", size=2500, use_cache=True):
    """사진 → {"json": 읽은 값, "cost", "seconds", "provider", "cached", "W", "H", "sent_w"}. 실패하면 json 이 None."""
    oc.refuse_real(image)
    model_id = MODELS.get(model, model)
    cf = CACHE / _cache_key(image, model_id, doc, size)
    if use_cache and cf.exists():
        r = json.loads(cf.read_text(encoding="utf-8"))
        return {**r, "cached": True, "cost": 0, "seconds": 0}
    r = ar.ask(model_id, image, prompt(doc), size)
    if r.get("ok") and not ar.usable(r.get("json"), doc):             # 빈 틀만 돌려주면 크기를 바꿔 한 번 더
        r2 = ar.ask(model_id, image, prompt(doc), 3200)
        r2["cost"] = (r2.get("cost") or 0) + (r.get("cost") or 0)
        r2["seconds"] = (r2.get("seconds") or 0) + (r.get("seconds") or 0)
        r = r2
    keep = {k: r.get(k) for k in ("ok", "json", "text", "error", "cost", "seconds", "provider", "W", "H", "sent_w")}
    if keep["ok"] and keep["json"]:
        CACHE.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    return {**keep, "cached": False}


def _row_key(name):
    """VLM 이 적은 항목 이름 → 로직 엔진 행 이름. 모르는 이름은 None."""
    canon = ar.canon_row(name)
    if canon is None:
        return None
    if canon in CANON_TO_ROW:
        return CANON_TO_ROW[canon]
    n = re.sub(r"[\s·ㆍ_()\-]", "", str(name))
    for key, labels in ROW_ORDER:
        if key in ("total", "cap", "med_act", "med_drug", "inj_act", "inj_drug", "etc_early"):
            continue
        if any(lab.replace("ㆍ", "").replace("·", "") in n for lab in labels):
            return key
    return None


def _inst_type(s):
    s = str(s or "")
    for word, key in INST_TYPE.items():
        if s.startswith(word) or (word == "상급" and "상급" in s):
            return key
    return None


def to_reading(j, doc):
    """VLM 판독(JSON) → logic_detector Reading. 돌려주는 것: (Reading, 메모 목록)."""
    r = ar.shape(j)
    num, ymd = ar.num, ar.ymd
    notes = []
    if doc == "영수증":
        inst = r.get("요양기관") or {}
        p = r.get("진료기간") or {}
        f = {"form": "2024",
             "institution.name": inst.get("이름"), "institution.address": inst.get("주소"),
             "institution.bizno": ar.inst_of(inst).get("biz_no"),
             "institution.type": _inst_type(r.get("요양기관종류")),
             "patient.rrn": r.get("주민등록번호"), "patient.type": r.get("환자구분"),
             "visit.start": ymd(p.get("시작")), "visit.end": ymd(p.get("끝")) or ymd(p.get("시작")),
             "receipt_no": str(r["영수증번호"]).strip() if r.get("영수증번호") else None,
             "issue_date": ymd(r.get("발행일"))}
        io_ = str(r.get("입원외래") or "")
        if io_:
            f["visit.inpatient"] = "입원" in io_
        for k_src, k_dst in (("야간", "visit.night"), ("공휴일", "visit.holiday")):
            if isinstance(r.get(k_src), bool):
                f[k_dst] = r[k_src]
        unknown = 0
        for it in r.get("행") or []:
            if not isinstance(it, dict):
                continue
            key = _row_key(it.get("항목"))
            if key is None:
                if ar.canon_row(it.get("항목")) is not None:
                    unknown += 1
                    notes.append(f"모르는 항목 이름: {it.get('항목')}")
                continue
            for src, col in COLS:                                     # 읽은 행은 4열을 모두 채운다 (서식의 빈 칸 = 0)
                k = f"rows.{key}.{col}"
                f[k] = (f.get(k) or 0) + (num(it.get(src)) or 0)
        t, s, pay = r.get("합계") or {}, r.get("금액산정") or {}, r.get("납부") or {}
        totals = {"totals.t1": t.get("본인부담"), "totals.t2": t.get("공단부담"), "totals.t3": t.get("전액본인부담"),
                  "totals.t4": t.get("비급여"), "totals.cap": r.get("상한액초과금"),
                  "grand_total": s.get("진료비총액"), "nhis_total": s.get("공단부담총액"), "patient_total": s.get("환자부담총액"),
                  "already_paid": s.get("이미납부한금액"), "to_pay": s.get("납부할금액")}
        have_totals = any(num(v) is not None for v in totals.values())
        for k, v in totals.items():                                   # 금액 칸을 읽었으면 빈 칸은 0 (서식 관례, ai_reader.to_values 와 같음)
            if num(v) is not None or have_totals:
                f[k] = num(v) or 0
        if any(num(pay.get(k)) is not None for k in ("카드", "현금영수증", "현금", "합계")):
            for src, dst in (("카드", "paid.card"), ("현금영수증", "paid.cash_receipt"), ("현금", "paid.cash"), ("합계", "paid.total")):
                f[dst] = num(pay.get(src)) or 0
        f["rows_complete"] = bool(r.get("행")) and unknown == 0 and num(t.get("본인부담")) is not None
        return Reading("receipt", f), notes
    diag = []
    sick = sorted((x for x in (r.get("병명") or []) if isinstance(x, dict)),
                  key=lambda x: 0 if str(x.get("구분", "")).startswith("주") else 1)
    for i, x in enumerate(sick):
        code = ar.canon_code(x.get("코드"))
        if code:
            diag.append({"code": code, "name": ar.canon_name(code, x.get("병명")), "main": i == 0})
    f = {"patient.rrn": r.get("주민등록번호"), "diagnoses": diag or None,
         "dates.onset": ymd(r.get("발병일")), "dates.diagnosis": ymd(r.get("진단일")), "dates.issue": ymd(r.get("발행일")),
         "dates.admit": ymd(r.get("입원일")), "dates.discharge": ymd(r.get("퇴원일")),
         "opinion": str(r.get("소견") or "") or None,
         "institution.name": (r.get("요양기관") or {}).get("이름"), "institution.address": (r.get("요양기관") or {}).get("주소")}
    return Reading("diagnosis", f), notes


# ── 근거 위치: 로직 규칙이 걸린 칸만 VLM 에게 상자를 한 번 더 묻는다 (걸린 서류만, 1회) ────────────────
FIELD_LABEL = {"totals.t1": "합계 줄 ① 본인부담금", "totals.t2": "합계 줄 ② 공단부담금", "totals.t3": "합계 줄 ③ 전액본인부담",
               "totals.t4": "합계 줄 ④ 비급여", "totals.cap": "⑤ 상한액 초과금", "grand_total": "⑥ 진료비 총액",
               "nhis_total": "⑦ 공단부담 총액", "patient_total": "⑧ 환자부담 총액", "already_paid": "⑨ 이미 납부한 금액",
               "to_pay": "⑩ 납부할 금액", "paid.card": "카드", "paid.cash_receipt": "현금영수증", "paid.cash": "현금",
               "paid.total": "납부 합계", "visit.start": "진료기간 시작일", "visit.end": "진료기간 끝 날짜", "issue_date": "발행일",
               "receipt_no": "영수증번호", "institution.bizno": "사업자등록번호", "patient.rrn": "주민등록번호",
               "dates.onset": "발병 연월일", "dates.diagnosis": "진단 연월일", "dates.issue": "발행일", "diagnoses": "질병분류기호",
               "dates.admit": "입원일", "dates.discharge": "퇴원일"}


def locate(image, fields, model="qwen27b", size=2500):
    """칸 이름들 → {칸: 'x,y,w,h'(원래 사진 픽셀)}. 행(rows.*) 칸은 줄 이름으로 묻는다."""
    labels = {}
    for fld in fields:
        if fld.startswith("rows."):
            _, row, col = fld.split(".")
            name = next((k for k, v in CANON_TO_ROW.items() if v == row), row)
            labels[fld] = f"{name} 줄의 {dict((c, s) for s, c in COLS).get(col, col)}"
        elif fld in FIELD_LABEL:
            labels[fld] = FIELD_LABEL[fld]
    if not labels:
        return {}, 0
    q = ("이 서류 사진에서 아래 칸마다 값이 적힌 자리만 상자로 답하라. 값은 적지 않는다.\n칸: "
         + ", ".join(f"{k}({v})" for k, v in labels.items())
         + "\n답은 JSON 하나: {\"위치\": {\"칸\": [x0, y0, x1, y1]}} — 사진 전체를 가로·세로 0~1000으로 본 좌표, 값 글자만 꼭 맞게 감싼다")
    r = ar.ask(MODELS.get(model, model), image, q, size, max_tokens=900)
    j = r.get("json") if isinstance(r.get("json"), dict) else {}
    pos = j.get("위치") if isinstance(j.get("위치"), dict) else j
    out = {}
    for k in labels:
        b = ar.to_box(pos.get(k), r.get("W"), r.get("H"), r.get("sent_w"))
        if b:
            out[k] = b
    return out, r.get("cost") or 0
