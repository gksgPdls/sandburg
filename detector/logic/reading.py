"""
판독 결과 = Reading. 판독기(OCR·VLM)가 이미지에서 읽은 서식 칸 값을 이 형식으로 넘기면 규칙이 검사한다.
규칙은 이미지를 보지 않는다 → 판독기를 바꿔도 규칙은 그대로.

    r = Reading("receipt", {
        "institution.type": "clinic",
        "visit.start": "2024-03-08",
        "rows.consult.self": {"value": 5283, "box": (x, y, w, h)},
        "totals.t1": {"value": 16200, "uncertain": True},   # 판독기 2개가 다르게 읽은 칸
        ...
    })

칸 값은 그냥 값 또는 {"value", "box", "uncertain"}. 못 읽은 칸은 넣지 않는다(None과 같음).

칸 이름 (진료비 계산서·영수증, 요양급여규칙 별지 제6호서식)
  form                 "2024"(2024.7.18 개정, 열 4개) / "old"(예전 서식, 선택진료료 열 포함). 없으면 "2024"
  institution.type     clinic(의원) / hospital(병원) / general(종합병원) / tertiary(상급종합) — 하단 요양기관 종류 체크
  institution.name / institution.bizno / institution.address
  patient.rrn          주민번호 (뒷자리 가린 경우 앞 6자리 + 성별 1자리만, 예: "891203-1")
  patient.type         건강보험 / 의료급여 / 산재 / 자보 ...
  visit.inpatient      True(입원) / False(외래)
  visit.start / visit.end      진료기간 (ISO 날짜)
  visit.night / visit.holiday  야간·공휴일 진료 체크
  receipt_no / issue_date
  rows_complete        True면 금액 행을 전부 읽었음 (열 합계 검산에 필요)
  rows.<행>.<열>        행: consult(진찰료) room_1 room_23 room_4 meal med_act med_drug inj_act inj_drug anesth proc lab
                       imaging radio material rehab psych blood ct mri pet us prosth cert selective senior ... etc
                       열: self(본인부담금) nhis(공단부담금) full(전액본인부담) nonpay(비급여)
  totals.t1 ~ t4       열 합계 ①②③④ (예전 서식은 t1 ~ t5: 본인부담·공단부담·전액본인·선택진료료·선택진료료 이외)
  totals.cap           상한액 초과금 (개정 ⑤ / 예전 ⑥)
  grand_total          진료비 총액 (개정 ⑥ / 예전 ⑦)
  nhis_total(⑦, 개정만) patient_total(⑧) already_paid(⑨) to_pay(⑩)
  paid.card / paid.cash_receipt / paid.cash / paid.total(⑪)

진단서 (의료법 시행규칙 별지 제5호의2서식)
  patient.rrn, diagnoses = [{"code", "name", "main": bool}] (칸 하나로: {"value": [...], "box": ...})
  dates.onset / dates.diagnosis / dates.issue / dates.admit / dates.discharge
"""
from datetime import date


class Reading:
    def __init__(self, doc_type, fields):
        self.doc_type = doc_type
        self.fields = {k: v if isinstance(v, dict) and "value" in v else {"value": v} for k, v in fields.items()
                       if v is not None and not (isinstance(v, dict) and "value" in v and v["value"] is None)}

    def get(self, path, default=None):
        f = self.fields.get(path)
        return default if f is None else f["value"]

    def date(self, path):
        v = self.get(path)
        if v is None or isinstance(v, date):
            return v
        try:
            return date.fromisoformat(str(v))
        except ValueError:
            return None

    def box(self, path):
        f = self.fields.get(path)
        return f.get("box") if f else None

    def uncertain(self, path):
        f = self.fields.get(path)
        return bool(f and f.get("uncertain"))

    def has(self, *paths):
        return all(p in self.fields for p in paths)

    def rows(self):
        """{행: {열: 값}} — 읽은 칸만."""
        out = {}
        for k, f in self.fields.items():
            if k.startswith("rows."):
                _, row, col = k.split(".")
                out.setdefault(row, {})[col] = f["value"]
        return out
