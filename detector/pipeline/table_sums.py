"""서식에 상관없이 표에 적힌 값으로 합계 맞추기 (A2 열 합계 · 행 합계).

로직 엔진 A2 는 개정 서식 4열(본인부담 · 공단부담 · 전액본인부담 · 비급여)과 정해진 항목 이름에 맞춰 계산한다. 옛 서식(선택진료료 ·
선택진료료 이외 열)이나 병원별로 열 · 줄이 다른 서식은 칸을 못 맞춰 검사를 건너뛰거나 엉뚱한 칸끼리 더한다.
그래서 VLM 이 표를 '보이는 그대로' 읽어 오게 한다 — 열 이름 목록, 금액이 적힌 줄마다 열 순서대로의 값, 맨 아래 합계 줄.
여기서는 열 이름 · 항목 이름을 해석하지 않고 숫자만으로 맞춘다:
    열마다   (금액이 적힌 줄의 합) == 합계 줄 값
    합계 줄  (열 합) == 표에 '소계' · '합계' 열이 따로 있으면 그 값
본인부담금 · 공단부담금 열은 둘을 더해서 맞춘다: 합계 줄에서 본인부담금을 절사(외래 100원 미만 · 10원 미만, 국민건강보험법
시행령 [별표 2])하면 그 차이가 공단부담금으로 넘어가 열 하나씩은 수십 원 어긋나는 게 정상이다 (실제 서류에서 확인, 엔진 A2 와 같음).
본인부담금 열 혼자 절사 범위를 벗어나면 참고만.

    table = parse(vlm_json["금액표"])  → {"cols": [...], "rows": [{"name", "vals"}], "total": [...]}
    check(table, inpatient)            → A2 근거 목록 (엔진 근거와 같은 모양)
"""
import re

TOL = 10                                       # 원 단위 절사 허용 (엔진 ROUND_TOL 과 같음)
SOURCE = "표에 적힌 금액 (서식 무관: 열마다 항목 줄 합 = 합계 줄)"
SUBTOTAL = re.compile(r"소계|계$|합$")
SELF = re.compile(r"본인\s*부담\s*금?$|^일부\s*본인|본인부담금")
NHIS = re.compile(r"공단|보험자")
FULL = re.compile(r"전액")


def _num(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"[^\d-]", "", str(v))
    try:
        return int(s)
    except ValueError:
        return None


def parse(t):
    """VLM '금액표' → {"cols": [열 이름], "rows": [{"name", "vals": [정수|None]}], "total": [정수|None]}. 못 쓰면 None."""
    if not isinstance(t, dict):
        return None
    cols = [str(c or "").strip() for c in (t.get("열") or [])]
    if len(cols) < 2:
        return None
    n = len(cols)

    def vals(xs):
        xs = list(xs or [])[:n]
        return [_num(x) for x in xs] + [None] * (n - len(xs))

    rows = []
    for r in t.get("줄") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("항목") or "").strip()
        if "합계" in name.replace(" ", ""):
            continue
        v = vals(r.get("값"))
        if any(x for x in v):
            rows.append({"name": name, "vals": v})
    total = vals(t.get("합계"))
    if not rows or all(x is None for x in total):
        return None
    return {"cols": cols, "rows": rows, "total": total}


def _allowed_self(raw, inpatient):
    ok = {raw, raw // 10 * 10, round(raw, -1)}
    if not inpatient:
        ok |= {raw // 100 * 100, round(raw, -2)}
    return ok


def check(table, inpatient=False):
    """열마다 항목 줄 합과 합계 줄 비교 + 소계 열이 있으면 줄마다 열 합과 비교."""
    cols, rows, total = table["cols"], table["rows"], table["total"]
    found = []
    sub = [i for i, c in enumerate(cols) if SUBTOTAL.search(c.replace(" ", ""))]
    parts = [i for i in range(len(cols)) if i not in sub]
    raw = [sum(r["vals"][i] or 0 for r in rows) for i in range(len(cols))]
    fields = lambda i: [f"table.{i}.{k}" for k, r in enumerate(rows) if r["vals"][i]] + [f"table.{i}.total"]   # noqa: E731
    self_i = next((i for i, c in enumerate(cols) if SELF.search(c) and not FULL.search(c)), None)
    nhis_i = next((i for i, c in enumerate(cols) if NHIS.search(c)), None)
    pair = self_i is not None and nhis_i is not None and total[self_i] is not None and total[nhis_i] is not None

    def add(grade, flds, title, want, got, caveat=""):
        found.append({"rule": "A2", "grade": grade, "fields": flds, "title": title, "detail": "서식과 상관없이 표에 적힌 값으로 계산",
                      "expected": want, "actual": got, "source": SOURCE, "caveat": caveat})

    for i, col in enumerate(cols):
        if total[i] is None or (pair and i in (self_i, nhis_i)):
            continue
        if abs(total[i] - raw[i]) >= TOL:
            add("확정", fields(i), f"열 합계 불일치: {col} 합계 줄 ≠ 항목 줄 합", raw[i], total[i])
    if pair:
        want, got = raw[self_i] + raw[nhis_i], total[self_i] + total[nhis_i]
        if abs(got - want) >= 2 * TOL:
            add("확정", fields(self_i) + fields(nhis_i), f"열 합계 불일치: {cols[self_i]}+{cols[nhis_i]} 합계 줄 ≠ 항목 줄 합", want, got)
        if total[self_i] not in _allowed_self(raw[self_i], inpatient):
            add("참고", fields(self_i), f"{cols[self_i]} 합계 줄이 항목 줄 합(또는 절사)과 다름", raw[self_i], total[self_i],
                "외래 100원 미만 · 10원 미만 절사 허용, 그 밖의 차이는 감면 줄일 수 있음")
    if len(sub) == 1 and len(parts) >= 2:                                  # 소계 열: 같은 줄의 다른 열 합 (선택진료료 열 등은 서식마다 달라 강함 아님)
        s = sub[0]
        for k, r in enumerate(rows + [{"name": "합계", "vals": total}]):
            if r["vals"][s] is None:
                continue
            want = sum(r["vals"][i] or 0 for i in parts if i < s)
            if want and abs(r["vals"][s] - want) >= TOL * 2:
                found.append({"rule": "A2", "grade": "참고", "fields": [f"table.{s}.{k}"],
                              "title": f"줄 소계 불일치: {r['name']} {cols[s]} ≠ 앞 열 합",
                              "detail": "소계 열이 어느 열까지 더하는지는 서식마다 다를 수 있음", "expected": want,
                              "actual": r["vals"][s], "source": SOURCE, "caveat": "소계 범위가 서식마다 다름"})
    return found
