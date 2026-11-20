"""서식에 인쇄된 산식으로 계산하기 (A1).

로직 엔진의 A1 은 새 서식(2024) 번호에 맞춘 식(⑥ = ①+②+③+④ …)을 코드에 박아 두고 계산한다. 그런데 실제 영수증에는
옛 서식(⑦ 진료비 총액 = ①+②+③+④+⑤, 선택진료료 열)과 병원별 변형이 섞여 있어(실제 서류 약 1/4), 번호가 어긋나면 오탐이 난다.
그래서 VLM 이 동그라미 번호 칸마다 '번호 · 인쇄된 이름 · 괄호 안에 인쇄된 계산식 · 값'을 읽어 오고, 여기서는 그 서류에
찍힌 식 그대로 계산한다. 인쇄된 식을 하나도 못 읽었으면 엔진의 기존 식을 그대로 쓴다 (engines.logic).

    cells = parse(vlm_json["번호칸"])   → {"⑥": {"name": "진료비 총액", "formula": "①+②+③+④", "value": 1234}, ...}
    check(cells)                        → A1 근거 목록 (엔진 근거와 같은 모양)
"""
import ast
import re

CIRCLED = {chr(0x2460 + i): i + 1 for i in range(20)}          # ① ~ ⑳
TOL = 10                                                         # 항 하나당 원 단위 절사 허용 (엔진 ROUND_TOL 과 같음)
SOURCE = "서식에 인쇄된 금액산정 산식 (요양급여규칙 별지 제6호서식 — 옛 서식 · 개정 서식 공통)"


# 인쇄된 식이 깨져 읽혔을 때 쓰는 표준식: 칸 이름으로 찾는다. 서식 버전은 '진료비 총액'이 붙은 번호로 가린다
# (개정 서식 ⑥ 진료비 총액 / 옛 서식 ⑦ 진료비 총액). 요양급여규칙 별지 제6호서식.
STANDARD = {
    "new": {"진료비총액": "①+②+③+④", "공단부담총액": "②+⑤", "환자부담총액": "(①-⑤)+③+④", "납부할금액": "⑧-⑨"},
    "old": {"진료비총액": "①+②+③+④+⑤", "환자부담총액": "①-⑥+③+④+⑤", "납부할금액": "⑧-⑨"},
}


def _key(name):
    return re.sub(r"[\s·()]", "", str(name or ""))


def version(cells):
    """'진료비 총액' 칸 번호로 서식 버전: ⑥ → new, ⑦ → old, 모르면 None."""
    for k, c in cells.items():
        if _key(c["name"]) == "진료비총액":
            return {"⑥": "new", "⑦": "old"}.get(k)
    return None


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


def _mark(s):
    """'⑥' · '6' · '(6)' → '⑥'. 모르면 None."""
    s = str(s or "").strip()
    for ch in s:
        if ch in CIRCLED:
            return ch
    m = re.search(r"\d+", s)
    if m and 1 <= int(m.group()) <= 20:
        return chr(0x2460 + int(m.group()) - 1)
    return None


def parse(items):
    """VLM '번호칸' 목록 → {번호: {name, formula, value}}. 같은 번호가 여러 번이면 값이 있는 첫 칸."""
    out = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        k = _mark(it.get("번호"))
        if not k:
            continue
        cell = {"name": str(it.get("이름") or "").strip(), "formula": _clean(it.get("식")), "value": _num(it.get("값"))}
        if k not in out or (out[k]["value"] is None and cell["value"] is not None):
            out[k] = cell
    return out


def _clean(f):
    """인쇄된 식에서 번호 · + · - · 괄호만 남긴다. 번호가 없으면 None."""
    if not f:
        return None
    s = str(f).replace("－", "-").replace("−", "-").replace("＋", "+").replace("（", "(").replace("）", ")")
    s = "".join(ch for ch in s if ch in CIRCLED or ch in "+-()")
    return s if any(ch in CIRCLED for ch in s) else None


def _ok(formula, values):
    """식이 계산되는 모양인지 (괄호 짝 · 번호 사이 연산자 · 쓰인 번호의 값)."""
    return formula is not None and evaluate(formula, values) is not None


def terms(formula):
    return [ch for ch in formula if ch in CIRCLED]


def evaluate(formula, values):
    """'(①-⑤)+③+④' 와 값 → 정수. 쓰인 번호 중 값이 없으면 None. 식이 깨졌으면 None."""
    expr = ""
    for ch in formula:
        if ch in CIRCLED:
            if values.get(ch) is None:
                return None
            expr += f" ({values[ch]}) "        # 괄호로 감싸 연산자가 빠진 '①⑥' 이 101 처럼 숫자로 붙지 않고 깨진 식이 되게
        else:
            expr += ch
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.USub, ast.UAdd, ast.Constant)
    if not all(isinstance(n, ok) for n in ast.walk(tree)):
        return None
    return int(eval(compile(tree, "<식>", "eval")))              # 숫자 · + · - · 괄호만 남은 식


def _values(cells):
    """값을 하나라도 읽었으면 빈 칸(null)은 0 (서식 관례: 해당 없음 = 빈 칸)."""
    have = sum(c["value"] is not None for c in cells.values()) >= 3
    return {k: (c["value"] if c["value"] is not None else (0 if have else None)) for k, c in cells.items()}


def formulas(cells):
    """칸마다 쓸 식: 인쇄된 식이 계산되면 그 식, 깨졌거나 없으면 이름으로 찾은 표준식. → {번호: (식, '인쇄' | '표준')}"""
    values, ver, out = _values(cells), version(cells), {}
    for k, c in cells.items():
        if _ok(c["formula"], values) and k not in terms(c["formula"]):
            out[k] = (c["formula"], "인쇄")
        elif ver and _key(c["name"]) in STANDARD[ver] and (c["formula"] or _key(c["name"]) != "공단부담총액"):
            std = STANDARD[ver][_key(c["name"])]
            if _ok(std, values) and k not in terms(std):
                out[k] = (std, "표준")
    return out


def check(cells):
    """식이 있는 칸마다 계산해 값과 비교. 납부할·납부한 금액 줄은 참고(감면·할인 줄이 따로 있을 수 있음), 나머지는 확정."""
    values = _values(cells)
    found = []
    for k, (f, how) in formulas(cells).items():
        c = cells[k]
        if c["value"] is None:
            continue
        want = evaluate(f, values)
        n = len(terms(f))
        if abs(c["value"] - want) >= TOL * n:
            pay = "납부" in c["name"]
            title = f"{'인쇄된' if how == '인쇄' else '서식'} 산식 불일치: {k} {c['name']} = {f}"
            found.append({"rule": "A1", "grade": "참고" if pay else "확정",
                          "fields": [f"num.{x}" for x in terms(f)] + [f"num.{k}"], "title": title,
                          "detail": "서류에 인쇄된 식으로 계산" if how == "인쇄" else f"인쇄된 식이 깨져 읽혀 이름({c['name']})으로 찾은 {'개정' if version(cells) == 'new' else '옛'} 서식 표준식으로 계산", "expected": want, "actual": c["value"], "source": SOURCE,
                          "caveat": "감면·할인·미수금 줄이 따로 있을 수 있음" if pay else ""})
    return found


def usable(cells):
    """계산할 식이 하나라도 있으면 True (이때만 엔진의 박아 둔 식을 대신한다)."""
    return bool(formulas(cells))
