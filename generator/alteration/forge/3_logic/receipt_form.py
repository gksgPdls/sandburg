"""C6: 진료비 계산서·영수증 서식 규칙. 한국어 OCR(ocr_ko.py)로 칸 이름을 찾아 각 금액이 무엇인지 정하고,
서식의 계산 관계를 모두 맞추며 바꾼다.

서식 (보건복지부 진료비 계산서·영수증)
  열: 급여[본인부담금 c1, 공단부담금 c2, 전액본인부담 c3]  비급여[선택진료료 c4, 선택진료료 이외 c5]
  합계 줄: ① ② ③ ④ ⑤ (각 열 합계. ①은 본인부담 합을 100원 단위로 내리는 경우가 많고 ② = 급여 합 - ①)
  ⑥ 상한액 초과금
  ⑦ 진료비 총액 = ①+②+③+④+⑤
  ⑧ 환자부담 총액 = ①-⑥+③+④+⑤ (100원 미만 내림인 경우가 많음)
  ⑨ 이미 납부한 금액,  ⑩ 납부할 금액 = ⑧-⑨
  ⑪ 납부한 금액: 카드 + 현금영수증 + 현금 = 합계,  납부하지 않은 금액 = ⑩-⑪
바꾸는 방법 (차이는 100원 단위라 내림 규칙이 그대로 맞음)
  본인부담금 항목  -> ①, ⑦, ⑧, ⑩, 납부 수단 하나와 ⑪ 합계
  공단부담금 항목  -> ②, ⑦ (환자가 내는 돈은 그대로)
  전액본인·비급여 항목 -> ③/④/⑤, ⑦, ⑧, ⑩, 납부 수단 하나와 ⑪ 합계
찾은 칸들로 위 식을 먼저 계산해 보고, 맞는 식에 들어 있는 칸만 고친다. 꼭 바꿔야 할 칸을 못 찾았거나
식이 안 맞으면 그 방법은 쓰지 않는다."""
import re
from difflib import SequenceMatcher

AMT = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _norm(t):
    return re.sub(r"[\s·.,:()\[\]（）_\-~]", "", t)


def _like(text, key, th=0.75):
    t = _norm(text)
    if key in t:
        return True
    if len(t) < 2:
        return False
    n = len(key)
    best = max((SequenceMatcher(None, t[i:i + n], key).ratio() for i in range(max(1, len(t) - n + 1))), default=0)
    return best >= th


def _cx(c):
    return c["x"] + c["w"] / 2


def _cy(c):
    return c["y"] + c["h"] / 2


def _amount(text):
    """칸 글자에서 금액. '6,000②', '⑤13,160' 같은 번호 붙은 것도 금액만. 금액이 아니면 None."""
    t = text.replace(" ", "").replace(".", ",")
    toks = AMT.findall(t)
    if not toks:
        return None
    tok = max(toks, key=len)
    if "," in tok and not re.fullmatch(r"\d{1,3}(,\d{3})+", tok):
        return None
    return int(tok.replace(",", ""))


def find_layout(ko, nums):
    """ko: 한국어 OCR 결과 [{text,x,y,w,h}], nums: 금액 칸 [{x,y,w,h,text,amount}] (logic_list.read_cells).
    서식이 아니면 None. 맞으면 dict(열 항목, 합계칸, 오른쪽 칸들, 확인된 식)."""
    lab = lambda key, th=0.75: [o for o in ko if _like(o["text"], key, th)]
    h1, h2 = lab("본인부담금"), lab("공단부담금")
    if not h1 or not h2:
        return None
    hb, hg = min(h1, key=lambda o: o["y"]), min(h2, key=lambda o: o["y"])
    head_y = max(hb["y"] + hb["h"], hg["y"] + hg["h"])
    h3 = [o for o in lab("전액") if o["y"] < head_y + 30 and o["x"] > hg["x"]]
    sel = sorted([o for o in lab("선택진료") if abs(o["y"] - hb["y"]) < 80 and o["x"] > hg["x"]], key=lambda o: o["x"])
    heads = {"c1": hb, "c2": hg}
    if h3:
        heads["c3"] = min(h3, key=lambda o: o["x"])
    if len(sel) >= 2:
        heads["c4"], heads["c5"] = sel[0], sel[-1]
    elif len(sel) == 1:
        heads["c4"] = sel[0]
    order = sorted(heads, key=lambda k: _cx(heads[k]))
    xs = [_cx(heads[k]) for k in order]
    right_edge = xs[-1] + (xs[-1] - xs[-2]) / 2 if len(xs) >= 2 else xs[-1] + 60
    # 합계 줄: 왼쪽에 있는 '합계' 중 머리줄 아래
    sums = [o for o in lab("합계", 0.99) if o["y"] > head_y and _cx(o) < xs[0]]
    if not sums:
        return None
    srow = min(sums, key=lambda o: o["y"])
    sy = _cy(srow)

    def col_of(c):
        cx = _cx(c)
        k = min(order, key=lambda k: abs(_cx(heads[k]) - cx))
        # 금액은 칸 오른쪽 정렬이라 가운데가 머리 글자보다 오른쪽에 있음. 열 간격 절반 넘게 벗어나면 그 열이 아님
        gap = (xs[-1] - xs[0]) / max(len(xs) - 1, 1)
        return k if abs(_cx(heads[k]) - cx) < 0.75 * gap else None

    items = {k: [] for k in order}
    totals = {}
    for c in nums:
        if c["amount"] is None or _cx(c) > right_edge:
            continue
        k = col_of(c)
        if k is None:
            continue
        if abs(_cy(c) - sy) < max(c["h"], srow["h"]):
            totals[k] = c
        elif head_y < _cy(c) < srow["y"]:
            items[k].append(c)
    # 오른쪽 금액산정 칸: 이름 오른쪽 같은 줄(또는 바로 아래 한 줄 안)의 금액
    right = [c for c in nums if c["amount"] is not None and _cx(c) > right_edge]

    def value_for(keys, th=0.75):
        cands = [o for o in ko if any(_like(o["text"], k, th) for k in keys) and _cx(o) > right_edge - 40]
        for o in sorted(cands, key=lambda o: o["y"]):
            row = [c for c in right if c["x"] > o["x"] + o["w"] * 0.5 and abs(_cy(c) - _cy(o)) < 1.5 * max(o["h"], c["h"])]
            if row:
                return min(row, key=lambda c: abs(_cy(c) - _cy(o)) * 10 + (c["x"] - o["x"]) * 0.01)
        return None
    R = {"r7": value_for(["진료비총액"]), "r8": value_for(["환자부담총액"]),
         "r9": value_for(["이미납부한금액", "이미납부"]), "r10": value_for(["납부할금액"]),
         "card": value_for(["카드"], 0.99), "cashr": value_for(["현금영수증"]), "unpaid": value_for(["납부하지않은금액", "납부하지"])}
    cash = [o for o in ko if _norm(o["text"]) == "현금" and _cx(o) > right_edge - 40]
    if cash:
        o = cash[0]
        row = [c for c in right if c["x"] > o["x"] and abs(_cy(c) - _cy(o)) < 1.5 * max(o["h"], c["h"])]
        R["cash"] = min(row, key=lambda c: c["x"]) if row else None
    psum = [o for o in lab("합계", 0.99) if _cx(o) > right_edge - 40]
    if psum:
        o = min(psum, key=lambda o: o["y"])
        row = [c for c in right if c["x"] > o["x"] and abs(_cy(c) - _cy(o)) < 1.5 * max(o["h"], c["h"])]
        R["paid"] = min(row, key=lambda c: c["x"]) if row else None
    R = {k: v for k, v in R.items() if v is not None}
    L = {"order": order, "items": items, "totals": totals, "right": R}
    L["ok"] = check(L)
    return L


def _v(c):
    return c["amount"] if c is not None else None


def check(L):
    """찾은 칸으로 서식 식을 계산해서 맞는 식 이름 목록. ('①내림'은 ①이 본인부담 합을 100원 내림한 경우)"""
    T, I, R = L["totals"], L["items"], L["right"]
    ok = set()
    s = {k: sum(c["amount"] for c in I[k]) for k in I}
    for k in ("c3", "c4", "c5"):
        if k in T and I.get(k) and s[k] == T[k]["amount"]:
            ok.add(k)
        if k in T and not I.get(k) and T[k]["amount"] == 0:
            ok.add(k)
    if "c1" in T and "c2" in T and I.get("c1") is not None:
        g = s["c1"] + s["c2"]
        if T["c1"]["amount"] + T["c2"]["amount"] == g:
            if T["c1"]["amount"] == s["c1"]:
                ok.add("①합")
            elif T["c1"]["amount"] == s["c1"] // 100 * 100:
                ok.add("①내림")
            ok.add("①+②")
    tot = sum(T[k]["amount"] for k in ("c1", "c2", "c3", "c4", "c5") if k in T)
    if "r7" in R and len(T) == len(L["order"]) and R["r7"]["amount"] == tot:
        ok.add("⑦")
    pt = sum(T[k]["amount"] for k in ("c1", "c3", "c4", "c5") if k in T)
    if "r8" in R:
        v8 = R["r8"]["amount"]
        if v8 == pt:
            ok.add("⑧")
        elif v8 == pt // 100 * 100:
            ok.update({"⑧", "⑧내림"})
        elif v8 == pt // 10 * 10:
            ok.update({"⑧", "⑧10내림"})
    if "r10" in R and "r8" in R:
        if R["r10"]["amount"] == R["r8"]["amount"] - (_v(R.get("r9")) or 0):
            ok.add("⑩")
    pays = [R[k]["amount"] for k in ("card", "cashr", "cash") if k in R]
    if "paid" in R and pays and sum(pays) == R["paid"]["amount"]:
        ok.add("⑪")
    return ok


def plans(L, rng, new_amount, fmt):
    """바꿀 칸 목록 후보들을 차례로. [(칸, 원래 글자, 새 글자)], 설명"""
    T, I, R, ok = L["totals"], L["items"], L["right"], L["ok"]
    kinds = []
    if "①+②" in ok and "c1" in T:
        kinds.append(("c1", "본인부담"))
    if "①+②" in ok and "c2" in T:
        kinds.append(("c2", "공단부담"))
    for k, name in (("c3", "전액본인"), ("c4", "선택진료"), ("c5", "비급여")):
        if k in ok and k in T:
            kinds.append((k, name))
    rng.shuffle(kinds)
    for col, name in kinds:
        cands = [c for c in I[col] if c["amount"] >= 1000]
        rng.shuffle(cands)
        for it in cands:
            for _ in range(5):
                new = new_amount(it["amount"], rng)
                if not new:
                    break
                d = new - it["amount"]
                patient = col != "c2"   # 공단부담 말고는 환자가 내는 돈이 바뀜
                if patient and d % 100:
                    continue
                ch = {id(it): (it, d), id(T[col]): (T[col], d)}
                desc = [name, {"c1": "①", "c2": "②", "c3": "③", "c4": "④", "c5": "⑤"}[col]]
                if "⑦" in ok:
                    ch[id(R["r7"])] = (R["r7"], d); desc.append("⑦")
                if patient:
                    if "⑧" in ok:
                        ch[id(R["r8"])] = (R["r8"], d); desc.append("⑧")
                    elif "r8" in R:
                        break   # ⑧이 있는데 식을 확인 못 하면 맞출 수 없음
                    if "⑩" in ok:
                        ch[id(R["r10"])] = (R["r10"], d); desc.append("⑩")
                    elif "r10" in R:
                        break
                    if "⑪" in ok:
                        pays = [R[k] for k in ("cash", "card", "cashr") if k in R and R[k]["amount"] + d >= 0 and R[k]["amount"] > 0]
                        if not pays:
                            break
                        p = max(pays, key=lambda c: c["amount"])
                        ch[id(p)] = (p, d); ch[id(R["paid"])] = (R["paid"], d); desc.append("⑪")
                    elif "paid" in R:
                        break
                pairs, good = [], True
                for c, dv in ch.values():
                    nv = c["amount"] + dv
                    if nv < 0 or len(str(nv)) != len(str(c["amount"])):
                        good = False
                        break
                    pairs.append((c, fmt(c["amount"]), fmt(nv)))
                if good:
                    yield pairs, "진료비서식 " + "".join(desc)
                    break


# ---------- 구조로 찾기 (10/23: 사진 문서에서는 한국어 OCR이 칸 이름을 자주 틀리게 읽어서 ('합계' -> '아계'),
#            숫자 칸이 줄·열로 늘어선 모양과 서식 식으로 각 칸을 정한다. 칸 이름은 있으면 참고만) ----------

def _right(c):
    """금액의 오른쪽 끝 (오른쪽 정렬 기준). '6,000②'처럼 뒤에 붙은 글자는 그만큼 뺀다."""
    t = c.get("raw", c["text"])
    m = re.search(r"[\d,]+(?=[^\d,]*$)", t)
    if m and len(t) > 0 and m.end() < len(t):
        return c["x"] + c["w"] * m.end() / len(t)
    return c["x"] + c["w"]


def _clusters(vals, tol=10):
    vals = sorted(vals)
    groups = []
    for v in vals:
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def find_layout_struct(nums, ko=None):
    """숫자 칸만으로 서식 찾기. 예전 서식(열 5개)과 새 서식(열 4개). 아니면 None."""
    am = [c for c in nums if c["amount"] is not None]
    if len(am) < 20:
        return None
    for c in am:
        c["_r"] = _right(c)
    groups = _clusters([c["_r"] for c in am])
    cols = [(sum(g) / len(g), len(g)) for g in groups if len(g) >= 5]
    if len(cols) < 5:
        return None
    # 본 표: 숫자가 많은 열(0이 줄줄이). 금액산정 칸: 그 오른쪽에 떨어진 열
    big = [c for c in cols if c[1] >= 8]
    if len(big) < 4:
        return None
    big.sort()
    # 가장 오른쪽 열이 앞 열들과 많이 떨어져 있으면 금액산정 칸
    gaps = [b[0] - a[0] for a, b in zip(big, big[1:])]
    med = sorted(gaps)[len(gaps) // 2] if gaps else 0
    main = [big[0]]
    for g, c in zip(gaps, big[1:]):
        if g > 2.2 * med:
            break
        main.append(c)
    if len(main) not in (4, 5):
        return None
    version = "예전" if len(main) == 5 else "새"
    mx = [m[0] for m in main]
    panel_x = [c[0] for c in cols if c[0] > mx[-1] + 1.5 * med]
    tol = max(10, 0.35 * med)

    def col_of(c):
        i = min(range(len(mx)), key=lambda i: abs(mx[i] - c["_r"]))
        return i if abs(mx[i] - c["_r"]) < tol else None
    grid = [(c, col_of(c)) for c in am]
    grid = [(c, k) for c, k in grid if k is not None]
    panel = sorted([c for c in am if any(abs(c["_r"] - p) < tol for p in panel_x)], key=lambda c: c["y"])
    # 줄 묶기
    rows = []
    for c, k in sorted(grid, key=lambda ck: _cy(ck[0])):
        if rows and abs(_cy(c) - rows[-1]["y"]) < 0.6 * c["h"]:
            rows[-1]["c"][k] = c
        else:
            rows.append({"y": _cy(c), "c": {k: c}})
    n = len(mx)
    best = None
    for ri in range(len(rows) - 1, 1, -1):
        tr = rows[ri]["c"]
        items = {k: [r["c"][k] for r in rows[:ri] if k in r["c"]] for k in range(n)}
        s = {k: sum(c["amount"] for c in items[k]) for k in range(n)}
        tv = {k: (tr[k]["amount"] if k in tr else (0 if s[k] == 0 else None)) for k in range(n)}
        if tv[0] is None or tv[1] is None:
            continue
        score = 0
        if tv[0] + tv[1] == s[0] + s[1] and s[0] + s[1] > 0:
            score += 2
        for k in range(2, n):
            if tv[k] is not None and tv[k] == s[k]:
                score += 1
        if score >= 3 and (best is None or score > best[0]):
            best = (score, ri, items, s, tr, tv)
    if not best:
        return None
    _, ri, items, s, tr, tv = best
    names = ["c1", "c2", "c3", "c4", "c5"] if n == 5 else ["c1", "c2", "c3", "c5"]   # 새 서식 비급여는 c5 자리로
    L = {"version": version, "order": names,
         "items": {names[k]: items[k] for k in range(n)},
         "totals": {names[k]: tr[k] for k in range(n) if k in tr},
         "tv": {names[k]: tv[k] for k in range(n)}, "s": {names[k]: s[k] for k in range(n)}}
    # 상한액 초과금: 합계 줄 바로 아래 줄 첫 열
    cap = rows[ri + 1]["c"].get(0) if ri + 1 < len(rows) else None
    L["cap"] = cap
    L["right"] = _assign_panel(panel, L, version)
    L["ok"] = check_struct(L)
    return L


def _assign_panel(panel, L, version):
    tv = L["tv"]
    cap = L["cap"]["amount"] if L.get("cap") else 0
    allv = [v for v in tv.values()]
    if any(v is None for v in allv):
        return {}
    total = sum(allv)
    pt = tv["c1"] - cap + sum(tv[k] for k in tv if k not in ("c1", "c2"))
    R = {}
    vals = panel
    i7 = next((i for i, c in enumerate(vals) if c["amount"] == total), None)
    if i7 is None:
        return R
    R["r7"] = vals[i7]
    rest = vals[i7 + 1:]
    if version == "새":   # ⑦ 공단부담 총액 = ② + ⑤(상한액)
        j = next((i for i, c in enumerate(rest) if c["amount"] == tv["c2"] + cap), None)
        if j is not None:
            R["gong"] = rest[j]
            rest = rest[j + 1:]
    j8 = next((i for i, c in enumerate(rest) if c["amount"] in (pt, pt // 100 * 100, pt // 10 * 10)), None)
    if j8 is None:
        return R
    R["r8"] = rest[j8]
    R["_r8mode"] = "내림" if rest[j8]["amount"] != pt else "합"
    rest = rest[j8 + 1:]
    if len(rest) >= 2 and rest[1]["amount"] == R["r8"]["amount"] - rest[0]["amount"]:
        R["r9"], R["r10"] = rest[0], rest[1]
        rest = rest[2:]
    # 납부 수단 + 합계: 뒤에서 '앞 몇 개의 합 = 그 다음 값' 이고 그 값이 ⑩과 같으면
    for m in (3, 2, 1):
        if len(rest) > m and sum(c["amount"] for c in rest[:m]) == rest[m]["amount"] and \
                ("r10" not in R or rest[m]["amount"] == R["r10"]["amount"]):
            R["pays"] = rest[:m]
            R["paid"] = rest[m]
            break
    return R


def check_struct(L):
    ok = set()
    s, tv, R, T = L["s"], L["tv"], L["right"], L["totals"]
    if tv["c1"] + tv["c2"] == s["c1"] + s["c2"]:
        ok.add("①+②")
    for k in ("c3", "c4", "c5"):
        if k in tv and tv[k] == s[k] and k in T:
            ok.add(k)
    if "r7" in R:
        ok.add("⑦")
    if "r8" in R:
        ok.add("⑧")
    if "r10" in R:
        ok.add("⑩")
    if "paid" in R:
        ok.add("⑪")
    if "gong" in R:
        ok.add("공단총액")
    return ok


def plans_struct(L, rng, new_amount, fmt):
    """구조로 찾은 서식에서 바꿀 칸 목록 후보. plans()와 같은 모양."""
    T, I, R, ok = L["totals"], L["items"], L["right"], L["ok"]
    if "⑦" not in ok:
        return
    kinds = []
    if "①+②" in ok and "c1" in T:
        kinds.append(("c1", "본인부담"))
    if "①+②" in ok and "c2" in T:
        kinds.append(("c2", "공단부담"))
    for k, name in (("c3", "전액본인"), ("c4", "선택진료"), ("c5", "비급여")):
        if k in ok:
            kinds.append((k, name))
    rng.shuffle(kinds)
    sym = {"c1": "①", "c2": "②", "c3": "③", "c4": "④", "c5": "⑤" if L["version"] == "예전" else "④"}
    for col, name in kinds:
        cands = [c for c in I[col] if c["amount"] >= 1000]
        rng.shuffle(cands)
        for it in cands:
            for _ in range(5):
                new = new_amount(it["amount"], rng)
                if not new:
                    break
                d = new - it["amount"]
                patient = col != "c2"
                if patient and d % 100:
                    continue
                if col == "c1" and L["s"]["c1"] != T["c1"]["amount"] and d % 100:
                    continue
                ch = {id(it): (it, d), id(T[col]): (T[col], d), id(R["r7"]): (R["r7"], d)}
                desc = [name, sym[col], "총액"]
                if col == "c2" and "gong" in R:
                    ch[id(R["gong"])] = (R["gong"], d); desc.append("공단총액")
                if patient:
                    if "r8" not in R or "r10" not in R:
                        break
                    ch[id(R["r8"])] = (R["r8"], d); ch[id(R["r10"])] = (R["r10"], d); desc += ["환자부담", "납부할"]
                    if "paid" in R:
                        pays = [p for p in R["pays"] if p["amount"] > 0 and p["amount"] + d >= 0]
                        if not pays:
                            break
                        p = max(pays, key=lambda c: c["amount"])
                        ch[id(p)] = (p, d); ch[id(R["paid"])] = (R["paid"], d); desc.append("납부")
                pairs, good = [], True
                for c, dv in ch.values():
                    nv = c["amount"] + dv
                    if nv < 0 or len(str(nv)) != len(str(c["amount"])):
                        good = False
                        break
                    pairs.append((c, fmt(c["amount"]), fmt(nv)))
                if good:
                    yield pairs, f"진료비서식({L['version']}) " + "→".join(desc)
                    break


# ---------- 약제비 계산서·영수증 (약국) ----------
# 한 열에 위에서부터: 약제비 총액 = 본인부담금① + 보험자부담금② + 비급여(전액본인부담)③,
# 총수납금액 = ① + ③ (10원·100원 미만 내림인 경우 있음), 납부 수단(현금영수증, 카드, 현금 등)의 합 = 합계 = 총수납금액

def find_pharmacy(nums):
    """한 열에서 [총액, 항목 2~3개(합 = 총액), 납부 수단 0~3개, 합계(= 환자 몫 항목 합, 10·100원 내림 가능)]를 찾는다.
    항목 중 합계에 들어가는 것은 환자 몫(본인부담①·비급여③), 아닌 것은 보험자부담②. (사진에서 0이 안 읽히는 칸이 있어
    항목은 2개도 허용. 환자 몫 항목은 어느 쪽이든 바꿀 때 같이 바뀌는 칸이 같음)"""
    from itertools import combinations
    am = [c for c in nums if c["amount"] is not None]
    for c in am:
        c["_r"] = _right(c)
    for g in _clusters([c["_r"] for c in am]):
        if len(g) < 5:
            continue
        lo, hi = min(g) - 1, max(g) + 1
        col = sorted([c for c in am if lo <= c["_r"] <= hi], key=lambda c: c["y"])
        v = [c["amount"] for c in col]
        for i in range(len(v) - 3):
            if v[i] < 1000:
                continue
            for n in (3, 2):
                items = col[i + 1:i + 1 + n]
                if len(items) < n or sum(c["amount"] for c in items) != v[i]:
                    continue
                after = col[i + 1 + n:]
                for m in (3, 2, 1):
                    if len(after) <= m:
                        continue
                    pays, paid = after[:m], after[m]
                    if sum(c["amount"] for c in pays) != paid["amount"] or paid["amount"] == 0:
                        continue
                    for r in range(1, n + 1):
                        for S in combinations(items, r):
                            w = sum(c["amount"] for c in S)
                            if paid["amount"] in (w, w // 10 * 10, w // 100 * 100):
                                return {"type": "약제비", "total": col[i], "items": items, "patient": list(S),
                                        "insurer": [c for c in items if c not in S], "pays": pays, "paid": paid,
                                        "ok": {"총액", "납부"}}
    return None


def plans_pharmacy(P, rng, new_amount, fmt):
    kinds = [(c, "환자몫") for c in P["patient"] if c["amount"] >= 1000] + \
            [(c, "보험자부담") for c in P["insurer"] if c["amount"] >= 1000]
    rng.shuffle(kinds)
    for it, name in kinds:
        for _ in range(5):
            new = new_amount(it["amount"], rng)
            if not new:
                break
            d = new - it["amount"]
            patient = name == "환자몫"
            if patient and d % 100:
                continue
            ch = [(it, d), (P["total"], d)]
            desc = [name, "총액"]
            if patient:
                pays = [p for p in P["pays"] if p["amount"] > 0 and p["amount"] + d >= 0]
                if not pays:
                    break
                p = max(pays, key=lambda c: c["amount"])
                ch += [(p, d), (P["paid"], d)] if p is not P["paid"] else [(p, d)]
                desc.append("납부")
            pairs, good = [], True
            for c, dv in ch:
                nv = c["amount"] + dv
                if nv < 0 or len(str(nv)) != len(str(c["amount"])):
                    good = False
                    break
                pairs.append((c, fmt(c["amount"]), fmt(nv)))
            if good:
                yield pairs, "약제비서식 " + "→".join(desc)
                break
