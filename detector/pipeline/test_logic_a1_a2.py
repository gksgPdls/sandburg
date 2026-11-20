"""인쇄된 산식(A1) · 서식 무관 표 합계(A2) 시험 — 외부 호출 없음. 가짜 VLM 판독으로 판독 → 로직 판정까지 돌려 기대와 맞는지 본다.

    .venv/bin/python detector/pipeline/test_logic_a1_a2.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detector.pipeline import engines as EN  # noqa: E402
from detector.pipeline import printed_formula as PF  # noqa: E402
from detector.pipeline import vlm_read as V  # noqa: E402

BASE = {"진료기간": {"시작": "2025-03-04", "끝": "2025-03-04"}, "발행일": "2025-03-04", "행": [],
        "합계": {"본인부담": 0, "공단부담": 0, "전액본인부담": 0, "비급여": 160000}, "상한액초과금": 0,
        "금액산정": {"진료비총액": 160000, "공단부담총액": 0, "환자부담총액": 160000, "이미납부한금액": 0, "납부할금액": 160000},
        "납부": {"카드": 160000, "현금영수증": 0, "현금": 0, "합계": 160000}}
# 개정 서식: ⑥ 진료비 총액 = ①+②+③+④
NEW = [("①", "본인부담금", None, 0), ("②", "공단부담금", None, 0), ("③", "전액본인부담", None, 0), ("④", "비급여", None, 160000),
       ("⑤", "상한액 초과금", None, 0), ("⑥", "진료비 총액", "①+②+③+④", 160000), ("⑦", "공단부담 총액", "②+⑤", 0),
       ("⑧", "환자부담 총액", "(①-⑤)+③+④", 160000), ("⑨", "이미 납부한 금액", None, 0), ("⑩", "납부할 금액", "⑧-⑨", 160000)]
# 옛 서식: ⑦ 진료비 총액 = ①+②+③+④+⑤ (④ 선택진료료 ⑤ 선택진료료 이외 ⑥ 상한액)
OLD = [("①", "본인부담금", None, 0), ("②", "공단부담금", None, 0), ("③", "전액본인부담", None, 0), ("④", "선택진료료", None, 0),
       ("⑤", "선택진료료 이외", None, 160000), ("⑥", "상한액 초과금", None, 0), ("⑦", "진료비 총액", "①+②+③+④+⑤", 160000),
       ("⑧", "환자부담 총액", "①-⑥+③+④+⑤", 160000), ("⑨", "이미 납부한 금액", None, 0), ("⑩", "납부할 금액", "⑧-⑨", 160000)]
# 옛 서식을 새 서식 이름으로 받으면 VLM 이 ⑦(진료비 총액) 값을 '공단부담총액'에 넣는다 (실제 서류 NM_009 에서 본 일)
OLD_BASE = copy.deepcopy(BASE)
OLD_BASE["금액산정"].update({"공단부담총액": 160000, "환자부담총액": 0})


def cells(rows, **change):
    out = [{"번호": a, "이름": b, "식": c, "값": d} for a, b, c, d in rows]
    for k, (field, v) in change.items():
        out[int(k[1:])][field] = v
    return out


def judge(j):
    rd, notes = V.to_reading(j, "영수증")
    lg = EN.logic(rd)
    return lg["level"], [f["title"] for f in lg["findings"] if f["rule"] == "A1"], notes


CASES = [  # (이름, 판독, 기대 판정, 기대 A1 제목에 들어갈 말들)
    ("개정 서식 · 정상", {**BASE, "번호칸": cells(NEW)}, "정상", []),
    ("개정 서식 · ④ 비급여만 고침", {**BASE, "번호칸": cells(NEW, i3=("값", 190000))}, "의심", ["⑥ 진료비 총액", "⑧ 환자부담 총액"]),
    ("옛 서식 · 인쇄된 식 없이 (예전 방식)", copy.deepcopy(OLD_BASE), "의심", ["⑦ 공단부담 총액"]),
    ("옛 서식 · 정상 (인쇄된 식)", {**OLD_BASE, "번호칸": cells(OLD)}, "정상", []),
    ("옛 서식 · ⑦ 진료비 총액 고침", {**OLD_BASE, "번호칸": cells(OLD, i6=("값", 190000))}, "의심", ["⑦ 진료비 총액"]),
    ("옛 서식 · ⑧ 식이 깨져 읽힘 → 표준식", {**OLD_BASE, "번호칸": cells(OLD, i7=("식", "(①⑥)+③+④+⑤)"))}, "정상", []),
    ("옛 서식 · ⑧ 식 깨짐 + ⑤ 고침", {**OLD_BASE, "번호칸": cells(OLD, i7=("식", "(①⑥)+③+④+⑤)"), i4=("값", 100000))}, "의심",
     ["⑦ 진료비 총액", "서식 산식 불일치: ⑧"]),
    ("빈 칸이 null 로 읽힘 (0 으로 봄)", {**BASE, "번호칸": cells(NEW, i2=("값", None), i8=("값", None))}, "정상", []),
    ("번호칸은 있는데 식을 하나도 못 읽음", {**BASE, "번호칸": [dict(c, 식=None) for c in cells(NEW)]}, "정상", []),
]


def main():
    ok = 0
    print(f"{'시험':34s} {'기대':5s} {'결과':5s}  A1 근거")
    for name, j, want, words in CASES:
        level, titles, _ = judge(copy.deepcopy(j))
        passed = level == want and all(any(w in t for t in titles) for w in words) and (bool(words) or not titles)
        ok += passed
        print(f"{'통과' if passed else '실패'} {name:31s} {want:5s} {level:5s}  {' / '.join(t[:40] for t in titles) or '-'}")
    # 서식 무관 표 합계 (A2): 옛 서식 열(선택진료료 · 선택진료료 이외)도 열 이름 해석 없이 숫자만으로
    from detector.pipeline import table_sums as TS
    OLD_T = {"열": ["본인부담금", "공단부담금", "전액본인부담", "선택진료료", "선택진료료 이외"],
             "줄": [{"항목": "진찰료", "값": [5050, 11800, None, None, None]},
                   {"항목": "주사료 약품비", "값": [None, None, None, None, 120000]},
                   {"항목": "검사료", "값": [1220, 2850, None, None, 40000]}],
             "합계": [6200, 14720, 0, 0, 160000]}   # 본인부담 절사 70원이 공단부담으로 넘어감
    NEW_T = {"열": ["본인부담금", "공단부담금", "전액본인부담", "비급여", "소계"],
             "줄": [{"항목": "진찰료", "값": [5000, 11800, None, None, 16800]},
                   {"항목": "처치 및 수술료", "값": [3000, 7000, None, 90000, 100000]}],
             "합계": [8000, 18800, 0, 90000, 116800]}
    tcases = [("옛 서식 표 · 정상 (외래 100원 절사)", OLD_T, False, []),
              ("옛 서식 표 · 선택진료료 이외 합계 고침", {**OLD_T, "합계": [6200, 14720, 0, 0, 190000]}, False, ["선택진료료 이외"]),
              ("옛 서식 표 · 입원이면 절사 없음 (참고)", OLD_T, True, ["본인부담금 합계 줄이"]),
              ("본인+공단 합이 어긋남 (공단 합계 고침)", {**OLD_T, "합계": [6200, 24720, 0, 0, 160000]}, False, ["본인부담금+공단부담금"]),
              ("개정 서식 표 · 소계 열 정상", NEW_T, False, []),
              ("개정 서식 표 · 비급여 줄 값 고침", {**NEW_T, "줄": [NEW_T["줄"][0], {"항목": "처치 및 수술료", "값": [3000, 7000, None, 150000, 100000]}]},
               False, ["비급여", "소계"]),
              ("합계 줄을 못 읽음 → 검사 안 함", {**OLD_T, "합계": [None] * 5}, False, None)]
    for name, t, inpatient, words in tcases:
        table = TS.parse(copy.deepcopy(t))
        if words is None:
            passed = table is None
            titles = []
        else:
            titles = [f["title"] for f in TS.check(table, inpatient)]
            passed = all(any(w in x for x in titles) for w in words) and (bool(words) or not titles)
        ok += passed
        print(f"{'통과' if passed else '실패'} 표 {name:30s} {' / '.join(x[:44] for x in titles) or '-'}")
    # 표를 읽었으면 엔진의 개정 서식 A2 대신 표 검사가 쓰인다 (판독 → 판정 끝까지)
    j = {**OLD_BASE, "번호칸": cells(OLD), "금액표": copy.deepcopy(OLD_T)}
    j["금액표"]["합계"] = [6200, 14720, 0, 0, 190000]
    level, _, _ = judge(j)
    rd, _ = V.to_reading(j, "영수증")
    a2 = [f["title"] for f in EN.logic(rd)["findings"] if f["rule"] == "A2"]
    passed = level == "의심" and a2 and all("열 합계 불일치" in x for x in a2)
    ok += passed
    print(f"{'통과' if passed else '실패'} 판정까지 · 옛 서식 표 합계 고침 → {level} {a2}")
    # 식 계산기 자체
    v = {"①": 10, "②": 20, "③": 3, "④": 4, "⑤": 5, "⑥": 1}
    calc = [("①+②+③+④", 37), ("(①-⑤)+③+④", 12), ("①-⑥+③+④+⑤", 21), ("(①⑥)+③", None), ("①+⑫", None)]
    for f, want in calc:
        got = PF.evaluate(f, v)
        passed = got == want
        ok += passed
        print(f"{'통과' if passed else '실패'} 계산 {f:28s} {str(want):5s} {str(got):5s}")
    n = len(CASES) + len(calc) + len(tcases) + 1
    print(f"\n{ok}/{n} 통과")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
