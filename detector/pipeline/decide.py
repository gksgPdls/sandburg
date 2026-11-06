"""두 엔진 결과 → 최종 판정. 둘 중 하나라도 걸리면 잡는 OR 구조이고, 근거 세기에 따라 등급만 나눈다.

등급형 OR (기본, "graded")
    로직 확정·강함                         → 의심   (정상 서류에선 나올 수 없는 계산·날짜 모순)
    픽셀 2% 기준 넘음 (strong)              → 의심
    로직 걸림 + 픽셀 걸림 (둘 다 약해도)      → 의심   (서로 다른 근거가 같은 서류를 가리킴)
    로직만 약하게 / 픽셀만 10% 기준 넘음      → 확인필요
    둘 다 안 걸림                           → 정상
순수 OR ("or")       : 어느 엔진이든 의심·확인필요면 의심
로직만 / 픽셀만       : 비교용 (한 엔진 혼자)

같은 문서별 결과(문서별.jsonl)로 방식을 바꿔 다시 계산할 수 있다 → run.py --다시판정
"""
MODES = ("graded", "or", "logic", "pixel")
FLAG = ("의심", "확인필요")


def decide(logic, pixel, mode="graded"):
    """→ (판정, 이유 목록)."""
    lf, pf = logic["level"] in FLAG, pixel["level"] in FLAG
    why = []
    if lf:
        why.append(f"로직 {logic['level']}")
    if pf:
        why.append(f"픽셀 {pixel['level']}")
    if mode == "logic":
        return logic["level"] if logic["level"] != "보류" else "확인필요", why
    if mode == "pixel":
        return (pixel["level"] if pixel["level"] != "보류" else "정상"), why
    if mode == "or":
        return ("의심" if lf or pf else "정상"), why
    if logic["strong"] or pixel["strong"] or (lf and pf):
        return "의심", why
    if lf or pf:
        return "확인필요", why
    return "정상", why
