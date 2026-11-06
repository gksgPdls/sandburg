"""합친 탐지기 결과(로직 · 픽셀 · 판정) → 실험 화면이 그리는 형식 (logic_detector/web 화면과 같은 모양).

    view(logic, pixel, 판정, 판독 칸 dict, 문서 종류, 메모) → {verdict, kind, engines, findings, focus, reading, notes}
"""
VERDICT = {"의심": "위변조 의심", "확인필요": "확인 필요", "정상": "정상"}
LEVEL = {"의심": "위변조 의심", "확인필요": "확인 필요", "정상": "정상", "보류": "보류"}


def _box(b):
    if isinstance(b, str):
        b = [int(float(v)) for v in b.split(",")]
    return [int(v) for v in b][:4]


def view(logic, pixel, verdict, fields, doc, note=""):
    findings = []
    for f in logic.get("findings", []):
        findings.append({k: f.get(k) for k in ("rule", "grade", "title", "detail", "expected", "actual", "source", "caveat")}
                        | {"boxes": [_box(b) for b in f.get("boxes") or []]})
    if pixel.get("level") in ("의심", "확인필요"):
        hot = [c for c in pixel.get("findings", []) if _box(c["bbox"]) and ",".join(map(str, _box(c["bbox"]))) in
               {",".join(map(str, _box(b))) for b in pixel.get("boxes", [])}] or pixel.get("findings", [])[:1]
        findings.append({"rule": "P-A11", "grade": "강함" if pixel.get("strong") else "참고",
                         "title": f"다시 그린 흔적이 있는 칸 {len(hot)}개 (문서 점수 {pixel.get('score')})",
                         "detail": " / ".join(str(c.get("why", "")) for c in hot)[:300],
                         "caveat": "" if pixel.get("strong") else "정상 서류 10% 기준만 넘음 — 로직도 걸리면 위변조 의심",
                         "boxes": [_box(c["bbox"]) for c in hot]})
    findings.sort(key=lambda f: ("확정", "강함", "참고").index(f["grade"]) if f["grade"] in ("확정", "강함", "참고") else 3)
    n_logic = len(logic.get("findings", []))
    engines = [
        {"name": "로직 (VLM 판독 → 규칙)", "level": LEVEL.get(logic.get("level"), logic.get("level")),
         "detail": f"걸린 규칙 {n_logic}개" if n_logic else "걸린 규칙 없음"},
        {"name": "픽셀 (다시 그린 칸)", "level": LEVEL.get(pixel.get("level"), pixel.get("level")),
         "detail": (pixel.get("note") or "") + (f" 문서 점수 {pixel['score']}" if pixel.get("score") is not None else "")},
    ]
    return {"verdict": VERDICT.get(verdict, verdict), "kind": {"영수증": "receipt", "진단서": "diagnosis"}.get(doc, doc),
            "engines": engines, "findings": findings, "focus": [],
            "reading": {k: {"value": v} for k, v in (fields or {}).items()},
            "notes": [n for n in (note or "").split("; ") if n]}
