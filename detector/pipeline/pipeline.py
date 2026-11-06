"""서류 한 장: 읽기는 VLM, 탐지는 로직 · 픽셀 두 엔진을 병렬로 → OR 판정.

    사진 ─┬─ VLM 읽기(OpenRouter) → Reading → 로직 엔진 ─┐
          └─ 픽셀 엔진 (읽기와 상관없이 바로 시작) ────────┴→ decide (등급형 OR) → 판정 · 근거 · 위치

    from combine.pipeline import inspect
    r = inspect("영수증.png", "영수증")          # 문서 종류: 영수증 / 진단서 / 자동(VLM 이 판별)
    r["판정"], r["이유"], r["위치"]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combine import decide as D  # noqa: E402
from combine import engines as E  # noqa: E402
from combine import vlm_read as V  # noqa: E402

CLASSIFY = """이 사진의 서류 종류를 하나만 골라라: "진료비 계산서·영수증", "진단서", "기타". JSON 하나로만: {"종류": "…"}"""
_PIXEL_POOL = ThreadPoolExecutor(2)       # 픽셀은 CPU 작업이라 동시에 2장까지만 (VLM 기다리는 동안 돌게)


def classify(image, model):
    r = V.ar.ask(V.MODELS.get(model, model), image, CLASSIFY, 1200, max_tokens=60)
    kind = str((r.get("json") or {}).get("종류", ""))
    return ("영수증" if "영수증" in kind else "진단서" if "진단서" in kind else "기타"), r.get("cost") or 0


def read_and_logic(image, doc, model, size, use_cache, locate):
    """VLM 읽기 → 로직. (로직 결과, 판독 기록)"""
    t0 = time.time()
    cost = 0
    if doc == "자동":
        doc, c = classify(image, model)
        cost += c
    if doc not in ("영수증", "진단서"):
        return ({"engine": "logic", "level": "확인필요", "strong": False, "findings": [], "boxes": [], "score": None,
                 "seconds": 0, "note": f"서류 종류를 모름 ({doc})"}, {"문서": doc, "비용": cost})
    try:
        r = V.read(image, doc, model, size, use_cache)
    except SystemExit:
        raise                                                         # 실서류 막기는 그대로 멈춤
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "json": None, "error": f"{type(e).__name__}: {e}"}
    cost += r.get("cost") or 0
    rec = {"문서": doc, "비용": cost, "업체": r.get("provider"), "캐시": r.get("cached"), "판독_원본": r.get("json"),
           "판독_오류": r.get("error")}
    if not r.get("json"):
        return ({"engine": "logic", "level": "확인필요", "strong": False, "findings": [], "boxes": [], "score": None,
                 "seconds": round(time.time() - t0, 2), "note": "VLM 판독 실패"}, rec)
    reading, notes = V.to_reading(r["json"], doc)
    rec["판독"] = reading.fields
    res = E.logic(reading)
    res["note"] = "; ".join(notes)
    if locate and res["fields"]:                                      # 걸린 칸만 위치를 한 번 더 묻는다
        try:
            pos, c = V.locate(image, res["fields"], model, size)
            rec["비용"] += c
            res["boxes"] = list(dict.fromkeys(pos.values()))
        except Exception as e:  # noqa: BLE001
            res["note"] += f"; 위치 묻기 실패 {type(e).__name__}"
    res["seconds"] = round(time.time() - t0, 2)
    return res, rec


def inspect(image, doc="자동", model="qwen27b", size=2500, mode="graded", use_cache=True, locate=True, pixel_thr=None,
            use_pixel=True):
    image = Path(image)
    V.oc.refuse_real(image)                                           # 픽셀도 돌리기 전에 실서류면 멈춤 (외부로 나가는 쪽 기준)
    t0 = time.time()
    pf = _PIXEL_POOL.submit(E.pixel, image, pixel_thr) if use_pixel else None
    logic, rec = read_and_logic(image, doc, model, size, use_cache, locate)
    pixel = pf.result() if pf else {"engine": "pixel", "level": "보류", "strong": False, "findings": [], "boxes": [],
                                     "score": None, "seconds": 0, "note": "끔"}
    verdict, why = D.decide(logic, pixel, mode)
    boxes = (logic["boxes"] if logic["level"] in D.FLAG else []) + (pixel["boxes"] if pixel["level"] in D.FLAG else [])
    return {"파일": str(image), "문서": rec.get("문서"), "판정": verdict, "이유": why, "위치": list(dict.fromkeys(boxes)),
            "logic": logic, "pixel": pixel, "판독": rec, "초": round(time.time() - t0, 2), "비용": rec.get("비용") or 0}
